"""Spatial and local Mamba blocks used by the restored myself decoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_cluster import knn


def spatial_slab_sort(xyz: torch.Tensor, slab_thickness: float, descending: bool = False):
    """Stable lexicographic slab ordering for [..., L, 3] coordinates."""
    if slab_thickness <= 0 or not math.isfinite(slab_thickness):
        raise ValueError("slab_thickness must be finite and positive")
    xyz = xyz.float()
    slab = torch.floor(xyz[..., 2] / slab_thickness).long()
    order = torch.argsort(xyz[..., 0], stable=True, descending=descending)
    for key in (xyz[..., 1], slab):
        order = order.gather(-1, torch.argsort(key.gather(-1, order), stable=True, descending=descending))
    return order


_spatial_sort = spatial_slab_sort


class MambaAggregation(nn.Module):
    """Local spatial feature aggregation replacing cross-attention (LaSSM §3.2).

    For each query the k spatially nearest superpoints are looked up and their
    features are aggregated with learned softmax kernel weights:

        k_weights = softmax(W_k @ q + b)          shape (Q, k)
        w         = Σ_k k_weights[q,k] * (q[q] ⊙ v[q,k])  shape (Q, d)
        output    = LayerNorm(dropout(W_o @ w) + query)

    When query/source XYZ positions are unavailable the layer falls back to a
    feature-space nearest-neighbour lookup (dot-product similarity), which
    preserves the kernel-aggregation mechanism without spatial grounding.

    Args:
        d_model (int): Feature dimension.
        k (int): Number of nearest neighbours per query.
        dropout (float): Dropout on the output projection.
    """

    def __init__(self, d_model: int, k: int = 8, dropout: float = 0.0):
        super().__init__()
        self.k = k
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Parameter(torch.empty(k, d_model))  # learned kernels
        self.w_b = nn.Parameter(torch.zeros(k))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w_k)

    def forward(self, query, src_feats, query_pos=None, src_pos=None,
                mask=None, neighbor_indices=None):
        """Forward pass.

        Args:
            query (Tensor): (Q, d) query features.
            src_feats (Tensor): (N, d) source (superpoint) features.
            query_pos (Tensor | None): (Q, 3) query 3-D positions.
            src_pos (Tensor | None): (N, 3) source 3-D positions.

        Returns:
            Tensor: (Q, d) updated query features.
        """
        shortcut = query
        k = min(self.k, src_feats.shape[0])
        if k == 0 or query.shape[0] == 0:
            return query

        # ── k-NN lookup ──────────────────────────────────────────────────
        if neighbor_indices is not None:
            idx = neighbor_indices[:, :k]
        elif query_pos is not None and src_pos is not None:
            with torch.no_grad():
                dists = torch.cdist(query_pos.float(), src_pos.float())
                idx = dists.topk(k, dim=-1, largest=False).indices
        else:
            # Feature-space fallback: similarity via dot product
            q_n = F.normalize(self.w_q(query), dim=-1)
            s_n = F.normalize(self.w_v(src_feats), dim=-1)
            dists = -torch.mm(q_n, s_n.T)
            idx = dists.topk(k, dim=-1, largest=False).indices

        if mask is not None:
            local_mask = mask.gather(1, idx).bool()
            local_mask &= ~local_mask.all(dim=1, keepdim=True)
        else:
            local_mask = None

        # ── project + gather ─────────────────────────────────────────────
        q   = self.w_q(query)              # (Q, d)
        v   = self.w_v(src_feats)          # (N, d)
        feat = v[idx]                      # (Q, k, d)
        q_exp = q.unsqueeze(1).expand(-1, k, -1)  # (Q, k, d)

        # ── learned softmax kernel weights ───────────────────────────────
        logits = F.linear(q, self.w_k[:k], self.w_b[:k])
        if local_mask is not None:
            logits = logits.masked_fill(local_mask, -torch.inf)
        k_w = F.softmax(logits, dim=-1)  # (Q, k)

        # ── weighted aggregation ─────────────────────────────────────────
        w = torch.einsum('qk,qkd->qd', k_w, q_exp * feat)           # (Q, d)

        out = self.dropout(self.w_o(w)) + shortcut
        return self.norm(out)


class SSMSpatialSelfLayer(nn.Module):
    """Spatial SSM replacing self-attention on queries (LaSSM §3.3).

    Faithful to the LaSSM paper's SSM class:

    1. Apply pre-norm to queries.
    2. Sort queries by two complementary spatial orderings (ascending and
       descending slab-order, approximating Hilbert + Hilbert-transposed).
    3. Stack both orderings as a batch: shape (2, Q, d).
    4. Pass through a **single shared Mamba instance** — both orderings use
       the same weights (batch dimension = orderings, identical to the paper).
    5. Unsort each ordering back to the original query positions.
    6. **Average** the two outputs (LaSSM: ``sum(paths) / len(paths)``).
    7. Add shortcut residual and apply final-norm.

    When query positions are unavailable the layer falls back to identity
    and reversed-index orderings (no spatial grounding, same shared-weight
    logic otherwise preserved).

    Args:
        d_model (int): Feature dimension.
        d_state (int): Mamba SSM hidden-state size.
        d_conv (int): Mamba depthwise-conv kernel width.
        expand (int): Mamba inner expansion factor.
        slab_thickness (float): Z-axis bin width for spatial slab ordering.
    """

    def __init__(self, d_model: int, d_state: int = 64,
                 d_conv: int = 4, expand: int = 2,
                 slab_thickness: float = 5.0):
        super().__init__()
        self.slab_thickness = slab_thickness
        self.pre_norm   = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)
        from mamba_ssm import Mamba
        # ONE shared Mamba — processes all orderings as the batch dimension,
        # matching the paper: self.layers[l](query) where query is (num_ord, Q, d).
        self.mamba = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, queries, query_xyzs=None):
        """Forward pass.

        Args:
            queries (List[Tensor]): len batch_size, each (Q_i, d_model).
            query_xyzs (List[Tensor] | None): len batch_size, each (Q_i, 3).
                When provided, queries are serialised spatially before scanning.

        Returns:
            List[Tensor]: Updated queries, same shapes as input.
        """
        outputs = []
        for i, q in enumerate(queries):
            # Keep the variable-length sample interface safe when a caller
            # supplies an empty query set (for example, a filtered scene).
            if q.shape[0] == 0:
                outputs.append(q)
                continue
            shortcut = q
            normed   = self.pre_norm(q)          # (Q, d)
            pos      = query_xyzs[i] if query_xyzs is not None else None

            if pos is not None and pos.shape[0] > 1:
                ord_a = _spatial_sort(pos, self.slab_thickness, descending=False)
                ord_b = _spatial_sort(pos, self.slab_thickness, descending=True)
            else:
                n     = q.shape[0]
                ord_a = torch.arange(n, device=q.device)
                ord_b = torch.arange(n - 1, -1, -1, device=q.device)

            inv_a = torch.argsort(ord_a)
            inv_b = torch.argsort(ord_b)

            # Stack both orderings as batch dim → (2, Q, d)
            # ONE shared Mamba scans both orderings simultaneously (same weights)
            stacked = torch.stack([normed[ord_a], normed[ord_b]], dim=0)
            # The CUDA Mamba kernels generally expect their parameter dtype;
            # cast only the SSM input and restore the query dtype afterwards.
            mamba_dtype = next(self.mamba.parameters()).dtype
            with torch.autocast(device_type=stacked.device.type, enabled=False):
                out = self.mamba(stacked.to(mamba_dtype).contiguous())
            out = out.to(normed.dtype)           # (2, Q, d)

            # Unsort each ordering back to original positions then average
            out_a = out[0][inv_a]                # (Q, d)
            out_b = out[1][inv_b]                # (Q, d)
            q_out = (out_a + out_b) / 2          # average — paper: sum/len

            q_out = self.final_norm(shortcut + q_out)
            outputs.append(q_out)
        return outputs


class SpatialMambaLayer(nn.Module):
    """Shared-weight forward/backward Mamba over spatially ordered queries."""

    def __init__(self, dim, dropout=0.0, d_state=64, d_conv=4, expand=2, slab_thickness=2.0):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except (ImportError, OSError) as exc:
            raise ImportError(f"SaTv2 Mamba decoder requires mamba_ssm.Mamba: {exc}") from exc
        self.slab_thickness = slab_thickness
        self.pre_norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, query_xyz=None):
        if not len(query):
            return query
        if query_xyz is None:
            query_xyz = query.new_zeros((len(query), 3))
        forward = spatial_slab_sort(query_xyz, self.slab_thickness)
        backward = spatial_slab_sort(query_xyz, self.slab_thickness, True)
        sequence = torch.stack((self.pre_norm(query)[forward], self.pre_norm(query)[backward]))
        with torch.autocast(device_type=sequence.device.type, enabled=False):
            output = self.mamba(sequence.to(next(self.mamba.parameters()).dtype).contiguous()).to(sequence.dtype)
        update = (output[0][forward.argsort()] + output[1][backward.argsort()]) * 0.5
        return self.norm(query + self.dropout(update.to(query.dtype)))


class LocalKnnAggregation(nn.Module):
    """Query-dependent weighted aggregation over XYZ KNN neighbors."""

    def __init__(self, dim, cross_k=32, dropout=0.0):
        super().__init__()
        self.cross_k = cross_k
        self.query_norm = nn.LayerNorm(dim)
        self.weight_logits = nn.Linear(dim, cross_k)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, source, query, source_xyz, query_xyz, mask=None, neighbor_indices=None):
        if not len(query) or not len(source):
            return query
        if neighbor_indices is None:
            k = min(self.cross_k, len(source_xyz))
            edges = knn(source_xyz.float().contiguous(), query_xyz.float().contiguous(), k)
            grouping = edges[0].argsort(stable=True)
            indices = edges[1, grouping].reshape(len(query), k).long()
        else:
            indices = neighbor_indices
        local_mask = None if mask is None else mask.gather(1, indices).bool()
        if local_mask is not None:
            local_mask &= ~local_mask.all(dim=1, keepdim=True)
        logits = self.weight_logits(self.query_norm(query))[:, : indices.shape[1]].float()
        if local_mask is not None:
            logits = logits.masked_fill(local_mask, -torch.inf)
        weights = logits.softmax(dim=-1)
        values = self.value_proj(source)[indices]
        update = (weights[..., None] * values.float()).sum(dim=1).to(query.dtype)
        return self.norm(query + self.dropout(self.output_proj(update).to(query.dtype)))


class LocalCrossMamba(nn.Module):
    """Shared Mamba over batched neighbor-neighbor-query local sequences."""

    def __init__(self, dim, cross_k=32, dropout=0.0, d_state=64, d_conv=4, expand=2, slab_thickness=2.0, use_relative_pos=True):
        super().__init__()
        from mamba_ssm import Mamba
        self.cross_k = cross_k
        self.slab_thickness = slab_thickness
        self.source_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.source_proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(dim, dim)
        self.relative_pos_mlp = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim)) if use_relative_pos else None
        self.cross_mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, source, query, source_xyz, query_xyz, mask=None, neighbor_indices=None):
        if not len(query) or not len(source):
            return query
        if neighbor_indices is None:
            k = min(self.cross_k, len(source_xyz))
            edges = knn(source_xyz.float().contiguous(), query_xyz.float().contiguous(), k)
            grouping = edges[0].argsort(stable=True)
            indices = edges[1, grouping].reshape(len(query), k).long()
        else:
            indices = neighbor_indices
        relative_xyz = source_xyz[indices].float() - query_xyz[:, None].float()
        neighbor = self.source_proj(self.source_norm(source))[indices]
        if self.relative_pos_mlp is not None:
            neighbor = neighbor + self.relative_pos_mlp(relative_xyz.to(neighbor.dtype))
        local_mask = None if mask is None else mask.gather(1, indices).bool()
        if local_mask is not None:
            local_mask &= ~local_mask.all(dim=1, keepdim=True)
            neighbor = neighbor.masked_fill(local_mask[..., None], 0)
        forward = spatial_slab_sort(relative_xyz, self.slab_thickness)
        backward = spatial_slab_sort(relative_xyz, self.slab_thickness, True)
        query_token = self.query_proj(self.query_norm(query))[:, None].to(neighbor.dtype)
        sequence = torch.cat((torch.cat((neighbor.gather(1, forward[..., None].expand_as(neighbor)), query_token), 1), torch.cat((neighbor.gather(1, backward[..., None].expand_as(neighbor)), query_token), 1)))
        with torch.autocast(device_type=sequence.device.type, enabled=False):
            output = self.cross_mamba(sequence.to(next(self.cross_mamba.parameters()).dtype).contiguous()).to(sequence.dtype)
        q = len(query)
        update = (output[:q, -1] + output[q:, -1]) * 0.5
        return self.norm(query + self.dropout(self.output_proj(update).to(query.dtype)))
