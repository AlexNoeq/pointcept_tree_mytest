"""Pointcept implementation of SegmentAnyTreeV2.

The implementation follows Sections 3.3--3.8 of the paper.  PTv3 remains a
regular Pointcept backbone; this module implements the semantic head, ISA
embedding/FPS, tree-only masked transformer decoder, curriculum, and losses.
"""

from __future__ import annotations

from functools import partial
import math

import torch
import torch.nn as nn
from torch_cluster import fps as torch_cluster_fps
from torch_cluster import knn

from .mamba_layers import MambaAggregation, SSMSpatialSelfLayer, LocalCrossMamba

from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point

from .loss import SegmentAnyTreeV2Loss
from .nms import greedy_mask_nms


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor, pos):
        if pos is None:
            return tensor
        return tensor + pos.to(device=tensor.device, dtype=tensor.dtype)

    def forward(self, source, query, attn_masks=None, pe=None, query_pe=None):
        """Apply cross-attention to a list of variable-length samples.

        Point crops in a batch do not necessarily contain the same number of
        tree points, so padding them only for attention would make the mask
        logic unnecessarily complicated.  The list interface also mirrors
        the positional-encoding aware decoder used by SGIFormer.
        """

        if len(source) != len(query):
            raise ValueError("source and query must contain the same batch size")
        if attn_masks is not None and len(attn_masks) != len(source):
            raise ValueError("attn_masks must contain one mask per sample")
        if pe is not None and len(pe) != len(source):
            raise ValueError("pe must contain one source encoding per sample")
        if query_pe is not None and len(query_pe) != len(query):
            raise ValueError("query_pe must contain one encoding per sample")

        outputs = []
        for i in range(len(source)):
            q_pos = query_pe[i] if query_pe is not None else None
            source_pos = pe[i] if pe is not None else None
            q = self.with_pos_embed(query[i], q_pos)
            k = self.with_pos_embed(source[i], source_pos)
            v = source[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(q, k, v, attn_mask=attn_mask)
            output = self.dropout(output) + query[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        if pos is None:
            return tensor
        return tensor + pos.to(device=tensor.device, dtype=tensor.dtype)

    def forward(self, query, pe=None):
        """Apply self-attention independently to variable-length samples."""

        if pe is not None and len(pe) != len(query):
            raise ValueError("pe must contain one encoding per sample")
        outputs = []
        for i, sample in enumerate(query):
            pos = pe[i] if pe is not None else None
            q = k = self.with_pos_embed(sample, pos)
            update, _ = self.attention(q, k, sample, need_weights=False)
            outputs.append(self.norm(sample + self.dropout(update)))
        return outputs


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "relu",
    ):
        super().__init__()
        activation_fn = activation_fn.lower()
        if activation_fn == "relu":
            activation = nn.ReLU(inplace=True)
        elif activation_fn == "gelu":
            activation = nn.GELU()
        else:
            raise ValueError(
                "activation_fn must be either 'relu' or 'gelu', "
                f"got {activation_fn!r}"
            )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query) -> list[torch.Tensor]:
        """Apply the feed-forward block independently to each sample."""

        return [self.norm(sample + self.mlp(sample)) for sample in query]


class CoordinatePositionEncoding(nn.Module):
    """Deterministic sine/cosine encoding for XYZ point coordinates.

    The decoder works on a list of point crops, therefore the encoding is
    generated per crop after normalizing its coordinates to ``[0, 1]``.  The
    implementation intentionally has no trainable parameters and supports any
    even decoder dimension used by the attention heads.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("position encoding dimension must be positive")
        self.dim = dim

    def forward(
        self,
        coordinates: torch.Tensor,
        input_range: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
            raise ValueError("coordinates must have shape [N, 3]")
        if coordinates.shape[0] == 0:
            return coordinates.new_empty((0, self.dim), dtype=torch.float32)

        xyz = coordinates.float()
        if input_range is None:
            xyz_min = xyz.amin(dim=0, keepdim=True)
            xyz_max = xyz.amax(dim=0, keepdim=True)
        else:
            xyz_min, xyz_max = input_range
            xyz_min = xyz_min.to(device=xyz.device, dtype=xyz.dtype).reshape(1, 3)
            xyz_max = xyz_max.to(device=xyz.device, dtype=xyz.dtype).reshape(1, 3)
        xyz = (xyz - xyz_min) / (xyz_max - xyz_min).clamp_min(1e-6)

        # Each coordinate contributes sine and cosine features at multiple
        # frequencies.  Truncating/padding makes this valid for dim=8 in the
        # lightweight tests as well as the normal dim=128 configuration.
        frequency_count = max(1, math.ceil(self.dim / 6))
        frequencies = (2.0 ** torch.arange(
            frequency_count, device=xyz.device, dtype=xyz.dtype
        )) * torch.pi
        phase = xyz.unsqueeze(-1) * frequencies
        encoding = torch.cat((phase.sin(), phase.cos()), dim=-1).flatten(1)
        if encoding.shape[-1] < self.dim:
            encoding = torch.nn.functional.pad(
                encoding, (0, self.dim - encoding.shape[-1])
            )
        return encoding[:, : self.dim]


class SegmentAnyTreeV2Decoder(nn.Module):
    """Tree-only, asymmetric mask decoder from Section 3.6."""

    def __init__(
        self,
        in_channels: int = 128,
        dim: int | None = None,
        d_model: int | None = None,
        num_queries: int = 400,
        num_layers: int = 6,
        num_heads: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        activation_fn: str = "relu",
    ):
        super().__init__()
        # ``dim`` is retained as a backwards-compatible alias.  New configs
        # should use ``d_model`` so the decoder width is independent from the
        # backbone output channels.
        if d_model is None:
            d_model = 128 if dim is None else dim
        if d_model <= 0:
            raise ValueError("decoder d_model must be positive")
        self.d_model = d_model
        self.dim = d_model  # compatibility for callers using decoder.dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.position_encoding = CoordinatePositionEncoding(d_model)

        # The paper specifies separate asymmetric pathways and describes the
        # mask pathway as the deeper one with an intermediate LayerNorm.  It
        # does not publish the exact projection layer sequence, so it remains
        # isolated here and can be changed without touching decoder logic.
        self.attention_projection = nn.Linear(in_channels, d_model)
        self.query_projection = (
            nn.Identity()
            if in_channels == d_model
            else nn.Linear(in_channels, d_model)
        )
        self.mask_projection = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )
        self.learned_queries = nn.Embedding(num_queries, d_model)
        self.learned_query_pos = nn.Embedding(num_queries, d_model)

        self.cross_attention = nn.ModuleList(
            self._make_cross_layer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        )
        self.self_attention = nn.ModuleList(
            self._make_self_layer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        )
        self.feed_forward = nn.ModuleList(
            FFN(
                d_model=d_model,
                hidden_dim=hidden_dim,
                dropout=dropout,
                activation_fn=activation_fn,
            )
            for _ in range(num_layers)
        )

        self.output_norm = nn.LayerNorm(d_model)
        self.mask_head = nn.Linear(d_model, d_model)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 1)
        )

    def _make_cross_layer(self, dim, num_heads, dropout):
        return CrossAttentionLayer(dim, num_heads, dropout)

    def _make_self_layer(self, dim, num_heads, dropout):
        return SelfAttentionLayer(dim, num_heads, dropout)

    def _cross_step(self, layer, memory, query, mask, source_pos, query_pos,
                    tree_xyz, query_xyz, neighbor_indices):
        return self.cross_attention[layer]([memory], [query], [mask], pe=source_pos,
                                           query_pe=query_pos)[0]

    def _self_step(self, layer, query, query_pos, query_xyz):
        return self.self_attention[layer]([query], pe=query_pos)[0]

    def _neighbors(self, tree_xyz, query_xyz):
        return None

    def _query_anchors(self, features, coordinates, tree_xyz, tree_indices,
                       seed_indices, use_isa):
        if use_isa and len(tree_indices) and coordinates is not None:
            return coordinates[seed_indices]
        return features.new_zeros((self.num_queries, 3), dtype=torch.float32)

    def _predict(
        self, query: torch.Tensor, mask_memory: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        normalized = self.output_norm(query)
        mask_logits = self.mask_head(normalized) @ mask_memory.transpose(0, 1)
        score_logits = self.score_head(normalized).squeeze(-1)
        return {"mask_logits": mask_logits, "score_logits": score_logits}

    @staticmethod
    def _attention_mask(mask_logits: torch.Tensor) -> torch.Tensor:
        attention_mask = mask_logits.sigmoid() < 0.5
        fully_masked = attention_mask.all(dim=1)
        attention_mask[fully_masked] = False
        return attention_mask.detach()

    def _position_lists(
        self,
        tree_coordinates: torch.Tensor | None,
        seed_indices: torch.Tensor,
        tree_indices: torch.Tensor,
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
        """Build source and query positional encodings for one sample."""

        if tree_coordinates is None:
            return None, None

        source_coordinates = tree_coordinates[tree_indices]
        reference_coordinates = tree_coordinates.float()
        if reference_coordinates.shape[0] == 0:
            reference_coordinates = reference_coordinates.new_zeros((1, 3))
        reference_min = reference_coordinates.amin(dim=0)
        reference_max = reference_coordinates.amax(dim=0)
        input_range = (reference_min, reference_max)
        if source_coordinates.shape[0] == 0:
            # The empty-tree path feeds a single zero memory token to MHA.
            source_coordinates = tree_coordinates.new_zeros((1, 3))
        source_pos = self.position_encoding(source_coordinates, input_range)
        if seed_indices.numel() and (seed_indices >= 0).all():
            query_pos = self.position_encoding(
                tree_coordinates[seed_indices], input_range
            )
        else:
            # Learned queries have no point coordinate.  Their learned
            # positional vectors keep them distinguishable in self-attention.
            query_pos = self.learned_query_pos.weight
        return [source_pos], [query_pos]

    def forward_sample(
        self,
        features: torch.Tensor,
        embeddings: torch.Tensor,
        tree_mask: torch.Tensor,
        instance: torch.Tensor | None,
        use_isa: bool,
        coordinates: torch.Tensor | None = None,
    ) -> dict:
        tree_indices = tree_mask.nonzero(as_tuple=False).flatten()
        tree_features = features[tree_indices]
        tree_embeddings = embeddings[tree_indices]
        attention_memory = self.attention_projection(tree_features)
        mask_memory = self.mask_projection(tree_features)

        if use_isa and len(tree_indices) and self.num_queries:
            count = min(len(tree_embeddings), self.num_queries)
            seed_local = torch_cluster_fps(
                tree_embeddings.detach().float().contiguous(),
                ratio=float(count) / len(tree_embeddings),
                random_start=False,
            )[:count].long()
            if count < self.num_queries:
                seed_local = seed_local.repeat(
                    (self.num_queries + count - 1) // count
                )[: self.num_queries]
            seed_indices = tree_indices[seed_local]
            query = self.query_projection(features[seed_indices])
            seed_instance = (
                instance[seed_indices]
                if instance is not None
                else torch.full_like(seed_indices, -1)
            )
        else:
            seed_indices = torch.full(
                (self.num_queries,), -1, dtype=torch.long, device=features.device
            )
            seed_instance = seed_indices.clone()
            query = self.learned_queries.weight

        tree_xyz = (
            coordinates[tree_indices]
            if coordinates is not None
            else features.new_zeros((len(tree_indices), 3), dtype=torch.float32)
        )
        query_xyz = self._query_anchors(
            features, coordinates, tree_xyz, tree_indices, seed_indices, use_isa
        )
        source_pos_list, query_pos_list = self._position_lists(
            coordinates, seed_indices, tree_indices
        )
        neighbor_indices = self._neighbors(tree_xyz, query_xyz)

        if len(tree_indices) == 0:
            # MHA cannot consume an empty key sequence.  The dummy memory never
            # reaches a mask because all full-resolution logits remain -100.
            attention_memory = features.new_zeros((1, self.dim))
            initial = {
                "mask_logits": features.new_empty((self.num_queries, 0)),
                "score_logits": self.score_head(self.output_norm(query)).squeeze(-1),
            }
            layer_predictions = []
            for layer in range(self.num_layers):
                query = self._cross_step(
                    layer, attention_memory, query, None, source_pos_list,
                    query_pos_list, tree_xyz, query_xyz, neighbor_indices
                )
                query = self._self_step(layer, query, query_pos_list, query_xyz)
                query = self.feed_forward[layer]([query])[0]
                layer_predictions.append(
                    {
                        "mask_logits": features.new_empty((self.num_queries, 0)),
                        "score_logits": self.score_head(
                            self.output_norm(query)
                        ).squeeze(-1),
                    }
                )
        else:
            initial = self._predict(query, mask_memory)
            layer_predictions = []
            prediction = initial
            for layer in range(self.num_layers):
                query = self._cross_step(
                    layer, attention_memory, query,
                    self._attention_mask(prediction["mask_logits"]),
                    source_pos_list, query_pos_list, tree_xyz, query_xyz,
                    neighbor_indices
                )
                query = self._self_step(layer, query, query_pos_list, query_xyz)
                query = self.feed_forward[layer]([query])[0]
                prediction = self._predict(query, mask_memory)
                layer_predictions.append(prediction)

        final = layer_predictions[-1] if layer_predictions else initial
        return {
            **final,
            "aux_outputs": layer_predictions[:-1],
            "tree_indices": tree_indices,
            "seed_indices": seed_indices,
            "seed_instance": seed_instance,
            "tree_instance": (
                instance[tree_indices] if instance is not None else None
            ),
            "num_points": features.shape[0],
        }


class SegmentAnyTreeV2MambaDecoder(SegmentAnyTreeV2Decoder):
    """Unified Mamba decoder with aggregation or local Cross-Mamba."""

    def __init__(self, *args, cross_mode="aggregation", cross_k=32,
                 mamba_d_state=64, mamba_d_conv=4, mamba_expand=2,
                 slab_thickness=2.0, use_relative_pos=True, **kwargs):
        if cross_mode not in ("attention", "aggregation", "mamba"):
            raise ValueError("cross_mode must be attention, aggregation or mamba")
        self.cross_mode = cross_mode
        self.cross_k = cross_k
        self.mamba_options = dict(
            d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand,
            slab_thickness=slab_thickness,
        )
        self.use_relative_pos = use_relative_pos
        super().__init__(*args, **kwargs)

    def _make_cross_layer(self, dim, num_heads, dropout):
        if self.cross_mode == "attention":
            return super()._make_cross_layer(dim, num_heads, dropout)
        if self.cross_mode == "aggregation":
            return MambaAggregation(dim, k=self.cross_k, dropout=dropout)
        return LocalCrossMamba(
            dim, self.cross_k, dropout, use_relative_pos=self.use_relative_pos,
            **self.mamba_options
        )

    def _make_self_layer(self, dim, num_heads, dropout):
        return SSMSpatialSelfLayer(dim, **self.mamba_options)

    def _self_step(self, layer, query, query_pos, query_xyz):
        return self.self_attention[layer]([query], [query_xyz])[0]

    def _neighbors(self, tree_xyz, query_xyz):
        if self.cross_mode == "attention" or not len(tree_xyz) or not len(query_xyz):
            return None if self.cross_mode == "attention" else torch.empty(
                (len(query_xyz), 0), dtype=torch.long, device=query_xyz.device
            )
        k = min(self.cross_k, len(tree_xyz))
        edges = knn(tree_xyz.float().contiguous(), query_xyz.float().contiguous(), k)
        grouping = edges[0].argsort(stable=True)
        indices = edges[1, grouping].reshape(len(query_xyz), k).long()
        distance = (tree_xyz[indices].float() - query_xyz[:, None].float()).square().sum(-1)
        return indices.gather(1, distance.argsort(dim=1, stable=True))

    def _position_lists(self, tree_coordinates, seed_indices, tree_indices):
        if self.cross_mode == "attention":
            return super()._position_lists(tree_coordinates, seed_indices, tree_indices)
        return None, None

    def _cross_step(self, layer, memory, query, mask, source_pos, query_pos,
                    tree_xyz, query_xyz, neighbor_indices):
        if self.cross_mode == "attention":
            return super()._cross_step(
                layer, memory, query, mask, source_pos, query_pos,
                tree_xyz, query_xyz, neighbor_indices
            )
        if self.cross_mode == "aggregation":
            return self.cross_attention[layer](
                query,
                memory[: len(tree_xyz)],
                query_pos=query_xyz,
                src_pos=tree_xyz,
                mask=mask,
                neighbor_indices=neighbor_indices,
            )
        return self.cross_attention[layer](
            memory[: len(tree_xyz)], query, tree_xyz, query_xyz, mask,
            neighbor_indices=neighbor_indices
        )

    def _query_anchors(self, features, coordinates, tree_xyz, tree_indices,
                       seed_indices, use_isa):
        if use_isa and len(tree_indices):
            return super()._query_anchors(
                features, coordinates, tree_xyz, tree_indices, seed_indices, use_isa
            )
        if len(tree_xyz) and self.num_queries:
            count = min(len(tree_xyz), self.num_queries)
            anchor_local = torch_cluster_fps(
                tree_xyz.detach().float().contiguous(),
                ratio=float(count) / len(tree_xyz), random_start=False,
            )[:count].long()
            if count < self.num_queries:
                anchor_local = anchor_local.repeat(
                    (self.num_queries + count - 1) // count
                )[: self.num_queries]
            return tree_xyz[anchor_local]
        return features.new_zeros((self.num_queries, 3), dtype=torch.float32)


__all__ = ["SegmentAnyTreeV2Decoder", "SegmentAnyTreeV2MambaDecoder"]
