"""Pointcept implementation of SegmentAnyTreeV2.

The implementation follows Sections 3.3--3.8 of the paper.  PTv3 remains a
regular Pointcept backbone; this module implements the semantic head, ISA
embedding/FPS, tree-only masked transformer decoder, curriculum, and losses.
"""

from __future__ import annotations

from functools import partial
import logging
import math

import torch
import torch.nn as nn
from torch_cluster import fps as torch_cluster_fps

from pointcept.models.builder import MODELS, build_model
from pointcept.models.modules import PointModel
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.normalization import SparseBatchNorm1d
from pointcept.models.utils.structure import Point

from .loss import SegmentAnyTreeV2Loss
from .nms2 import mask_matrix_nms


@torch.no_grad()
def farthest_point_sample(features: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Run optimized FPS in an arbitrary-dimensional feature space.

    ``pointops`` FPS kernels are hard-coded for XYZ input, while ISA samples
    the learned 5-D embedding.  ``torch_cluster.fps`` supports arbitrary
    feature dimensions and dispatches to its compiled CPU/CUDA operator.
    When candidates are fewer than queries, every candidate is sampled once
    before indices are repeated to preserve the requested query count. The
    decoder requests at most the number of available tree candidates, so this
    helper's fixed-count behavior does not duplicate sparse point features.
    """

    if features.ndim != 2:
        raise ValueError("FPS features must have shape [N, D]")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    num_points = features.shape[0]
    if num_points == 0:
        return torch.empty(0, dtype=torch.long, device=features.device)

    unique_samples = min(num_points, num_samples)
    selected = torch_cluster_fps(
        features.detach().float().contiguous(),
        ratio=float(unique_samples) / num_points,
        random_start=False,
    )
    if selected.numel() < unique_samples:
        raise RuntimeError(
            "torch_cluster.fps returned fewer samples than requested: "
            f"{selected.numel()} < {unique_samples}"
        )
    selected = selected[:unique_samples].long()

    if unique_samples < num_samples:
        repeats = (num_samples + unique_samples - 1) // unique_samples
        selected = selected.repeat(repeats)[:num_samples]
    return selected


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
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

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
        dim: int = 128,
        num_queries: int = 400,
        num_layers: int = 6,
        num_heads: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.position_encoding = CoordinatePositionEncoding(dim)

        # The paper specifies separate asymmetric pathways and describes the
        # mask pathway as the deeper one with an intermediate LayerNorm.  It
        # does not publish the exact projection layer sequence, so it remains
        # isolated here and can be changed without touching decoder logic.
        self.attention_projection = nn.Linear(in_channels, dim)
        self.query_projection = (
            nn.Identity() if in_channels == dim else nn.Linear(in_channels, dim)
        )
        self.mask_projection = nn.Sequential(
            nn.Linear(in_channels, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
        )
        self.learned_queries = nn.Embedding(num_queries, dim)
        self.learned_query_pos = nn.Embedding(num_queries, dim)

        self.cross_attention = nn.ModuleList(
            CrossAttentionLayer(dim, num_heads, dropout) for _ in range(num_layers)
        )
        self.self_attention = nn.ModuleList(
            SelfAttentionLayer(dim, num_heads, dropout) for _ in range(num_layers)
        )
        self.feed_forward = nn.ModuleList(
            FFN(dim, hidden_dim, dropout) for _ in range(num_layers)
        )

        self.output_norm = nn.LayerNorm(dim)
        self.mask_head = nn.Linear(dim, dim)
        self.score_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 1)
        )

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

        if use_isa and len(tree_indices):
            seed_local = farthest_point_sample(
                tree_embeddings.detach(), min(self.num_queries, len(tree_indices))
            )
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

        source_pos_list, query_pos_list = self._position_lists(
            coordinates, seed_indices, tree_indices
        )

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
                query = self.cross_attention[layer](
                    [attention_memory],
                    [query],
                    [None],
                    pe=source_pos_list,
                    query_pe=query_pos_list,
                )[0]
                query = self.self_attention[layer]([query], pe=query_pos_list)[0]
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
                query = self.cross_attention[layer](
                    [attention_memory],
                    [query],
                    [self._attention_mask(prediction["mask_logits"])],
                    pe=source_pos_list,
                    query_pe=query_pos_list,
                )[0]
                query = self.self_attention[layer]([query], pe=query_pos_list)[0]
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


@MODELS.register_module("SaTv2-v1m1")
class SegmentAnyTreeV2(PointModel):
    def __init__(
        self,
        backbone,
        backbone_out_channels: int = 128,
        semantic_num_classes: int = 3,
        semantic_ignore_index: int = -1,
        tree_class_indices=(1, 2),
        embedding_dim: int = 5,
        warmup_epochs: int = 30,
        decoder=None,
        criteria=None,
        score_threshold: float = 0.15,
        mask_threshold: float = 0.5,
        min_tree_candidates_for_decoder: int = 10,
        min_instance_points: int = 10,
        min_points: int | None = None,
        sparse_log_interval: int = 100,
    ):
        super().__init__()
        norm = partial(SparseBatchNorm1d, eps=1e-3, momentum=0.01)
        self.backbone = build_model(backbone)
        self.semantic_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm(backbone_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels, semantic_num_classes),
        )
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_out_channels, embedding_dim),
        )
        decoder = {} if decoder is None else decoder
        self.decoder = SegmentAnyTreeV2Decoder(
            in_channels=backbone_out_channels, **decoder
        )
        criteria = {} if criteria is None else criteria
        self.criteria = SegmentAnyTreeV2Loss(
            ignore_index=semantic_ignore_index,
            warmup_epochs=warmup_epochs,
            **criteria,
        )

        self.semantic_num_classes = semantic_num_classes
        self.semantic_ignore_index = semantic_ignore_index
        self.tree_class_indices = tuple(tree_class_indices)
        self.warmup_epochs = warmup_epochs
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        if min_points is not None:
            # Backward-compatible alias for older configs.
            min_instance_points = min_points
        if min_tree_candidates_for_decoder < 0 or min_instance_points < 0:
            raise ValueError("sparse instance thresholds must be non-negative")
        self.min_tree_candidates_for_decoder = min_tree_candidates_for_decoder
        self.min_instance_points = min_instance_points
        self.sparse_log_interval = int(sparse_log_interval)
        self._forward_count = 0
        self._pending_input_sizes = []
        self._pending_decoder_skips = []
        self.current_epoch = 0

    def before_epoch(self):
        # Pointcept folds several logical data passes into one trainer epoch
        # when validation is less frequent.  Recover the paper's logical epoch.
        loop = int(getattr(self.trainer.cfg.data.train, "loop", 1))
        self.current_epoch = int(self.trainer.epoch) * loop

    def set_epoch(self, epoch: int):
        """Set the logical epoch for non-Pointcept/custom training loops."""

        self.current_epoch = int(epoch)

    def _tree_mask(
        self, semantic_logits: torch.Tensor, segment: torch.Tensor | None
    ) -> torch.Tensor:
        if self.training and segment is not None:
            labels = segment
        else:
            labels = semantic_logits.argmax(dim=1)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for class_index in self.tree_class_indices:
            mask |= labels == class_index
        return mask

    def _full_masks(self, prediction: dict) -> torch.Tensor:
        full = prediction["mask_logits"].new_full(
            (prediction["mask_logits"].shape[0], prediction["num_points"]), -100
        )
        full[:, prediction["tree_indices"]] = prediction["mask_logits"]
        return full

    @staticmethod
    def _empty_instance_prediction(
        features: torch.Tensor,
        tree_mask: torch.Tensor,
        instance: torch.Tensor | None,
        reason: str,
    ) -> dict:
        tree_indices = tree_mask.nonzero(as_tuple=False).flatten()
        return {
            "mask_logits": features.new_empty((0, len(tree_indices))),
            "score_logits": features.new_empty((0,)),
            "aux_outputs": [],
            "tree_indices": tree_indices,
            "seed_indices": torch.empty(
                0, dtype=torch.long, device=features.device
            ),
            "seed_instance": torch.empty(
                0, dtype=torch.long, device=features.device
            ),
            "tree_instance": (
                instance[tree_indices] if instance is not None else None
            ),
            "num_points": features.shape[0],
            "instance_decode_skipped": True,
            "instance_decode_skip_reason": reason,
        }

    @staticmethod
    def _input_size_records(data_dict, counts, exact_voxel_counts=False):
        if torch.is_tensor(counts):
            counts = counts.tolist()
        names = data_dict.get("name", [])
        if isinstance(names, str):
            names = [names]
        raw_point_counts = data_dict.get("raw_point_count")
        if torch.is_tensor(raw_point_counts):
            raw_point_counts = raw_point_counts.reshape(-1).tolist()
        elif raw_point_counts is None:
            raw_point_counts = []
        elif not isinstance(raw_point_counts, (list, tuple)):
            raw_point_counts = [raw_point_counts]
        grid_coord = data_dict.get("grid_coord")
        records = []
        start = 0
        for sample_index, count in enumerate(counts):
            end = start + count
            voxels = None
            if grid_coord is not None:
                # GridSample yields one feature row per occupied input cell.
                voxels = count
                if exact_voxel_counts:
                    voxels = (
                        int(torch.unique(grid_coord[start:end], dim=0).shape[0])
                        if end > start
                        else 0
                    )
            records.append(
                {
                    "name": names[sample_index]
                    if sample_index < len(names)
                    else f"sample_{sample_index}",
                    "raw_points": raw_point_counts[sample_index]
                    if sample_index < len(raw_point_counts)
                    else None,
                    "points": count,
                    "voxels": voxels,
                }
            )
            start = end
        return records

    def _log_sparse_status(self, input_records, sparse_level_counts=None):
        interval = self.sparse_log_interval
        if interval <= 0 or self._forward_count % interval != 0:
            return
        sparse_level_counts = sparse_level_counts or []
        for sample_index, record in enumerate(input_records):
            record["downsample_active_voxels"] = (
                [
                    (stage, counts[sample_index])
                    for stage, counts in sparse_level_counts
                    if sample_index < len(counts)
                ]
                if sparse_level_counts
                else "n/a"
            )
            self._pending_input_sizes[-len(input_records) + sample_index].update(
                record
            )
        sizes = ", ".join(
            f"{record['name']}:raw_points={record.get('raw_points', 'n/a')},"
            f"points={record['points']},"
            f"voxels={record['voxels'] if record['voxels'] is not None else 'n/a'},"
            f"downsample_active={record.get('downsample_active_voxels', 'n/a')}"
            for record in self._pending_input_sizes
        )
        skipped = ", ".join(
            f"{record['name']}:{record['reason']}"
            f"(tree_candidates={record['tree_candidates']})"
            for record in self._pending_decoder_skips
        )
        logging.getLogger(__name__).info(
            "SaTv2 input sizes (sampled points / unique input voxels): %s; "
            "instance decoder skips below %d tree candidates since last report: %s",
            sizes or "none",
            self.min_tree_candidates_for_decoder,
            skipped or "none",
        )
        self._pending_input_sizes.clear()
        self._pending_decoder_skips.clear()

    @torch.no_grad()
    def inference(self, prediction: dict) -> dict[str, torch.Tensor]:
        if prediction.get("instance_decode_skipped"):
            device = prediction["mask_logits"].device
            return {
                "pred_scores": torch.empty(0, device=device).cpu(),
                "pred_masks": torch.empty(
                    (0, prediction["num_points"]), dtype=torch.bool, device=device
                ).cpu(),
                "pred_classes": torch.empty(0, dtype=torch.long),
            }
        scores = prediction["score_logits"].sigmoid()
        masks = self._full_masks(prediction).sigmoid() >= self.mask_threshold
        # Match the reference post-processing order: NMS first, followed by
        # score and cluster-size filtering.
        if len(scores):
            labels = torch.zeros_like(scores, dtype=torch.long)
            scores, _, masks, _ = mask_matrix_nms(masks, labels, scores)
            keep = (scores >= self.score_threshold) & (
                masks.sum(dim=1) >= max(self.min_instance_points, 1)
            )
            scores = scores[keep]
            masks = masks[keep]
            order = scores.argsort(descending=True)
            scores = scores[order]
            masks = masks[order]
        return {
            "pred_scores": scores.cpu(),
            "pred_masks": masks.cpu(),
            # Instance masks are class-agnostic trees.  Index 1 is used only
            # for compatibility with Pointcept's generic instance API.
            "pred_classes": torch.ones(len(scores), dtype=torch.long),
        }

    def forward(self, data_dict, return_point: bool = False):
        input_features = data_dict.get("feat")
        if input_features is None or input_features.shape[0] == 0:
            raise ValueError(
                "SaTv2 received an empty point set; empty samples must be skipped "
                "before entering the backbone"
            )
        self._forward_count += 1
        report_sizes = (
            self.sparse_log_interval > 0
            and self._forward_count % self.sparse_log_interval == 0
        )
        point = Point(data_dict)
        if report_sizes:
            point["_collect_sparse_counts"] = True
        backbone_output = self.backbone(point)
        if isinstance(backbone_output, Point):
            point = backbone_output
        elif isinstance(backbone_output, torch.Tensor):
            # Volt returns features in input point order; retain batch metadata.
            point.feat = backbone_output
        else:
            raise TypeError("SaTv2 requires a backbone returning Point or Tensor")
        counts = offset2bincount(point.offset)
        counts_list = counts.tolist()
        input_records = []
        if self.sparse_log_interval > 0:
            input_records = self._input_size_records(
                data_dict, counts_list, exact_voxel_counts=report_sizes
            )
            self._pending_input_sizes.extend(
                {
                    "name": record["name"],
                    "raw_points": record["raw_points"],
                    "points": record["points"],
                    "voxels": record["voxels"],
                }
                for record in input_records
            )
        sparse_level_counts = point.get("sparse_level_counts")
        if return_point:
            self._log_sparse_status(input_records, sparse_level_counts)
            return {"point": point}

        semantic_logits = self.semantic_head(point.feat)
        embeddings = self.embedding_head(point.feat)
        feature_list = point.feat.split(counts_list)
        semantic_list = semantic_logits.split(counts_list)
        embedding_list = embeddings.split(counts_list)
        coordinate_list = point.coord.split(counts_list)
        segment_list = (
            data_dict["segment"].split(counts_list)
            if "segment" in data_dict
            else [None] * len(counts)
        )
        instance_list = (
            data_dict["instance"].split(counts_list)
            if "instance" in data_dict
            else [None] * len(counts)
        )

        use_isa = (not self.training) or self.current_epoch >= self.warmup_epochs
        predictions = []
        sample_names = data_dict.get("name", [])
        if isinstance(sample_names, str):
            sample_names = [sample_names]
        for features, semantic, embedding, segment, instance, coordinates in zip(
            feature_list,
            semantic_list,
            embedding_list,
            segment_list,
            instance_list,
            coordinate_list,
        ):
            tree_mask = self._tree_mask(semantic, segment)
            tree_count = int(tree_mask.sum().item())
            if features.shape[0] == 0:
                prediction = self._empty_instance_prediction(
                    features, tree_mask, instance, "empty_sample"
                )
            elif tree_count < self.min_tree_candidates_for_decoder:
                prediction = self._empty_instance_prediction(
                    features, tree_mask, instance, "insufficient_tree_candidates"
                )
            else:
                prediction = self.decoder.forward_sample(
                    features=features,
                    embeddings=embedding,
                    tree_mask=tree_mask,
                    instance=instance,
                    use_isa=use_isa,
                    coordinates=coordinates,
                )
            predictions.append(prediction)
            if (
                prediction.get("instance_decode_skipped")
                and self.sparse_log_interval > 0
            ):
                sample_index = len(predictions) - 1
                self._pending_decoder_skips.append(
                    {
                        "name": sample_names[sample_index]
                        if sample_index < len(sample_names)
                        else f"sample_{sample_index}",
                        "tree_candidates": tree_count,
                        "reason": prediction["instance_decode_skip_reason"],
                    }
                )

        self._log_sparse_status(input_records, sparse_level_counts)

        output = {}
        if "segment" in data_dict and "instance" in data_dict:
            output.update(
                self.criteria(
                    semantic_logits=semantic_logits,
                    embeddings=embeddings,
                    predictions=predictions,
                    segment=data_dict["segment"],
                    instance=data_dict["instance"],
                    counts=counts,
                    current_epoch=self.current_epoch,
                    instance_active=use_isa,
                )
            )
        # Pointcept's training logger treats every returned item as a scalar.
        # Dense predictions remain available to validation/inference only.
        if self.training:
            return output

        output.update(
            seg_logits=semantic_logits,
            pred_semantic=semantic_logits.argmax(dim=1),
            embeddings=embeddings,
        )
        if len(predictions) != 1:
            raise ValueError("SaTv2 inference currently requires batch size 1")
        output.update(self.inference(predictions[0]))
        return output
