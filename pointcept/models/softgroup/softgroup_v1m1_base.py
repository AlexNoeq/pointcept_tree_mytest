"""SoftGroup instance segmentation model adapted to Pointcept.

This implementation keeps SoftGroup's two-stage design while using the
Pointcept data contract:

* Point-wise semantic logits and centroid offsets are produced by a Pointcept
  backbone.
* A point may enter proposals for multiple semantic classes (soft grouping).
* Proposal-local sparse convolutions refine class, mask, and IoU predictions.

The model intentionally relies on Pointcept's existing ``pointgroup_ops``
extension for radius queries and connected-component clustering.  No dataset,
runner, evaluator, or custom CUDA extension changes are required.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean, scatter_min

try:
    from pointgroup_ops import ballquery_batch_p, bfs_cluster
except ImportError:  # pragma: no cover - raised with a clear message at use time.
    ballquery_batch_p = None
    bfs_cluster = None

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils import offset2batch


def _get(cfg, key, default=None):
    """Read a key from either a Pointcept ConfigDict or a plain mapping."""
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class MLP(nn.Sequential):
    """The point-wise prediction head used by SoftGroup."""

    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_channels, in_channels))
            if norm_fn is not None:
                layers.append(norm_fn(in_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(in_channels, out_channels))
        super().__init__(*layers)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)


@MODELS.register_module("SoftGroup-v1m1")
class SoftGroup(nn.Module):
    """SoftGroup for Pointcept instance segmentation.

    Args:
        backbone (dict): Pointcept point backbone returning one feature per
            input point, for example ``SpUNet-v1m1``.
        backbone_out_channels (int): Number of features returned by
            ``backbone``.
        instance_backbone (dict, optional): Pointcept sparse backbone used on
            proposal-local voxels.  When omitted, a compact SpUNet with the
            same input/output channel count is created.
        instance_out_channels (int, optional): Number of features returned by
            ``instance_backbone``.  Defaults to ``backbone_out_channels``.
        semantic_num_classes (int): Number of semantic classes.
        thing_class_ids (sequence[int], optional): Semantic IDs evaluated as
            instances.  Defaults to every non-ignored semantic class.
        segment_ignore_index (sequence[int]): Non-instance semantic IDs.
        grouping_cfg (dict): Soft grouping settings.  Supported keys are
            ``score_thr``, ``radius``, ``mean_active``, ``npoint_thr``,
            ``class_numpoint_mean`` and ``max_proposal_num``.
        instance_voxel_cfg (dict): Proposal voxelization settings with
            ``scale`` and ``spatial_shape``.
        train_cfg/test_cfg (dict): Loss and inference thresholds.

    Input follows Pointcept's instance-segmentation convention: ``coord``,
    ``feat``, ``offset``, ``segment``, ``instance`` and
    ``instance_centroid``.  In evaluation the returned prediction keys are
    directly consumable by ``InsSegEvaluator`` and ``InsSegTester``.
    """

    def __init__(
        self,
        backbone,
        backbone_out_channels,
        instance_backbone=None,
        instance_out_channels=None,
        semantic_num_classes=20,
        thing_class_ids=None,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        with_coords=False,
        semantic_weight=None,
        grouping_cfg=None,
        instance_voxel_cfg=None,
        train_cfg=None,
        test_cfg=None,
        fixed_modules=(),
    ):
        super().__init__()
        self.semantic_num_classes = semantic_num_classes
        self.semantic_ignore_index = semantic_ignore_index
        self.segment_ignore_index = tuple(segment_ignore_index)
        self.instance_ignore_index = instance_ignore_index
        self.with_coords = with_coords
        self.grouping_cfg = grouping_cfg or {}
        self.instance_voxel_cfg = instance_voxel_cfg or {}
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}
        self.fixed_modules = tuple(fixed_modules)

        if thing_class_ids is None:
            thing_class_ids = [
                class_id
                for class_id in range(semantic_num_classes)
                if class_id not in self.segment_ignore_index
            ]
        self.thing_class_ids = tuple(int(class_id) for class_id in thing_class_ids)
        if not self.thing_class_ids:
            raise ValueError("thing_class_ids must contain at least one semantic class.")
        if any(class_id < 0 or class_id >= semantic_num_classes for class_id in self.thing_class_ids):
            raise ValueError("thing_class_ids must be valid semantic class IDs.")
        self.instance_num_classes = len(self.thing_class_ids)
        self.instance_out_channels = instance_out_channels or backbone_out_channels

        self.backbone = build_model(backbone)
        norm_fn = lambda channels: nn.BatchNorm1d(channels, eps=1e-4, momentum=0.1)
        self.semantic_head = MLP(
            backbone_out_channels, semantic_num_classes, norm_fn=norm_fn, num_layers=2
        )
        self.offset_head = MLP(backbone_out_channels, 3, norm_fn=norm_fn, num_layers=2)

        if instance_backbone is None:
            # This default is deliberately small.  Supplying a custom backbone
            # is recommended for reproducing a particular SoftGroup recipe.
            instance_backbone = dict(
                type="SpUNet-v1m1",
                in_channels=backbone_out_channels,
                num_classes=0,
                base_channels=min(32, backbone_out_channels),
                channels=(
                    backbone_out_channels,
                    backbone_out_channels,
                    backbone_out_channels,
                    backbone_out_channels,
                    backbone_out_channels,
                    backbone_out_channels,
                    backbone_out_channels,
                    backbone_out_channels,
                ),
                layers=(1, 1, 1, 1, 1, 1, 1, 1),
            )
        self.instance_backbone = build_model(instance_backbone)
        self.cls_head = nn.Linear(self.instance_out_channels, self.instance_num_classes + 1)
        self.mask_head = MLP(
            self.instance_out_channels,
            self.instance_num_classes + 1,
            norm_fn=None,
            num_layers=2,
        )
        self.iou_head = nn.Linear(self.instance_out_channels, self.instance_num_classes + 1)

        semantic_weight_tensor = None
        if semantic_weight is not None:
            semantic_weight_tensor = torch.as_tensor(semantic_weight, dtype=torch.float)
            if semantic_weight_tensor.numel() != semantic_num_classes:
                raise ValueError("semantic_weight must have semantic_num_classes entries.")
        self.register_buffer("semantic_weight", semantic_weight_tensor)

        self._init_weights()
        for module_name in self.fixed_modules:
            module = getattr(self, module_name)
            for parameter in module.parameters():
                parameter.requires_grad = False

    def _init_weights(self):
        self.semantic_head.init_weights()
        self.offset_head.init_weights()
        self.mask_head.init_weights()
        for module in (self.cls_head, self.iou_head):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        # Frozen BatchNorm statistics must remain frozen while fine-tuning the
        # proposal heads, matching the original SoftGroup training behaviour.
        for module_name in self.fixed_modules:
            for module in getattr(self, module_name).modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()
        return self

    def forward(self, data_dict):
        feat = self._forward_backbone(data_dict)
        semantic_logits = self.semantic_head(feat)
        offset_preds = self.offset_head(feat)

        output = {"seg_logits": semantic_logits}
        has_label = "segment" in data_dict and "instance" in data_dict
        if has_label:
            point_losses = self._point_losses(data_dict, semantic_logits, offset_preds)
            output.update(point_losses)
        else:
            output["loss"] = semantic_logits.sum() * 0

        if self.training:
            if has_label:
                proposal_losses = self._proposal_losses(
                    data_dict, feat, semantic_logits, offset_preds
                )
                output.update(proposal_losses)
                output["loss"] = output["loss"] + proposal_losses["instance_loss"]
            return output

        proposal_idx, proposal_offset = self._soft_group(
            data_dict["coord"], data_dict["offset"], semantic_logits, offset_preds
        )
        predictions = self._predict_instances(
            data_dict["coord"], feat, semantic_logits, proposal_idx, proposal_offset
        )
        output.update(predictions)
        return output

    def _forward_backbone(self, data_dict):
        backbone_input = data_dict
        if self.with_coords:
            # Avoid mutating the data dictionary shared with the evaluator.
            backbone_input = dict(data_dict)
            backbone_input["feat"] = torch.cat(
                [data_dict["feat"], data_dict["coord"]], dim=1
            )
        feature = self.backbone(backbone_input)
        # Some Pointcept backbones return a Point object; its feature tensor is
        # the finest-resolution result after following pooling inverses.
        if hasattr(feature, "feat"):
            while "pooling_parent" in feature:
                parent = feature.pop("pooling_parent")
                inverse = feature.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, feature.feat[inverse]], dim=-1)
                feature = parent
            feature = feature.feat
        return feature

    def _point_losses(self, data_dict, semantic_logits, offset_preds):
        segment = data_dict["segment"].long()
        instance = data_dict["instance"].long()
        semantic_loss = F.cross_entropy(
            semantic_logits,
            segment,
            weight=self.semantic_weight,
            ignore_index=self.semantic_ignore_index,
        )
        offset_target = data_dict["instance_centroid"].float() - data_dict["coord"].float()
        valid = instance != self.instance_ignore_index
        if valid.any():
            offset_loss = F.l1_loss(offset_preds[valid], offset_target[valid], reduction="mean")
        else:
            offset_loss = offset_preds.sum() * 0
        loss = semantic_loss + offset_loss
        return {
            "loss": loss,
            "semantic_loss": semantic_loss,
            "offset_loss": offset_loss,
        }

    def _proposal_losses(self, data_dict, point_feat, semantic_logits, offset_preds):
        proposal_idx, proposal_offset = self._soft_group(
            data_dict["coord"], data_dict["offset"], semantic_logits, offset_preds
        )
        max_proposals = _get(self.train_cfg, "max_proposal_num", 200)
        if proposal_offset.numel() - 1 > max_proposals:
            proposal_idx, proposal_offset = self._truncate_proposals(
                proposal_idx, proposal_offset, max_proposals
            )

        if proposal_offset.numel() <= 1:
            zero = point_feat.sum() * 0
            return {
                "instance_loss": zero,
                "cls_loss": zero,
                "mask_loss": zero,
                "iou_score_loss": zero,
                "num_pos": zero.detach(),
                "num_neg": zero.detach(),
            }

        cls_logits, iou_scores, mask_logits, proposal_ids, point_indices = self._instance_forward(
            data_dict["coord"], point_feat, proposal_idx, proposal_offset
        )
        global_instance, instance_classes, instance_counts = self._global_instances(data_dict)
        return self._instance_losses(
            cls_logits,
            iou_scores,
            mask_logits,
            proposal_ids,
            point_indices,
            global_instance,
            instance_classes,
            instance_counts,
        )

    def _soft_group(self, coord, offset, semantic_logits, offset_preds):
        """Generate class-wise connected components from soft semantic scores."""
        if ballquery_batch_p is None or bfs_cluster is None:
            raise ImportError(
                "SoftGroup-v1m1 requires pointgroup_ops. Build libs/pointgroup_ops "
                "before training or inference."
            )

        device = coord.device
        scores = semantic_logits.float().softmax(dim=-1)
        batch = offset2batch(offset).int()
        batch_size = int(offset.numel())
        proposals, offsets = [], []
        total_points = 0
        total_proposals = 0

        for semantic_class in self.thing_class_ids:
            point_ids = torch.nonzero(
                scores[:, semantic_class] > _get(self.grouping_cfg, "score_thr", 0.2),
                as_tuple=False,
            ).flatten()
            min_points = self._minimum_cluster_size(semantic_class)
            if point_ids.numel() < min_points:
                continue

            class_batch = batch[point_ids].contiguous()
            batch_counts = torch.bincount(class_batch, minlength=batch_size)
            class_offsets = torch.cat(
                [batch_counts.new_zeros(1), batch_counts.cumsum(0)]
            ).int().contiguous()
            neighbor_idx, start_len = ballquery_batch_p(
                (coord[point_ids].float() + offset_preds[point_ids].float()).contiguous(),
                class_batch,
                class_offsets,
                _get(self.grouping_cfg, "radius", 0.04),
                _get(self.grouping_cfg, "mean_active", 300),
            )
            # The existing PointGroup kernel only needs labels to prevent
            # cross-class merging.  We run one semantic class at a time, so a
            # zero label is sufficient and preserves SoftGroup's multi-class
            # membership behaviour.
            local_labels = torch.zeros(point_ids.numel(), dtype=torch.int32)
            component_idx, component_offset = bfs_cluster(
                local_labels, neighbor_idx.cpu(), start_len.cpu(), min_points
            )
            if component_offset.numel() <= 1:
                continue

            component_idx = component_idx.to(device=device, dtype=torch.long)
            component_offset = component_offset.to(device=device, dtype=torch.long)
            component_idx[:, 1] = point_ids[component_idx[:, 1]]
            component_idx[:, 0] += total_proposals
            offsets.append(component_offset[1:] + total_points)
            proposals.append(component_idx)
            total_proposals += component_offset.numel() - 1
            total_points += int(component_offset[-1])

        if not proposals:
            return (
                torch.empty((0, 2), dtype=torch.long, device=device),
                torch.zeros(1, dtype=torch.long, device=device),
            )
        proposal_idx = torch.cat(proposals, dim=0)
        proposal_offset = torch.cat([torch.zeros(1, dtype=torch.long, device=device), *offsets])
        return proposal_idx, proposal_offset

    def _minimum_cluster_size(self, semantic_class):
        threshold = _get(self.grouping_cfg, "npoint_thr", 1)
        means = _get(self.grouping_cfg, "class_numpoint_mean", None)
        if means is not None and 0 < threshold < 1:
            mean = means[semantic_class]
            if mean > 0:
                return max(1, int(ceil(mean * threshold)))
        return max(1, int(ceil(threshold)))

    @staticmethod
    def _truncate_proposals(proposal_idx, proposal_offset, max_proposals):
        last = int(proposal_offset[max_proposals].item())
        return proposal_idx[:last], proposal_offset[: max_proposals + 1]

    def _instance_forward(self, coord, point_feat, proposal_idx, proposal_offset):
        voxel_dict, proposal_ids, point_indices, voxel_to_member = self._voxelize_proposals(
            coord, point_feat, proposal_idx, proposal_offset
        )
        voxel_feat = self.instance_backbone(voxel_dict)
        member_feat = voxel_feat[voxel_to_member]
        num_proposals = proposal_offset.numel() - 1
        pooled_feat = scatter_mean(
            voxel_feat,
            voxel_dict["batch"],
            dim=0,
            dim_size=num_proposals,
        )
        return (
            self.cls_head(pooled_feat),
            self.iou_head(pooled_feat),
            self.mask_head(member_feat),
            proposal_ids,
            point_indices,
        )

    def _voxelize_proposals(self, coord, point_feat, proposal_idx, proposal_offset):
        """Create proposal-local voxels and Pointcept's sparse-backbone input."""
        num_proposals = proposal_offset.numel() - 1
        lengths = proposal_offset[1:] - proposal_offset[:-1]
        proposal_ids = torch.repeat_interleave(
            torch.arange(num_proposals, device=coord.device), lengths
        )
        point_indices = proposal_idx[:, 1].long()
        member_coord = coord[point_indices].float()
        member_feat = point_feat[point_indices]

        coord_min, _ = scatter_min(member_coord, proposal_ids, dim=0, dim_size=num_proposals)
        coord_max, _ = scatter_max(member_coord, proposal_ids, dim=0, dim_size=num_proposals)
        spatial_shape = int(_get(self.instance_voxel_cfg, "spatial_shape", 20))
        max_scale = float(_get(self.instance_voxel_cfg, "scale", 50))
        extent = (coord_max - coord_min).amax(dim=1).clamp_min(1e-6)
        scale = ((spatial_shape - 0.01) / extent).clamp(max=max_scale)
        grid_coord = torch.floor(
            (member_coord - coord_min[proposal_ids]) * scale[proposal_ids, None]
        ).long()
        grid_coord.clamp_(0, spatial_shape - 1)
        member_voxel_coord = torch.cat([proposal_ids[:, None], grid_coord], dim=1)
        voxel_coord, voxel_to_member = torch.unique(
            member_voxel_coord, dim=0, sorted=True, return_inverse=True
        )
        voxel_feat = scatter_mean(member_feat, voxel_to_member, dim=0)
        voxel_batch = voxel_coord[:, 0].long()
        voxel_count = torch.bincount(voxel_batch, minlength=num_proposals)
        voxel_dict = {
            "grid_coord": voxel_coord[:, 1:].int(),
            "feat": voxel_feat,
            "offset": voxel_count.cumsum(0).int(),
            "batch": voxel_batch,
        }
        return voxel_dict, proposal_ids, point_indices, voxel_to_member

    def _global_instances(self, data_dict):
        """Make per-scene instance IDs unique after Pointcept batch collation."""
        instance = data_dict["instance"].long()
        segment = data_dict["segment"].long()
        batch = offset2batch(data_dict["offset"]).long()
        valid = instance != self.instance_ignore_index
        global_instance = torch.full_like(instance, self.instance_ignore_index)
        if not valid.any():
            empty = torch.empty(0, dtype=torch.long, device=instance.device)
            return global_instance, empty, empty

        _, inverse = torch.unique(
            torch.stack([batch[valid], instance[valid]], dim=1),
            dim=0,
            return_inverse=True,
        )
        global_instance[valid] = inverse
        num_instances = int(inverse.max()) + 1
        counts = torch.bincount(inverse, minlength=num_instances)
        first = torch.full(
            (num_instances,), segment.numel(), dtype=torch.long, device=segment.device
        )
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        first.scatter_reduce_(0, inverse, valid_indices, reduce="amin", include_self=True)
        classes = segment[first]
        return global_instance, classes, counts

    def _instance_losses(
        self,
        cls_logits,
        iou_scores,
        mask_logits,
        proposal_ids,
        point_indices,
        global_instance,
        instance_classes,
        instance_counts,
    ):
        num_proposals = cls_logits.shape[0]
        num_instances = instance_classes.numel()
        zero = cls_logits.sum() * 0
        if num_proposals == 0 or num_instances == 0:
            return {
                "instance_loss": zero,
                "cls_loss": zero,
                "mask_loss": zero,
                "iou_score_loss": zero,
                "num_pos": zero.detach(),
                "num_neg": zero.detach(),
            }

        member_instance = global_instance[point_indices]
        valid_member = member_instance != self.instance_ignore_index
        intersections = cls_logits.new_zeros((num_proposals, num_instances))
        if valid_member.any():
            flat_index = proposal_ids[valid_member] * num_instances + member_instance[valid_member]
            intersections.view(-1).scatter_add_(
                0, flat_index, torch.ones_like(flat_index, dtype=cls_logits.dtype)
            )
        proposal_sizes = torch.bincount(proposal_ids, minlength=num_proposals).to(cls_logits.dtype)
        unions = proposal_sizes[:, None] + instance_counts[None, :].to(cls_logits.dtype) - intersections
        ious = intersections / unions.clamp_min(1)

        max_iou, matched_instance = ious.max(dim=1)
        positive = max_iou >= _get(self.train_cfg, "pos_iou_thr", 0.5)
        class_lookup = cls_logits.new_full(
            (self.semantic_num_classes,), self.instance_num_classes, dtype=torch.long
        )
        class_lookup[torch.as_tensor(self.thing_class_ids, device=cls_logits.device)] = torch.arange(
            self.instance_num_classes, device=cls_logits.device
        )
        matched_class = class_lookup[instance_classes[matched_instance]]
        cls_target = torch.full(
            (num_proposals,), self.instance_num_classes, dtype=torch.long, device=cls_logits.device
        )
        cls_target[positive] = matched_class[positive]
        cls_loss = F.cross_entropy(cls_logits, cls_target)

        selected_class = cls_target[proposal_ids]
        mask_logits_selected = mask_logits.gather(1, selected_class[:, None]).squeeze(1)
        member_match = member_instance == matched_instance[proposal_ids]
        mask_valid = positive[proposal_ids] & valid_member
        if mask_valid.any():
            mask_target = member_match[mask_valid].to(mask_logits.dtype)
            mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits_selected[mask_valid], mask_target, reduction="mean"
            )
        else:
            mask_loss = zero

        iou_prediction = iou_scores.gather(1, cls_target[:, None]).squeeze(1)
        if positive.any():
            iou_score_loss = F.mse_loss(iou_prediction[positive], max_iou[positive])
        else:
            iou_score_loss = zero
        instance_loss = cls_loss + mask_loss + iou_score_loss
        return {
            "instance_loss": instance_loss,
            "cls_loss": cls_loss,
            "mask_loss": mask_loss,
            "iou_score_loss": iou_score_loss,
            "num_pos": positive.sum().float().detach(),
            "num_neg": (~positive).sum().float().detach(),
        }

    def _predict_instances(self, coord, point_feat, semantic_logits, proposal_idx, proposal_offset):
        num_points = coord.shape[0]
        if proposal_offset.numel() <= 1:
            return {
                "pred_scores": torch.empty(0),
                "pred_masks": torch.empty((0, num_points), dtype=torch.int32),
                "pred_classes": torch.empty(0, dtype=torch.long),
            }

        cls_logits, iou_scores, mask_logits, proposal_ids, point_indices = self._instance_forward(
            coord, point_feat, proposal_idx, proposal_offset
        )
        cls_prob = cls_logits.softmax(dim=1)
        proposal_count = cls_logits.shape[0]
        score_threshold = _get(self.test_cfg, "cls_score_thr", 0.001)
        mask_threshold = _get(self.test_cfg, "mask_score_thr", -0.5)
        min_points = _get(self.test_cfg, "min_npoint", 100)

        all_scores, all_masks, all_classes = [], [], []
        for instance_class, semantic_class in enumerate(self.thing_class_ids):
            class_scores = cls_prob[:, instance_class] * iou_scores[:, instance_class].sigmoid()
            member_mask = mask_logits[:, instance_class] > mask_threshold
            mask = torch.zeros(
                (proposal_count, num_points), dtype=torch.bool, device=point_feat.device
            )
            mask[proposal_ids[member_mask], point_indices[member_mask]] = True
            valid = (class_scores > score_threshold) & (mask.sum(1) >= min_points)
            if valid.any():
                all_scores.append(class_scores[valid])
                all_masks.append(mask[valid])
                all_classes.append(
                    torch.full(
                        (int(valid.sum().item()),),
                        semantic_class,
                        dtype=torch.long,
                        device=point_feat.device,
                    )
                )

        if not all_scores:
            return {
                "pred_scores": torch.empty(0),
                "pred_masks": torch.empty((0, num_points), dtype=torch.int32),
                "pred_classes": torch.empty(0, dtype=torch.long),
            }
        return {
            "pred_scores": torch.cat(all_scores).detach().cpu(),
            "pred_masks": torch.cat(all_masks).to(torch.int32).detach().cpu(),
            "pred_classes": torch.cat(all_classes).detach().cpu(),
        }
