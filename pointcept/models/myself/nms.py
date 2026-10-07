"""Mask non-maximum suppression for SegmentAnyTreeV2 inference."""

from __future__ import annotations

import torch


def pairwise_mask_iou(masks: torch.Tensor) -> torch.Tensor:
    """Build the full proposal IoU matrix using the reference ``mm`` form."""

    if masks.ndim != 2:
        raise ValueError("masks must have shape [num_instances, num_points]")
    if len(masks) == 0:
        return masks.new_empty((0, 0), dtype=torch.float32)

    proposal_masks = masks.float()
    intersection = proposal_masks @ proposal_masks.transpose(0, 1)
    proposal_sizes = proposal_masks.sum(dim=1)
    union = proposal_sizes[:, None] + proposal_sizes[None, :] - intersection
    return intersection / union.clamp_min(1)


def non_max_suppression(
    ious: torch.Tensor, scores: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Greedily suppress lower-score proposals above an IoU threshold."""

    if ious.ndim != 2 or ious.shape[0] != ious.shape[1]:
        raise ValueError(
            "ious must be a square [num_instances, num_instances] matrix"
        )
    if scores.ndim != 1 or len(scores) != len(ious):
        raise ValueError("scores must have shape [num_instances]")
    if ious.device != scores.device:
        raise ValueError("ious and scores must be on the same device")
    if not 0 <= threshold <= 1:
        raise ValueError("NMS IoU threshold must be in [0, 1]")
    if len(scores) == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)

    order = scores.argsort(descending=True)
    keep = []
    while len(order):
        current = order[0]
        keep.append(current)
        rest = order[1:]
        if len(rest) == 0:
            break
        order = rest[ious[current, rest] <= threshold]
    return torch.stack(keep)


def greedy_mask_nms(
    masks: torch.Tensor, scores: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Run confidence-ordered hard NMS directly from proposal masks."""

    if masks.ndim != 2:
        raise ValueError("masks must have shape [num_instances, num_points]")
    if scores.ndim != 1 or len(scores) != len(masks):
        raise ValueError("scores must have shape [num_instances]")
    if masks.device != scores.device:
        raise ValueError("masks and scores must be on the same device")
    return non_max_suppression(pairwise_mask_iou(masks), scores, threshold)
