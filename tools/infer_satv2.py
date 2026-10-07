#!/usr/bin/env python3
"""Plot-level cylindrical inference and merging for SegmentAnyTreeV2.

This implements the reproducible parts of Section 3.9 and Appendix B: 16 m
cylinders, semantic voting, local confidence/boundary filtering, confidence
ordered asymmetric-overlap global merging, size filtering, and ground overwrite.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from pointcept.datasets.transform import CenterShift, GridSample
from pointcept.models import build_model
from pointcept.utils.config import Config
from pointcept.utils.satv2_metrics import (
    instance_metrics,
    semantic_confusion_matrix,
    semantic_iou,
    summarize_detection,
)


SEMANTIC_FIELDS = ("semantic_seg", "semantic", "segment", "class", "label")
INSTANCE_FIELDS = ("treeID", "tree_id", "tree", "instance", "instance_id")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input PLY plot")
    parser.add_argument("output", type=Path, help="Output PLY with predictions")
    parser.add_argument("--config", type=Path, default=Path("configs/for_instance/insseg-satv2-v1m1-0-ptv3.py"))
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--radius", type=float, default=16.0)
    parser.add_argument("--step", type=float, default=4.0)
    parser.add_argument("--grid-size", type=float, default=0.2)
    parser.add_argument("--max-points", type=int, default=650_000)
    parser.add_argument("--confidence-threshold", type=float, default=0.15)
    parser.add_argument("--boundary-margin", type=float, default=0.5)
    parser.add_argument("--global-overlap-threshold", type=float, default=0.1)
    parser.add_argument(
        "--min-points",
        type=int,
        default=10,
        help="minimum original input points retained in a merged predicted instance",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--semantic-offset",
        type=int,
        choices=(0, 1),
        default=1,
        help="offset of optional ground-truth semantic fields (FOR-instance: 1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-boundary", action="store_true")
    args = parser.parse_args()
    if args.radius <= 0 or args.step <= 0 or args.grid_size <= 0:
        parser.error("radius, step, and grid-size must be positive")
    if args.max_points <= 0 or args.min_points <= 0:
        parser.error("max-points and min-points must be positive")
    return args


def find_field(names, candidates):
    return next((candidate for candidate in candidates if candidate in names), None)


def normalize_semantic(values, offset):
    values = np.rint(np.asarray(values)).astype(np.int64)
    values -= offset
    unique = np.unique(values)
    if np.isin(unique, (0, 1, 2)).all():
        return values
    raise ValueError(
        f"semantic labels minus offset {offset} must be 0/1/2, got {unique.tolist()}"
    )


def load_model(config_path, weight_path, device, confidence_threshold):
    cfg = Config.fromfile(str(config_path))
    cfg.model.score_threshold = confidence_threshold
    model = build_model(cfg.model)
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    clean_state_dict = OrderedDict()
    for key, value in state_dict.items():
        clean_state_dict[key.removeprefix("module.")] = value
    model.load_state_dict(clean_state_dict, strict=True)
    model.to(device).eval()
    return model


def cylinder_centres(coord, step):
    maximum = coord[:, :2].max(axis=0)
    xs = np.arange(0.0, maximum[0] + step, step)
    ys = np.arange(0.0, maximum[1] + step, step)
    return np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)


def prepare_cylinder(coord, grid_size, device):
    data = {"coord": coord.astype(np.float32, copy=True)}
    data = CenterShift(apply_z=True)(data)
    data = GridSample(
        grid_size=grid_size,
        hash_type="fnv",
        mode="train",
        return_inverse=True,
        return_grid_coord=True,
    )(data)
    coord_tensor = torch.from_numpy(data["coord"]).float().to(device)
    return {
        "coord": coord_tensor,
        "grid_coord": torch.from_numpy(data["grid_coord"]).int().to(device),
        "feat": coord_tensor,
        "offset": torch.tensor([len(coord_tensor)], dtype=torch.int32, device=device),
    }, data["inverse"]


def local_candidates(masks, scores, global_indices, radial_distance, args):
    """Apply boundary/confidence filters and local max-conf assignment."""

    masks = np.asarray(masks, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    claimed = np.zeros(len(global_indices), dtype=bool)
    candidates = []
    for mask_index in order:
        if scores[mask_index] < args.confidence_threshold:
            continue
        support = masks[mask_index] & ~claimed
        if not support.any():
            continue
        if not args.keep_boundary and np.any(
            radial_distance[support] >= args.radius - args.boundary_margin
        ):
            continue
        claimed[support] = True
        candidates.append((float(scores[mask_index]), global_indices[support]))
    return candidates


def merge_candidates(candidates, num_points, overlap_threshold, min_points):
    claimed = np.zeros(num_points, dtype=bool)
    accepted = []
    for score, support in sorted(candidates, key=lambda item: item[0], reverse=True):
        support = np.unique(support)
        if len(support) < min_points:
            continue
        asymmetric_overlap = claimed[support].sum() / len(support)
        if asymmetric_overlap > overlap_threshold:
            continue
        accepted.append((score, support))
        claimed[support] = True

    labels = np.zeros(num_points, dtype=np.int32)
    score_by_id = {}
    for instance_id, (score, support) in enumerate(accepted, start=1):
        support = support[labels[support] == 0]
        if len(support) < min_points:
            continue
        labels[support] = instance_id
        score_by_id[instance_id] = score
    return relabel_instances(labels, score_by_id)


def relabel_instances(labels, score_by_id):
    """Restore contiguous positive IDs and carry their confidence scores."""

    relabelled = np.zeros_like(labels)
    relabelled_scores = {}
    for new_id, old_id in enumerate(np.unique(labels[labels > 0]), start=1):
        relabelled[labels == old_id] = new_id
        relabelled_scores[new_id] = float(score_by_id[int(old_id)])
    return relabelled, relabelled_scores


def append_predictions(vertex, semantic, instance, confidence):
    existing = vertex.dtype.names or ()
    additions = []
    if "semantic_pred" not in existing:
        additions.append(("semantic_pred", "u1"))
    if "instance_pred" not in existing:
        additions.append(("instance_pred", "<i4"))
    if "instance_confidence" not in existing:
        additions.append(("instance_confidence", "<f4"))
    output = np.empty(len(vertex), dtype=vertex.dtype.descr + additions)
    for name in existing:
        output[name] = vertex[name]
    output["semantic_pred"] = semantic.astype(np.uint8) + 1
    output["instance_pred"] = instance.astype(np.int32)
    output["instance_confidence"] = confidence.astype(np.float32)
    return output


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    ply = PlyData.read(str(args.input), mmap="r")
    if "vertex" not in ply:
        raise ValueError("input PLY has no vertex element")
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    if not all(name in names for name in ("x", "y", "z")):
        raise ValueError("input PLY requires x, y, z vertex properties")
    original_coord = np.column_stack([vertex[name] for name in ("x", "y", "z")])
    if not np.all(np.isfinite(original_coord)):
        raise ValueError("input coordinates contain NaN or Inf")
    shift = original_coord.min(axis=0)
    coord = (original_coord - shift).astype(np.float32)
    semantic_field = find_field(names, SEMANTIC_FIELDS)
    instance_field = find_field(names, INSTANCE_FIELDS)
    target_semantic = (
        normalize_semantic(vertex[semantic_field], args.semantic_offset)
        if semantic_field
        else None
    )
    target_instance = (
        np.where(np.asarray(vertex[instance_field]) > 0, vertex[instance_field], -1).astype(np.int64)
        if instance_field
        else None
    )

    model = load_model(
        args.config,
        args.weight,
        args.device,
        args.confidence_threshold,
    )
    xy_tree = cKDTree(coord[:, :2])
    centres = cylinder_centres(coord, args.step)
    semantic_votes = np.zeros((len(coord), 3), dtype=np.uint16)
    candidates = []
    processed = 0
    empty_cylinders = 0
    for cylinder_index, centre in enumerate(centres):
        indices = np.asarray(
            xy_tree.query_ball_point(centre, args.radius), dtype=np.int64
        )
        if len(indices) == 0:
            empty_cylinders += 1
            continue
        if len(indices) > args.max_points:
            indices = np.sort(
                rng.choice(indices, size=args.max_points, replace=False)
            )
        batch, inverse = prepare_cylinder(coord[indices], args.grid_size, args.device)
        with torch.no_grad():
            prediction = model(batch)
        point_semantic = prediction["pred_semantic"].detach().cpu().numpy()[inverse]
        np.add.at(semantic_votes, (indices, point_semantic), 1)
        point_masks = prediction["pred_masks"].numpy()[:, inverse]
        point_scores = prediction["pred_scores"].numpy()
        radial_distance = np.linalg.norm(coord[indices, :2] - centre, axis=1)
        candidates.extend(
            local_candidates(
                point_masks, point_scores, indices, radial_distance, args
            )
        )
        processed += 1
        if processed % 25 == 0:
            print(
                f"cylinders={processed}, grid={cylinder_index + 1}/{len(centres)}, "
                f"candidates={len(candidates)}",
                flush=True,
            )

    prediction_semantic = semantic_votes.argmax(axis=1).astype(np.int64)
    prediction_instance, score_by_id = merge_candidates(
        candidates,
        len(coord),
        args.global_overlap_threshold,
        args.min_points,
    )
    prediction_instance[prediction_semantic == 0] = 0
    prediction_instance, score_by_id = relabel_instances(
        prediction_instance, score_by_id
    )
    prediction_confidence = np.zeros(len(coord), dtype=np.float32)
    for instance_id, confidence in score_by_id.items():
        prediction_confidence[prediction_instance == instance_id] = confidence

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_vertex = append_predictions(
        vertex,
        prediction_semantic,
        prediction_instance,
        prediction_confidence,
    )
    PlyData(
        [PlyElement.describe(output_vertex, "vertex")],
        text=ply.text,
        byte_order=ply.byte_order,
        comments=list(ply.comments) + ["SegmentAnyTreeV2 Pointcept reproduction"],
    ).write(str(args.output))

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "points": len(coord),
        "cylinders": processed,
        "empty_cylinders": empty_cylinders,
        "total_cylinder_windows": len(centres),
        "local_candidates": len(candidates),
        "instances": int(prediction_instance.max(initial=0)),
        "settings": {
            "radius": args.radius,
            "step": args.step,
            "grid_size": args.grid_size,
            "confidence_threshold": args.confidence_threshold,
            "boundary_margin": args.boundary_margin,
            "global_overlap_threshold": args.global_overlap_threshold,
            "min_points_original": args.min_points,
        },
    }
    if target_semantic is not None:
        confusion = semantic_confusion_matrix(
            prediction_semantic, target_semantic, num_classes=3, ignore_index=-1
        )
        class_iou, mean_iou = semantic_iou(confusion)
        report["semantic_iou"] = class_iou.tolist()
        report["semantic_miou"] = mean_iou
    if target_instance is not None:
        metrics = instance_metrics(
            prediction_instance,
            target_instance,
            score_by_id=score_by_id,
            iou_threshold=args.iou_threshold,
        )
        report.update(summarize_detection(metrics["tp"], metrics["fp"], metrics["fn"]))
        report["coverage"] = metrics["weighted_coverage"]
        report.update({key: metrics[key] for key in ("tp", "fp", "fn")})
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
