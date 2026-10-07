#!/usr/bin/env python3
"""Infer raw WHU-Stree PLY scenes and save one positive instance ID per point.

Use the checkpoint's voxel size and crop radius, with overlapping cylinders.
Scene voxelization retains an inverse map to EVERY original PLY vertex. Points
outside accepted instance masks inherit their nearest predicted instance in XYZ.
Only <scene>.npy files are written to the output directory.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

from tools.infer_satv2 import load_model, prepare_cylinder
from pointcept.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--step", type=float, default=8.0)
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--overlap-threshold", type=float, default=0.1)
    parser.add_argument("--max-voxels", type=int, default=100_000)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        parser.error("invalid worker-index/num-workers")
    if min(args.step, args.max_voxels, args.chunk_size) <= 0:
        parser.error("step, max-voxels and chunk-size must be positive")
    if not 0 <= args.overlap_threshold <= 1:
        parser.error("overlap-threshold must lie in [0, 1]")
    return args


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def voxelize_vertex(vertex, grid_size, chunk_size):
    """Voxelize in local float64 coordinates; keep the original vertex order."""
    if not len(vertex):
        raise ValueError("empty PLY vertex element")
    names = ("x", "y", "z")
    if not all(name in (vertex.dtype.names or ()) for name in names):
        raise ValueError("PLY must contain x, y, z")
    minimum = np.array([vertex[name].min() for name in names], dtype=np.float64)
    maximum = np.array([vertex[name].max() for name in names], dtype=np.float64)
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError("non-finite PLY coordinates")
    dimensions = np.floor((maximum - minimum) / grid_size).astype(np.int64) + 1
    if int(np.prod(dimensions.astype(object))) > np.iinfo(np.int64).max:
        raise ValueError("coordinate range exceeds voxel key capacity")
    codes = np.empty(len(vertex), dtype=np.int64)
    for start in range(0, len(vertex), chunk_size):
        end = min(start + chunk_size, len(vertex))
        key = np.zeros(end - start, dtype=np.int64)
        for axis, name in enumerate(names):
            values = np.asarray(vertex[name][start:end], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("non-finite PLY coordinates")
            cells = np.floor((values - minimum[axis]) / grid_size).astype(np.int64)
            key *= dimensions[axis]
            key += cells
        codes[start:end] = key
    _, representatives, inverse = np.unique(
        codes, return_index=True, return_inverse=True
    )
    coord = np.column_stack(
        [vertex[name][representatives].astype(np.float64) - minimum[i]
         for i, name in enumerate(names)]
    ).astype(np.float32)
    if len(coord) > np.iinfo(np.int32).max:
        raise ValueError("too many voxels for int32 inverse map")
    return coord, inverse.astype(np.int32)


def occupied_centres(coord, step):
    cells = np.unique(np.floor(coord[:, :2] / step).astype(np.int32), axis=0)
    return (cells.astype(np.float64) + 0.5) * step


def mask_candidates(prediction, inverse, indices, coord, centre, radius):
    """Keep mutually exclusive masks in each crop, ordered by confidence."""
    masks = prediction["pred_masks"].numpy()
    scores = prediction["pred_scores"].numpy()
    claimed = np.zeros(len(indices), dtype=bool)
    candidates = []
    for row in np.argsort(-scores, kind="stable"):
        support = masks[row, inverse] & ~claimed
        if support.sum() < 10:
            continue
        claimed[support] = True
        ids = indices[support].astype(np.int32)
        # Prefer complete, central observations over peripheral fragments.
        radial = np.linalg.norm(coord[ids, :2] - centre, axis=1).mean()
        priority = float(scores[row]) * (1.0 - 0.25 * min(radial / radius, 1.0))
        candidates.append((priority, ids))
    return candidates


def merge_candidates(candidates, num_points, overlap_threshold):
    """Confidence-ordered asymmetric suppression and duplicate extension."""
    labels = np.zeros(num_points, dtype=np.int32)
    next_id = 0
    extensions = []
    for score, support in sorted(candidates, key=lambda item: item[0], reverse=True):
        existing = labels[support]
        overlap = existing > 0
        if overlap.mean() > overlap_threshold:
            # Extend a suppressed duplicate only when a single existing
            # instance covers most of it. Avoid bridging neighbouring trees.
            ids, counts = np.unique(existing[overlap], return_counts=True)
            best = int(np.argmax(counts))
            if counts[best] / len(support) >= 0.5:
                extensions.append((int(ids[best]), support[~overlap]))
            continue
        free = support[~overlap]
        if len(free) < 10:
            continue
        next_id += 1
        labels[free] = next_id
    # Finish suppression before extending so fragments cannot suppress
    # distinct lower-confidence predictions.
    for instance_id, support in extensions:
        labels[support[labels[support] == 0]] = instance_id
    return labels


def fill_unassigned(coord, labels, chunk_size):
    assigned = np.flatnonzero(labels > 0)
    if not len(assigned):
        raise RuntimeError("model produced no valid instances; cannot fill labels")
    unassigned = np.flatnonzero(labels == 0)
    if len(unassigned):
        tree = cKDTree(coord[assigned])
        for start in range(0, len(unassigned), chunk_size):
            ids = unassigned[start:start + chunk_size]
            _, neighbours = tree.query(coord[ids], workers=4)
            labels[ids] = labels[assigned[neighbours]]
    if labels.min() <= 0:
        raise RuntimeError("unassigned voxel labels remain")
    return len(unassigned)


def validate_output(path, point_count, chunk_size):
    labels = np.load(path, mmap_mode="r", allow_pickle=False)
    if labels.shape != (point_count,) or labels.dtype != np.dtype("int32"):
        raise ValueError(f"invalid label array {path}: {labels.shape}, {labels.dtype}")
    minimum = np.iinfo(np.int32).max
    maximum = 0
    for start in range(0, point_count, chunk_size):
        part = labels[start:start + chunk_size]
        minimum = min(minimum, int(part.min()))
        maximum = max(maximum, int(part.max()))
    if minimum <= 0:
        raise ValueError(f"unassigned labels in {path}: min={minimum}")
    return maximum


def save_original_labels(path, labels, inverse, chunk_size):
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        output = np.lib.format.open_memmap(
            temp_path, mode="w+", dtype=np.int32, shape=(len(inverse),)
        )
        for start in range(0, len(inverse), chunk_size):
            output[start:start + chunk_size] = labels[inverse[start:start + chunk_size]]
        output.flush()
        del output
        validate_output(temp_path, len(inverse), chunk_size)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def infer_scene(path, output_path, model, cfg, args):
    started = time.monotonic()
    ply = PlyData.read(str(path), mmap="r")
    if "vertex" not in ply:
        raise ValueError(f"{path} has no vertex element")
    vertex = ply["vertex"].data
    log(f"{path.name}: voxelizing {len(vertex):,} original points")
    coord, raw_inverse = voxelize_vertex(vertex, cfg.grid_size, args.chunk_size)
    radius = float(cfg.cylinder_radius)
    xy_tree = cKDTree(coord[:, :2])
    centres = occupied_centres(coord, args.step)
    log(f"{path.name}: {len(coord):,} voxels, {len(centres)} cylinders")
    candidates = []
    rng = np.random.default_rng(args.seed)
    inference_started = time.monotonic()
    for index, centre in enumerate(centres):
        indices = np.asarray(xy_tree.query_ball_point(centre, radius), dtype=np.int64)
        if not len(indices):
            continue
        if len(indices) > args.max_voxels:
            indices = np.sort(rng.choice(indices, args.max_voxels, replace=False))
        batch, inverse = prepare_cylinder(coord[indices], cfg.grid_size, args.device)
        with torch.inference_mode():
            prediction = model(batch)
        candidates.extend(mask_candidates(
            prediction, inverse, indices, coord, centre, radius
        ))
        del batch, prediction
        if (index + 1) % 50 == 0 or index + 1 == len(centres):
            log(f"{path.name}: cylinders {index + 1}/{len(centres)}, "
                f"candidates={len(candidates)}, "
                f"inference_seconds={time.monotonic() - inference_started:.1f}")
    del xy_tree
    labels = merge_candidates(candidates, len(coord), args.overlap_threshold)
    del candidates
    filled = fill_unassigned(coord, labels, args.chunk_size)
    save_original_labels(output_path, labels, raw_inverse, args.chunk_size)
    instances = validate_output(output_path, len(vertex), args.chunk_size)
    log(f"DONE {path.name}: points={len(vertex)}, instances={instances}, "
        f"filled_voxels={filled}/{len(coord)}, labels_min={int(labels.min())}, "
        f"seconds={time.monotonic() - started:.1f}, output={output_path}")


def main():
    args = parse_args()
    cfg = Config.fromfile(str(args.config))
    if args.step > np.sqrt(2) * cfg.cylinder_radius:
        raise ValueError("step exceeds complete cylinder coverage limit")
    paths = sorted(args.input_dir.glob("*.ply"))
    if args.scenes:
        requested = {Path(name).stem for name in args.scenes}
        paths = [path for path in paths if path.stem in requested]
        missing = requested - {path.stem for path in paths}
        if missing:
            raise FileNotFoundError(f"missing scenes: {sorted(missing)}")
    paths = paths[args.worker_index::args.num_workers]
    if not paths:
        raise ValueError("no selected input PLY files")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(torch.device(args.device))
    model = load_model(args.config, args.weight, args.device, args.score_threshold)
    model.sparse_log_interval = 0
    log(f"Loaded checkpoint strictly: {args.weight}; device={args.device}; "
        f"grid_size={cfg.grid_size}; radius={cfg.cylinder_radius}; "
        f"step={args.step}; scenes={len(paths)}")
    for path in paths:
        output_path = args.output_dir / f"{path.stem}.npy"
        if output_path.exists() and not args.overwrite:
            count = len(PlyData.read(str(path), mmap="r")["vertex"].data)
            validate_output(output_path, count, args.chunk_size)
            log(f"SKIP verified existing result: {output_path}")
            continue
        infer_scene(path, output_path, model, cfg, args)
        gc.collect()
        torch.cuda.empty_cache()
    log(f"ALL DONE worker={args.worker_index}: {len(paths)} scenes verified")


if __name__ == "__main__":
    main()
