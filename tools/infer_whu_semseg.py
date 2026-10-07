#!/usr/bin/env python3
"""Infer unlabeled WHU-STree PLYs; save class IDs in original vertex order."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

from pointcept.models import build_model
from pointcept.utils.config import Config


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def voxelize(vertex, grid_size, chunk_size):
    """Keep one representative per voxel and an inverse for every raw point."""
    names = ("x", "y", "z")
    if not len(vertex) or not set((*names, "intensity")) <= set(vertex.dtype.names or ()):
        raise ValueError("PLY requires nonempty x/y/z/intensity vertices")
    # Match preprocessing's float32 coordinates before subtracting the origin.
    minimum = np.array([np.float32(vertex[k].min()) for k in names], dtype=np.float64)
    maximum = np.array([np.float32(vertex[k].max()) for k in names], dtype=np.float64)
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError("non-finite coordinates")
    dimensions = np.floor((maximum - minimum) / grid_size).astype(np.int64) + 1
    if int(np.prod(dimensions.astype(object))) > np.iinfo(np.int64).max:
        raise ValueError("voxel keys exceed int64 capacity")
    codes = np.empty(len(vertex), dtype=np.int64)
    for start in range(0, len(vertex), chunk_size):
        end = min(start + chunk_size, len(vertex))
        key = np.zeros(end - start, dtype=np.int64)
        for axis, name in enumerate(names):
            values = vertex[name][start:end].astype(np.float32).astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("non-finite coordinates")
            key *= dimensions[axis]
            key += np.floor((values - minimum[axis]) / grid_size).astype(np.int64)
        codes[start:end] = key
    _, representatives, inverse = np.unique(codes, return_index=True, return_inverse=True)
    coord = np.column_stack([
        vertex[k][representatives].astype(np.float32).astype(np.float64) - minimum[i]
        for i, k in enumerate(names)
    ]).astype(np.float32)
    strength = (vertex["intensity"][representatives] / 65535.0).astype(np.float32)
    if not np.isfinite(strength).all() or (strength < 0).any():
        raise ValueError("invalid intensity")
    return coord, strength, inverse.astype(np.int32)


def prepare_crop(coord, strength, grid_size, device):
    coord = coord.copy()
    lower, upper = coord.min(0), coord.max(0)
    coord -= np.array([(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, lower[2]])
    grid = np.floor(coord / grid_size).astype(np.int32)
    grid -= grid.min(0)
    _, representatives, inverse = np.unique(grid, axis=0, return_index=True, return_inverse=True)
    coord = coord[representatives]
    lower, upper = coord.min(0), coord.max(0)
    coord -= np.array([(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, 0])
    feat = np.column_stack((coord, strength[representatives])).astype(np.float32)
    return {
        "coord": torch.from_numpy(coord).to(device),
        "grid_coord": torch.from_numpy(grid[representatives]).to(device),
        "feat": torch.from_numpy(feat).to(device),
        "offset": torch.tensor([len(coord)], dtype=torch.int32, device=device),
    }, inverse


def split_crop(indices, coord, limit):
    """Spatially split oversized crops without discarding any input voxels."""
    if len(indices) <= limit:
        yield indices
        return
    axis = np.ptp(coord[indices, :2], axis=0).argmax()
    order = np.argsort(coord[indices, axis], kind="stable")
    middle = len(indices) // 2
    yield from split_crop(indices[order[:middle]], coord, limit)
    yield from split_crop(indices[order[middle:]], coord, limit)


def validate_output(path, count, num_classes, chunk_size):
    labels = np.load(path, mmap_mode="r", allow_pickle=False)
    if labels.shape != (count,) or labels.dtype != np.uint8:
        raise ValueError(f"invalid output shape/dtype: {path}")
    histogram = np.zeros(num_classes, dtype=np.int64)
    for start in range(0, count, chunk_size):
        part = labels[start:start + chunk_size]
        if (part >= num_classes).any():
            raise ValueError(f"invalid class IDs: {path}")
        histogram += np.bincount(part, minlength=num_classes)
    return histogram.tolist()


def infer_scene(path, output_path, model, cfg, args):
    started = time.monotonic()
    vertex = PlyData.read(str(path), mmap="r")["vertex"].data
    if output_path.exists():
        counts = validate_output(output_path, len(vertex), cfg.num_classes, args.chunk_size)
        log(f"{path.name}: existing output verified, skipping")
        return dict(scene=path.stem, points=len(vertex), class_counts=counts, skipped=True)
    log(f"{path.name}: voxelizing {len(vertex):,} points")
    coord, strength, raw_inverse = voxelize(vertex, cfg.grid_size, args.chunk_size)
    cells = np.unique(np.floor(coord[:, :2] / args.step).astype(np.int32), axis=0)
    centres = (cells.astype(np.float64) + 0.5) * args.step
    tree = cKDTree(coord[:, :2])
    scores = np.zeros((len(coord), cfg.num_classes), dtype=np.float32)
    votes = np.zeros(len(coord), dtype=np.int32)
    log(f"{path.name}: {len(coord):,} voxels, {len(centres)} cylinders")
    with torch.inference_mode():
        for index, centre in enumerate(centres):
            indices = np.asarray(tree.query_ball_point(centre, cfg.cylinder_radius), dtype=np.int64)
            for ids in split_crop(indices, coord, args.max_voxels):
                if not len(ids):
                    continue
                batch, inverse = prepare_crop(coord[ids], strength[ids], cfg.grid_size, args.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.enable_amp):
                    logits = model(batch)["seg_logits"]
                probabilities = logits.float().softmax(-1).cpu().numpy()
                if not np.isfinite(probabilities).all():
                    raise RuntimeError(f"{path.name}: non-finite model predictions")
                scores[ids] += probabilities[inverse]
                votes[ids] += 1
                del batch, logits, probabilities
            if (index + 1) % 10 == 0 or index + 1 == len(centres):
                log(f"{path.name}: cylinders {index + 1}/{len(centres)}, elapsed={time.monotonic() - started:.1f}s")
    if (votes == 0).any():
        raise RuntimeError(f"{int((votes == 0).sum())} voxels have no prediction")
    labels = scores.argmax(1).astype(np.uint8)
    temp_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        output = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.uint8, shape=(len(vertex),))
        for start in range(0, len(vertex), args.chunk_size):
            output[start:start + args.chunk_size] = labels[raw_inverse[start:start + args.chunk_size]]
        output.flush()
        del output
        counts = validate_output(temp_path, len(vertex), cfg.num_classes, args.chunk_size)
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)
    report = dict(scene=path.stem, points=len(vertex), voxels=len(coord), cylinders=len(centres),
                  class_counts=counts, seconds=round(time.monotonic() - started, 2))
    log(f"{path.name}: saved {output_path.name}; {report['seconds']}s")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--step", type=float, default=16.0)
    parser.add_argument("--max-voxels", type=int, default=100000)
    parser.add_argument("--chunk-size", type=int, default=1000000)
    args = parser.parse_args()
    cfg = Config.fromfile(str(args.config))
    if not 0 < args.step <= cfg.cylinder_radius * np.sqrt(2):
        parser.error("step must be positive and <= radius * sqrt(2) for full coverage")
    if min(args.max_voxels, args.chunk_size) <= 0:
        parser.error("max-voxels and chunk-size must be positive")
    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        parser.error("invalid worker-index/num-workers")
    paths = sorted(args.input_dir.glob("*.ply"))
    if args.scenes:
        selected = list(dict.fromkeys(Path(name).stem for name in args.scenes))
        available = {p.stem: p for p in paths}
        if not set(selected) <= available.keys():
            parser.error("some requested scenes were not found")
        paths = [available[name] for name in selected]
    if not paths:
        parser.error("no input PLYs found")
    paths = paths[args.worker_index::args.num_workers]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Volt's FlashAttention")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed or 0)
    model = build_model(cfg.model)
    checkpoint = torch.load(args.weight, map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["state_dict"].items()}, strict=True)
    model.to(args.device).eval()
    checkpoint_epoch = checkpoint.get("epoch")
    log(f"Loaded {args.weight}, checkpoint epoch={checkpoint_epoch}; {len(paths)} scenes")
    del checkpoint
    metadata = dict(config=str(args.config.resolve()), weight=str(args.weight.resolve()),
                    checkpoint_epoch=checkpoint_epoch,
                    input_dir=str(args.input_dir.resolve()), grid_size=cfg.grid_size,
                    radius=cfg.cylinder_radius, step=args.step, max_voxels=args.max_voxels,
                    intensity_scale=65535.0, class_names=list(cfg.class_names),
                    output_format="<scene>_pred.npy: uint8 class IDs 0..19 in original PLY vertex order",
                    scenes=[])
    for index, path in enumerate(paths):
        log(f"Scene {index + 1}/{len(paths)}")
        metadata["scenes"].append(infer_scene(path, args.output_dir / f"{path.stem}_pred.npy", model, cfg, args))
        report_name = "manifest.json" if args.num_workers == 1 else f"manifest_worker_{args.worker_index}.json"
        report_path = args.output_dir / report_name
        temp_report = report_path.with_suffix(".json.tmp")
        temp_report.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_report.replace(report_path)
        gc.collect()
        torch.cuda.empty_cache()
    log("All scenes completed and validated")


if __name__ == "__main__":
    main()
