#!/usr/bin/env python3
"""Save SaTv2's binary semantic and tree-instance predictions per raw PLY point."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

from pointcept.utils.config import Config
from tools.infer_satv2 import load_model, prepare_cylinder
from tools.infer_whu_instances import voxelize_vertex, occupied_centres, mask_candidates, merge_candidates
from tools.infer_whu_semseg import split_crop


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def finish_instances(coord, instance, semantic, chunk_size):
    """Preserve semantic background; fill only tree points from tree masks."""
    instance[semantic == 0] = 0
    assigned = np.flatnonzero(instance > 0)
    missing = np.flatnonzero((semantic == 1) & (instance == 0))
    filled = 0
    if len(assigned) and len(missing):
        tree = cKDTree(coord[assigned])
        for start in range(0, len(missing), chunk_size):
            ids = missing[start:start + chunk_size]
            _, neighbours = tree.query(coord[ids], workers=4)
            instance[ids] = instance[assigned[neighbours]]
        filled = len(missing)
    # If a scene has no accepted instance masks, leave its tree points at ID 0
    # and record them as unassigned rather than inventing a tree instance.
    positive = instance > 0
    ids, inverse = np.unique(instance[positive], return_inverse=True)
    instance[positive] = inverse.astype(np.int32) + 1
    return len(ids), filled, int(np.count_nonzero((semantic == 1) & (instance == 0)))


def validate_scene(directory, point_count, chunk_size):
    semantic = np.load(directory / "semantic.npy", mmap_mode="r", allow_pickle=False)
    instance = np.load(directory / "instance.npy", mmap_mode="r", allow_pickle=False)
    if semantic.shape != (point_count,) or instance.shape != (point_count,):
        raise ValueError(f"point-count mismatch in {directory}")
    if semantic.dtype != np.uint8 or instance.dtype != np.int32:
        raise ValueError(f"invalid prediction dtype in {directory}")
    counts = np.zeros(2, dtype=np.int64)
    unassigned = 0
    for start in range(0, point_count, chunk_size):
        sem = semantic[start:start + chunk_size]
        inst = instance[start:start + chunk_size]
        if (sem > 1).any() or (inst < 0).any() or (inst[sem == 0] != 0).any():
            raise ValueError(f"invalid background/instance labels in {directory}")
        counts += np.bincount(sem, minlength=2)
        unassigned += int(np.count_nonzero((sem == 1) & (inst == 0)))
    return dict(points=point_count, non_tree_points=int(counts[0]), tree_points=int(counts[1]),
                unassigned_tree_points=unassigned, instances=int(instance.max(initial=0)))


def infer_scene(path, target, model, cfg, args):
    vertex = PlyData.read(str(path), mmap="r")["vertex"].data
    if target.exists():
        report = json.loads((target / "manifest.json").read_text())
        if report["weight"] != str(args.weight.resolve()) or report["grid_size"] != cfg.grid_size:
            raise ValueError(f"existing result has different settings: {target}")
        report.update(validate_scene(target, len(vertex), args.chunk_size))
        log(f"SKIP verified {path.name}")
        return report
    started = time.monotonic()
    log(f"{path.name}: voxelizing {len(vertex):,} raw points")
    coord, raw_inverse = voxelize_vertex(vertex, cfg.grid_size, args.chunk_size)
    centres = occupied_centres(coord, args.step)
    tree = cKDTree(coord[:, :2])
    probabilities = np.zeros((len(coord), 2), dtype=np.float32)
    votes = np.zeros(len(coord), dtype=np.int32)
    candidates = []
    log(f"{path.name}: {len(coord):,} voxels, {len(centres)} cylinders")
    with torch.inference_mode():
        for index, centre in enumerate(centres):
            indices = np.asarray(tree.query_ball_point(centre, cfg.cylinder_radius), dtype=np.int64)
            for ids in split_crop(indices, coord, args.max_voxels):
                if not len(ids):
                    continue
                batch, inverse = prepare_cylinder(coord[ids], cfg.grid_size, args.device)
                prediction = model(batch)
                probs = prediction["seg_logits"].float().softmax(-1).cpu().numpy()
                if not np.isfinite(probs).all():
                    raise RuntimeError(f"non-finite predictions for {path.name}")
                probabilities[ids] += probs[inverse]
                votes[ids] += 1
                candidates.extend(mask_candidates(prediction, inverse, ids, coord, centre, cfg.cylinder_radius))
                del batch, prediction, probs
            if (index + 1) % 50 == 0 or index + 1 == len(centres):
                log(f"{path.name}: cylinders {index + 1}/{len(centres)}, candidates={len(candidates)}, elapsed={time.monotonic() - started:.1f}s")
    if (votes == 0).any():
        raise RuntimeError("some voxels have no semantic prediction")
    semantic = probabilities.argmax(1).astype(np.uint8)
    del tree, probabilities, votes
    instance = merge_candidates(candidates, len(coord), args.overlap_threshold)
    del candidates
    _, filled, _ = finish_instances(coord, instance, semantic, args.chunk_size)
    temp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temp.mkdir(parents=True, exist_ok=False)
    try:
        for name, labels in (("semantic", semantic), ("instance", instance)):
            output = np.lib.format.open_memmap(temp / f"{name}.npy", mode="w+", dtype=labels.dtype, shape=(len(vertex),))
            for start in range(0, len(vertex), args.chunk_size):
                output[start:start + args.chunk_size] = labels[raw_inverse[start:start + args.chunk_size]]
            output.flush()
            del output
        report = dict(scene=path.stem, input=str(path.resolve()), weight=str(args.weight.resolve()),
                      config=str(args.config.resolve()), grid_size=cfg.grid_size, radius=cfg.cylinder_radius,
                      step=args.step, max_voxels=args.max_voxels, score_threshold=args.score_threshold,
                      overlap_threshold=args.overlap_threshold, filled_tree_voxels=filled,
                      seconds=round(time.monotonic() - started, 2), **validate_scene(temp, len(vertex), args.chunk_size))
        (temp / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temp.replace(target)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    log(f"DONE {path.name}: trees={report['instances']}, non_tree_points={report['non_tree_points']:,}, seconds={report['seconds']}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/home/wyj/225/fgl/data/test"))
    parser.add_argument("--output-dir", type=Path, default=Path("exp/whu_stree/satv2_mamba_aggregation/mamba_best_semantic_instance"))
    parser.add_argument("--config", type=Path, default=Path("exp/whu_stree/satv2_mamba_aggregation/config.py"))
    parser.add_argument("--weight", type=Path, default=Path("exp/whu_stree/satv2_mamba_aggregation/model/model_best.pth"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--step", type=float, default=8.0)
    parser.add_argument("--max-voxels", type=int, default=100000)
    parser.add_argument("--chunk-size", type=int, default=1000000)
    parser.add_argument("--score-threshold", type=float, default=0.15)
    parser.add_argument("--overlap-threshold", type=float, default=0.1)
    args = parser.parse_args()
    cfg = Config.fromfile(str(args.config))
    if not 0 < args.step <= np.sqrt(2) * cfg.cylinder_radius or min(args.max_voxels, args.chunk_size) <= 0:
        parser.error("invalid step/point limits")
    if not 0 <= args.overlap_threshold <= 1 or not 0 <= args.score_threshold <= 1:
        parser.error("thresholds must lie in [0,1]")
    paths = sorted(args.input_dir.glob("*.ply"))
    if args.scenes:
        requested = {Path(s).stem for s in args.scenes}
        paths = [p for p in paths if p.stem in requested]
        if {p.stem for p in paths} != requested:
            parser.error("some scenes are missing")
    if not paths:
        parser.error("no input PLYs")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_device(torch.device(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.config, args.weight, args.device, args.score_threshold)
    model.sparse_log_interval = 0
    log(f"Strictly loaded {args.weight} on {args.device}; {len(paths)} scenes")
    reports = []
    for index, path in enumerate(paths):
        log(f"Scene {index + 1}/{len(paths)}")
        reports.append(infer_scene(path, args.output_dir / path.stem, model, cfg, args))
        metadata = dict(weight=str(args.weight.resolve()), device=args.device, scenes=reports,
                        label_format="<scene>/semantic.npy (0=non-tree, 1=tree); instance.npy (0=background/unassigned, positive=tree ID); original PLY vertex order")
        temp = args.output_dir / "manifest.json.tmp"
        temp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temp.replace(args.output_dir / "manifest.json")
        gc.collect()
        torch.cuda.empty_cache()
    log("ALL DONE: semantic and instance predictions validated")


if __name__ == "__main__":
    main()
