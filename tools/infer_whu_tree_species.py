#!/usr/bin/env python3
"""Classify tree instances; save Nx2 [instance, semantic] arrays in PLY order."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from plyfile import PlyData

from pointcept.datasets import build_dataset, collate_fn
from pointcept.datasets.transform import Compose
from pointcept.datasets.preprocessing.whu_stree.preprocess_tree_classification import groups
from pointcept.models import build_model
from pointcept.utils.config import Config


def species_for_points(semantic, instance, tree_predictions):
    """Background and unclassified tree points stay -1; species use IDs 0..18."""
    output = np.full(len(semantic), -1, dtype=np.int16)
    keys = np.array(sorted(tree_predictions), dtype=np.int32)
    if not len(keys):
        return output
    classes = np.array([tree_predictions[int(i)]["semantic_class"] for i in keys], dtype=np.int16)
    if (classes < 0).any() or (classes > 18).any():
        raise ValueError("classified semantic IDs must lie in 0..18")
    positions = np.searchsorted(keys, instance)
    safe = np.minimum(positions, len(keys) - 1)
    matched = (semantic == 1) & (positions < len(keys)) & (keys[safe] == instance)
    output[matched] = classes[safe[matched]]
    return output


def predict_samples(samples, rows, model, cfg, device):
    batch = collate_fn(samples)
    batch.pop("name", None)
    batch.pop("category", None)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.enable_amp):
        probabilities = model(batch)["cls_logits"].float().softmax(-1).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise RuntimeError("non-finite classification predictions")
    result = {}
    for row, probs in zip(rows, probabilities):
        category = int(probs.argmax())
        result[row["instance_id"]] = dict(
            **row, category=category, semantic_class=category,
            species=cfg.class_names[category], confidence=float(probs[category]),
        )
    return result


def classify_raw_scene(path, semantic, instance, model, cfg, args):
    vertex = PlyData.read(str(path), mmap="r")["vertex"].data
    if semantic.shape != (len(vertex),) or instance.shape != semantic.shape:
        raise ValueError(f"PLY and label point counts differ: {path}")
    if not np.isin(semantic, (0, 1)).all() or not np.issubdtype(instance.dtype, np.integer) or (instance < 0).any():
        raise ValueError(f"invalid binary semantic/instance labels: {path}")
    transform = Compose(cfg.data.test.transform)
    point_limit = min(cfg.data.test.num_points, args.max_saved_points)
    samples, rows, predictions, skipped = [], [], {}, []
    for tree_id, indices in groups(instance, (semantic == 1) & (instance > 0)):
        count = len(indices)
        if count > point_limit:
            rng = np.random.default_rng((cfg.seed or 0) + tree_id)
            indices = np.sort(rng.choice(indices, point_limit, replace=False))
        coord = np.column_stack([vertex[k][indices] for k in ("x", "y", "z")]).astype(np.float32)
        strength = (vertex["intensity"][indices] / 65535.0).astype(np.float32).reshape(-1, 1)
        if not np.isfinite(coord).all() or not np.isfinite(strength).all():
            raise ValueError(f"non-finite features: {path}, instance {tree_id}")
        if len(coord) < 2 or np.ptp(coord, axis=0).max() <= 0:
            skipped.append(tree_id)
            continue
        coord -= coord.mean(0)
        data = dict(coord=coord, strength=strength, category=np.array([-1], dtype=np.int64),
                    name=f"{path.stem}__tree_{tree_id}")
        samples.append(transform(data))
        rows.append(dict(instance_id=tree_id, point_count=count))
        if len(samples) == args.batch_size:
            predictions.update(predict_samples(samples, rows, model, cfg, args.device))
            samples, rows = [], []
    if samples:
        predictions.update(predict_samples(samples, rows, model, cfg, args.device))
    return predictions, skipped


def save_scene_result(output_dir, scene, semantic, instance, trees, weight, source, skipped=()):
    """Write both columns atomically, preserving every original instance ID."""
    target = output_dir / f"{scene}.npy"
    if target.exists():
        raise ValueError(f"output already exists; use a fresh output-dir: {target}")
    temp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    unclassified = 0
    try:
        output = np.lib.format.open_memmap(temp, mode="w+", dtype=np.int32, shape=(len(semantic), 2))
        for start in range(0, len(semantic), 1000000):
            end = min(start + 1000000, len(semantic))
            species = species_for_points(semantic[start:end], instance[start:end], trees)
            output[start:end, 0] = instance[start:end]
            output[start:end, 1] = species
            unclassified += int(np.count_nonzero((semantic[start:end] == 1) & (species == -1)))
        output.flush()
        del output
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    report = dict(scene=scene, columns=["instance", "semantic"], trees=list(trees.values()),
                  points=len(semantic), skipped_degenerate_instances=list(skipped),
                  unclassified_tree_points=unclassified, weight=str(weight.resolve()),
                  source_predictions=str(source.resolve()))
    (output_dir / f"{scene}_trees.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{scene}: classified {len(trees)} trees; saved {target} ({len(semantic)}, 2)", flush=True)
    return dict(scene=scene, classified_trees=len(trees), points=len(semantic), unclassified_tree_points=unclassified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/whu_stree/cls-ptv3-v1m1-0-tree-species.py"))
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, help="Raw PLY directory; classify directly without test preprocessing")
    parser.add_argument("--data-root", type=Path, default=Path("data/whu_stree_treecls_all"))
    parser.add_argument("--prediction-root", type=Path, default=Path("exp/whu_stree/satv2_mamba_aggregation/mamba_best_semantic_instance"))
    parser.add_argument("--output-dir", type=Path, default=Path("exp/whu_stree/cls-ptv3-tree-species-all/test_predictions"))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-saved-points", type=int, default=65536,
                        help="Match the training preprocessing point cap in direct PLY mode")
    parser.add_argument("--scenes", nargs="+")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_saved_points < 2:
        parser.error("batch-size/max-saved-points must be positive")
    cfg = Config.fromfile(str(args.config))
    classes = list(cfg.class_names)
    if args.input_dir is None:
        if json.loads((args.data_root / "classes.json").read_text())["class_names"] != classes:
            raise ValueError("prepared data and config have different class mappings")
    if not args.weight.is_file():
        parser.error(f"checkpoint not found: {args.weight}")
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed or 0)
    np.random.seed(cfg.seed or 0)
    torch.cuda.set_device(torch.device(args.device))
    model = build_model(cfg.model)
    checkpoint = torch.load(args.weight, map_location="cpu", weights_only=False)
    model.load_state_dict({k.removeprefix("module."): v for k, v in checkpoint["state_dict"].items()}, strict=True)
    model.to(args.device).eval()
    del checkpoint
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.input_dir is not None:
        paths = sorted(args.input_dir.glob("*.ply"))
        if args.scenes:
            selected = {Path(name).stem for name in args.scenes}
            paths = [p for p in paths if p.stem in selected]
            if {p.stem for p in paths} != selected:
                parser.error("some requested scenes are missing")
        if not paths:
            parser.error("no PLY files found")
        reports = []
        for path in paths:
            source = args.prediction_root / path.stem
            semantic = np.load(source / "semantic.npy", mmap_mode="r")
            instance = np.load(source / "instance.npy", mmap_mode="r")
            print(f"Classifying {path.name} ...", flush=True)
            trees, skipped = classify_raw_scene(path, semantic, instance, model, cfg, args)
            reports.append(save_scene_result(args.output_dir, path.stem, semantic, instance,
                                             trees, args.weight, source, skipped))
        write_manifest(args.output_dir, classes, reports)
        return
    dataset_cfg = cfg.data.test.copy()
    dataset_cfg.update(data_root=str(args.data_root), split="test")
    dataset = build_dataset(dataset_cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    per_scene = {}
    with torch.inference_mode():
        for batch in loader:
            names = batch.pop("name")
            batch.pop("category")
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.enable_amp):
                probabilities = model(batch)["cls_logits"].float().softmax(-1).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise RuntimeError("non-finite classification predictions")
            for name, probs in zip(names, probabilities):
                metadata = json.loads((args.data_root / "test" / name / "metadata.json").read_text())
                category = int(probs.argmax())
                row = dict(instance_id=metadata["instance_id"], category=category, semantic_class=category,
                           species=classes[category], confidence=float(probs[category]), point_count=metadata["point_count"])
                per_scene.setdefault(metadata["scene"], {})[metadata["instance_id"]] = row
    scene_records = sorted((args.data_root / "records").glob("test-*.json"))
    reports = []
    for record_path in scene_records:
        scene = json.loads(record_path.read_text())["scene"]
        tree_predictions = per_scene.get(scene, {})
        source = args.prediction_root / scene
        semantic = np.load(source / "semantic.npy", mmap_mode="r")
        instance = np.load(source / "instance.npy", mmap_mode="r")
        reports.append(save_scene_result(args.output_dir, scene, semantic, instance,
                                         tree_predictions, args.weight, source))
    write_manifest(args.output_dir, classes, reports)


def write_manifest(output_dir, classes, reports):
    metadata = dict(columns=["instance", "semantic"], dtype="int32", shape="(N, 2)",
                    class_names=classes, semantic_class_ids=list(range(len(classes))),
                    semantic_format="-1=non-tree or unclassified tree; 0..18=species",
                    instance_labels="Original instance IDs; original PLY vertex order preserved", scenes=reports)
    (output_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
