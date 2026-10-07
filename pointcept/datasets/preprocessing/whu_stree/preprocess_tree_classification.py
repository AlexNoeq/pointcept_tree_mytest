"""Extract tree objects for PTV3 classification; exclude non-tree points."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from species import SPECIES_NAMES


CLASS_NAMES = (*SPECIES_NAMES, "Others")


def groups(instance, eligible):
    indices = np.flatnonzero(eligible)
    if not len(indices):
        return
    indices = indices[np.argsort(instance[indices], kind="stable")]
    ids, starts, counts = np.unique(instance[indices], return_index=True, return_counts=True)
    for tree_id, start, count in zip(ids, starts, counts):
        yield int(tree_id), indices[start:start + count]


def tree_category(segment, purity=0.8):
    """Majority species of annotated tree points; ignore unknown species (-1)."""
    segment = np.asarray(segment)
    if not np.isin(segment, np.arange(-1, 20)).all():
        raise ValueError("semantic labels must be -1..19 from species20 preprocessing")
    known = segment[(segment >= 1) & (segment <= 19)]
    if not len(known):
        return None, 0.0
    counts = np.bincount(known, minlength=20)
    winner = int(counts.argmax())
    ratio = float(counts[winner] / len(known))
    return (winner - 1 if ratio >= purity else None), ratio


def write_tree(directory, coord, strength, category, metadata):
    if len(coord) < 2 or np.ptp(coord, axis=0).max() <= 0:
        return False
    if not np.isfinite(coord).all() or not np.isfinite(strength).all():
        raise ValueError("non-finite tree features")
    temp = directory.with_name(f".{directory.name}.tmp-{os.getpid()}")
    temp.mkdir(parents=True, exist_ok=False)
    try:
        coord = np.asarray(coord, dtype=np.float32).copy()
        coord -= coord.mean(0)
        np.save(temp / "coord.npy", coord)
        np.save(temp / "strength.npy", np.asarray(strength, dtype=np.float32).reshape(-1, 1))
        np.save(temp / "category.npy", np.asarray([category], dtype=np.int64))
        (temp / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(directory)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return True


def scene_records(source_root):
    records = []
    for split in ("train", "val"):
        for scene in sorted((source_root / split).glob("*")):
            if not scene.is_dir() or scene.name.startswith("."):
                continue
            manifest = json.loads((scene / "manifest.json").read_text())
            if manifest.get("semantic_mode") != "species20":
                raise ValueError(f"requires species20 data: {scene}")
            original_split = Path(manifest["source_path"]).parent.name
            if original_split not in ("train", "val"):
                raise ValueError(f"unknown original split: {scene}")
            records.append((scene, original_split, manifest))
    if not records:
        raise ValueError("no species20 scenes found")
    return records


def extract_scene(scene, split, output_root, args, excluded_ids=()):
    record_path = output_root / "records" / f"{split}-{scene.name}.json"
    settings = dict(mode=args.mode, max_points=args.max_points, min_points=args.min_points,
                    purity=args.purity, seed=args.seed, input=str(scene.resolve()),
                    excluded_ids=sorted(int(i) for i in excluded_ids))
    if record_path.exists():
        existing = json.loads(record_path.read_text())
        if existing["settings"] != settings:
            raise ValueError(f"settings changed; use a fresh output root: {record_path}")
        if all((output_root / split / row["sample"] / "metadata.json").exists() for row in existing["trees"]):
            print(f"SKIP {scene.name}: {len(existing['trees'])} trees", flush=True)
            return existing
        raise ValueError(f"incomplete existing scene; use a fresh output root: {scene}")
    if args.mode == "train":
        coord = np.load(scene / "coord.npy", mmap_mode="r")
        strength = np.load(scene / "strength.npy", mmap_mode="r")
        instance = np.load(scene / "instance.npy", mmap_mode="r").reshape(-1)
        semantic = np.load(scene / "segment.npy", mmap_mode="r").reshape(-1)
        eligible = (instance > 0) & (semantic != 0)
    else:
        vertex = PlyData.read(str(args.input_dir / f"{scene.name}.ply"), mmap="r")["vertex"].data
        instance = np.load(scene / "instance.npy", mmap_mode="r").reshape(-1)
        semantic = np.load(scene / "semantic.npy", mmap_mode="r").reshape(-1)
        if len(vertex) != len(instance) or semantic.shape != instance.shape:
            raise ValueError(f"point-count mismatch: {scene}")
        if not np.isin(semantic, (0, 1)).all():
            raise ValueError(f"requires binary SaTv2 semantic predictions: {scene}")
        eligible = (instance > 0) & (semantic == 1)
    if semantic.shape != instance.shape:
        raise ValueError(f"label shape mismatch: {scene}")
    rng = np.random.default_rng(args.seed)
    report = dict(scene=scene.name, split=split, settings=settings, trees=[],
                  excluded_non_tree_points=int(np.count_nonzero(semantic == 0)),
                  skipped_shared_trees=0, skipped_small_trees=0, skipped_unlabeled_or_mixed_trees=0)
    for tree_id, indices in groups(instance, eligible):
        if tree_id in excluded_ids:
            report["skipped_shared_trees"] += 1
            continue
        if len(indices) < args.min_points:
            report["skipped_small_trees"] += 1
            continue
        category, purity = tree_category(semantic[indices], args.purity) if args.mode == "train" else (-1, None)
        if category is None:
            report["skipped_unlabeled_or_mixed_trees"] += 1
            continue
        point_count = len(indices)
        if len(indices) > args.max_points:
            indices = np.sort(rng.choice(indices, args.max_points, replace=False))
        if args.mode == "train":
            xyz, intensity = coord[indices], strength[indices]
        else:
            xyz = np.column_stack([vertex[k][indices] for k in ("x", "y", "z")])
            intensity = vertex["intensity"][indices] / 65535.0
        name = f"{scene.name}__tree_{tree_id}"
        row = dict(sample=name, scene=scene.name, instance_id=tree_id, category=category,
                   semantic_class=category + 1 if category >= 0 else None,
                   point_count=point_count, saved_points=len(indices), label_purity=purity)
        directory = output_root / split / name
        if directory.exists():
            raise ValueError(f"unrecorded sample already exists; use a fresh output root: {directory}")
        if write_tree(directory, xyz, intensity, category, row):
            report["trees"].append(row)
        else:
            report["skipped_small_trees"] += 1
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{scene.name} -> {split}: {len(report['trees'])} trees; excluded {report['excluded_non_tree_points']:,} non-tree points", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "test"))
    parser.add_argument("--source-root", type=Path, default=Path("/home/wyj/225/fgl/data/WHU-Stree-pointcept-20class"))
    parser.add_argument("--prediction-root", type=Path, default=Path("exp/whu_stree/satv2_mamba_aggregation/mamba_best_semantic_instance"))
    parser.add_argument("--input-dir", type=Path, default=Path("/home/wyj/225/fgl/data/test"))
    parser.add_argument("--output-root", type=Path, default=Path("data/whu_stree_treecls_all"))
    parser.add_argument("--max-points", type=int, default=65536)
    parser.add_argument("--min-points", type=int, default=None)
    parser.add_argument("--purity", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenes", nargs="+")
    args = parser.parse_args()
    args.min_points = args.min_points if args.min_points is not None else (64 if args.mode == "train" else 2)
    if args.max_points < args.min_points or args.min_points < 2 or not 0.5 < args.purity <= 1:
        parser.error("invalid point limits/purity")
    if args.mode == "train":
        records = scene_records(args.source_root)
        # Every original train/val scene contributes to the training split.
        records = [(scene, "train", ()) for scene, _, _ in records]
    else:
        records = [(scene, "test", ()) for scene in sorted(args.prediction_root.iterdir())
                   if scene.is_dir() and (scene / "manifest.json").exists()]
        expected = {p.stem for p in args.input_dir.glob("*.ply")}
        if not args.scenes and {scene.name for scene, _, _ in records} != expected:
            raise ValueError("test predictions are incomplete; finish SaTv2 inference first")
    if args.scenes:
        requested = set(args.scenes)
        records = [(scene, split, ids) for scene, split, ids in records if scene.name in requested]
        if {scene.name for scene, _, _ in records} != requested:
            parser.error("some requested scenes are missing")
    if not records:
        parser.error("no input scenes")
    reports = [extract_scene(scene, split, args.output_root, args, ids) for scene, split, ids in records]
    metadata = dict(num_classes=19, class_names=list(CLASS_NAMES),
                    category_to_semantic_class=list(range(1, 20)), non_tree_in_classifier=False)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "classes.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = dict(mode=args.mode, scenes=len(reports), splits={})
    for split in sorted({r["split"] for r in reports}):
        rows = [row for report in reports if report["split"] == split for row in report["trees"]]
        summary["splits"][split] = dict(trees=len(rows), class_counts=np.bincount(
            [r["category"] for r in rows if r["category"] >= 0], minlength=19).tolist())
    (args.output_root / f"{args.mode}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
