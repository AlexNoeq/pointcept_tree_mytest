#!/usr/bin/env python3
"""Convert FOR-instance PLY files to Pointcept's scene-directory layout.

The released FOR-instance data uses ``x/y/z``, ``semantic_seg`` (1=ground,
2=wood, 3=leaf), and ``treeID`` (>0 for annotated trees).  Property names can
be overridden for compatible forest datasets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData


SEMANTIC_CANDIDATES = ("semantic_seg", "semantic", "segment", "class", "label")
INSTANCE_CANDIDATES = ("treeID", "tree_id", "tree", "instance", "instance_id")
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=("train", "val", "test"))
    parser.add_argument("--semantic-property")
    parser.add_argument("--instance-property")
    parser.add_argument(
        "--semantic-offset",
        type=int,
        choices=(0, 1),
        default=1,
        help="subtract this value from semantic labels (FOR-instance uses 1)",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.source_root.resolve() == args.output_root.resolve():
        parser.error("source and output roots must differ")
    return args


def choose_property(names, requested, candidates, kind, path):
    if requested:
        if requested not in names:
            raise ValueError(f"{path}: requested {kind} property '{requested}' is absent")
        return requested
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise ValueError(f"{path}: no {kind} property found; tried {candidates}")


def integral(values, field, path):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: '{field}' contains NaN or Inf")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0, atol=1e-5):
        raise ValueError(f"{path}: '{field}' contains non-integral labels")
    return rounded.astype(np.int64, copy=False)


def normalize_semantic(values, field, path, offset):
    values = integral(values, field, path)
    normalized = values - offset
    unique = np.unique(normalized)
    if np.isin(unique, (0, 1, 2)).all():
        return normalized.astype(np.int16, copy=False)
    raise ValueError(
        f"{path}: semantic labels minus offset {offset} must be 0/1/2; "
        f"got {unique.tolist()}"
    )


def infer_flat_split(path):
    match = re.search(r"_(train|val|test)$", path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"{path}: flat input requires a _train/_val/_test filename suffix"
        )
    return match.group(1).lower()


def discover(source_root, requested_splits):
    scenes = []
    has_split_dirs = any((source_root / split).is_dir() for split in SPLITS)
    if has_split_dirs:
        for split in requested_splits:
            split_dir = source_root / split
            if split_dir.is_dir():
                scenes.extend((split, path) for path in sorted(split_dir.glob("*.ply")))
    else:
        for path in sorted(source_root.glob("*.ply")):
            split = infer_flat_split(path)
            if split in requested_splits:
                scenes.append((split, path))
    if not scenes:
        raise FileNotFoundError(f"no matching PLY scenes under {source_root}")
    return scenes


def convert(split, source, output_root, args):
    target = output_root / split / source.stem
    if target.exists():
        if not args.force:
            print(f"SKIP {target}")
            return
    ply = PlyData.read(str(source), mmap="r")
    if "vertex" not in ply:
        raise ValueError(f"{source}: missing vertex element")
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    missing_xyz = [name for name in ("x", "y", "z") if name not in names]
    if missing_xyz:
        raise ValueError(f"{source}: missing coordinate fields {missing_xyz}")
    semantic_field = choose_property(
        names, args.semantic_property, SEMANTIC_CANDIDATES, "semantic", source
    )
    instance_field = choose_property(
        names, args.instance_property, INSTANCE_CANDIDATES, "instance", source
    )
    if args.dry_run:
        print(
            f"DRY-RUN {source} -> {target} ({len(vertex):,} points, "
            f"semantic={semantic_field}, instance={instance_field})"
        )
        return

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        count = len(vertex)
        coord = np.lib.format.open_memmap(
            temporary / "coord.npy", mode="w+", dtype=np.float32, shape=(count, 3)
        )
        segment = np.lib.format.open_memmap(
            temporary / "segment.npy", mode="w+", dtype=np.int16, shape=(count,)
        )
        instance = np.lib.format.open_memmap(
            temporary / "instance.npy", mode="w+", dtype=np.int32, shape=(count,)
        )
        semantic_counts = np.zeros(3, dtype=np.int64)
        instance_ids = set()
        for start in range(0, count, args.chunk_size):
            end = min(start + args.chunk_size, count)
            xyz = np.column_stack(
                [np.asarray(vertex[name][start:end]) for name in ("x", "y", "z")]
            )
            if not np.all(np.isfinite(xyz)):
                raise ValueError(f"{source}: coordinates contain NaN or Inf")
            semantic = normalize_semantic(
                vertex[semantic_field][start:end],
                semantic_field,
                source,
                args.semantic_offset,
            )
            raw_instance = integral(
                vertex[instance_field][start:end], instance_field, source
            )
            mapped_instance = np.where(raw_instance > 0, raw_instance, -1)
            if mapped_instance.max(initial=-1) > np.iinfo(np.int32).max:
                raise ValueError(f"{source}: instance IDs exceed int32")
            coord[start:end] = xyz.astype(np.float32, copy=False)
            segment[start:end] = semantic
            instance[start:end] = mapped_instance.astype(np.int32, copy=False)
            semantic_counts += np.bincount(semantic, minlength=3)
            instance_ids.update(np.unique(raw_instance[raw_instance > 0]).tolist())
        for array in (coord, segment, instance):
            array.flush()
        del coord, segment, instance
        manifest = {
            "source": str(source.resolve()),
            "split": split,
            "point_count": count,
            "semantic_property": semantic_field,
            "instance_property": instance_field,
            "semantic_mapping": {"0": "ground", "1": "wood", "2": "leaf"},
            "semantic_offset": args.semantic_offset,
            "semantic_point_counts": semantic_counts.tolist(),
            "tree_instance_count": len(instance_ids),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
        print(f"WROTE {target}: {count:,} points, {len(instance_ids):,} trees")
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for split, source in discover(source_root, set(args.splits)):
        convert(split, source, output_root, args)


if __name__ == "__main__":
    main()
