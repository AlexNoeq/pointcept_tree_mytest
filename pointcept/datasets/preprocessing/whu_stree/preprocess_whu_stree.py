"""Stream WHU-Stree PLY scenes into Pointcept's standard dataset layout.

The raw PLY vertex properties must include ``x y z intensity tree``.  The
By default, the ``tree`` property is the only label used: ``tree == 0`` is semantic
``non-tree`` (class 0), and every non-zero value is semantic ``tree`` (class
1).  Non-tree points receive instance ``-1``; tree points keep their raw
``tree`` instance ID.

With --semantic-mode species20, also read the official ``label`` species field:
non-tree=0, raw species 0..17 -> 1..18, raw species >=18 -> Others=19.
Tree points without species annotations retain the semantic ignore label -1.

The source clouds can contain tens of millions of points.  This script uses a
memory-mapped PLY reader and writes NPY arrays in chunks, so it does not load a
whole cloud into RAM.  Output is written to a temporary sibling directory and
atomically renamed only after all assets have been completed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# Support both direct CLI execution and lightweight imports in tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from species import CLASS_NAMES, SEMANTIC_MAPPING, class_metadata, map_species_labels

try:
    from plyfile import PlyData
except ImportError as exc:  # pragma: no cover - actionable CLI error
    raise SystemExit(
        "Missing dependency 'plyfile'. Install it with: pip install plyfile"
    ) from exc


REQUIRED_FIELDS = ("x", "y", "z", "intensity", "tree")
OUTPUT_ASSETS = ("coord.npy", "strength.npy", "segment.npy", "instance.npy")
DEFAULT_SOURCE_ROOT = "/home/wyj/225/fgl/data/WHU-Stree_random50"
DEFAULT_OUTPUT_ROOT = "/home/wyj/225/fgl/data/WHU-Stree-pointcept"
SPECIES_OUTPUT_ROOT = "/home/wyj/225/fgl/data/WHU-Stree-pointcept-20class"
OUTPUT_FORMAT_VERSION = 4
NUM_SEMANTIC_CLASSES = 2
NON_TREE_LABEL = 0
TREE_LABEL = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(DEFAULT_SOURCE_ROOT),
        help="Raw WHU-Stree directory containing train/ and val/ PLY files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Destination Pointcept dataset root (kept separate from raw PLY files).",
    )
    parser.add_argument(
        "--semantic-mode", choices=("binary", "species20"), default="binary",
        help="Binary tree/non-tree or official 18 species + Others + background.",
    )
    parser.add_argument(
        "--merge-splits-to-train", action="store_true",
        help="Write all selected source splits into train/; reject duplicate scene names.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val"),
        choices=("train", "val", "test"),
        help="Raw split directories to convert. The supplied evaluation folder is val/.",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional PLY filenames (or stems) to process for a small smoke run.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Points written per chunk; lower it if host memory is constrained.",
    )
    parser.add_argument(
        "--intensity-scale",
        type=float,
        default=65535.0,
        help="Divisor for strength.npy. WHU-Stree stores 8-bit values in 16-bit range.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing processed scene. Never touches raw PLY files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selected PLY headers and list scenes without writing output.",
    )
    args = parser.parse_args()
    if args.output_root is None:
        args.output_root = Path(
            SPECIES_OUTPUT_ROOT if args.semantic_mode == "species20" else DEFAULT_OUTPUT_ROOT
        )
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if not np.isfinite(args.intensity_scale) or args.intensity_scale <= 0:
        parser.error("--intensity-scale must be a finite positive number")
    return args


def open_vertex_data(path: Path, semantic_mode: str = "binary") -> np.ndarray:
    """Open a binary PLY vertex element as a memory mapped structured array."""
    try:
        ply = PlyData.read(str(path), mmap="r")
    except Exception as exc:
        raise ValueError(f"Could not read PLY file: {path}") from exc

    if "vertex" not in ply:
        raise ValueError(f"{path} has no vertex element")
    vertex = ply["vertex"].data
    if vertex.dtype.names is None:
        raise ValueError(f"{path} vertex element has no named properties")
    fields = REQUIRED_FIELDS + (("label",) if semantic_mode == "species20" else ())
    missing = [field for field in fields if field not in vertex.dtype.names]
    if missing:
        raise ValueError(f"{path} is missing required vertex properties: {missing}")
    if not np.issubdtype(vertex.dtype["x"], np.number):
        raise ValueError(f"{path} has a non-numeric x property")
    return vertex


def scene_paths(source_root: Path, splits: Iterable[str], scenes: list[str] | None):
    selected = None
    if scenes:
        selected = {name if name.endswith(".ply") else f"{name}.ply" for name in scenes}

    for split in splits:
        split_dir = source_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Requested split directory does not exist: {split_dir}")
        paths = sorted(split_dir.glob("*.ply"))
        if selected is not None:
            paths = [path for path in paths if path.name in selected]
        if not paths:
            requested = f" matching {sorted(selected)}" if selected else ""
            raise FileNotFoundError(f"No PLY files found in {split_dir}{requested}")
        yield split, paths


def is_complete_scene(scene_dir: Path, semantic_mode="binary", intensity_scale=None) -> bool:
    manifest_path = scene_dir / "manifest.json"
    if not (
        scene_dir.is_dir()
        and all((scene_dir / asset).is_file() for asset in OUTPUT_ASSETS)
        and manifest_path.is_file()
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("format_version") == OUTPUT_FORMAT_VERSION
        and manifest.get("semantic_mode", "binary") == semantic_mode
        and (intensity_scale is None or manifest.get("intensity_scale") == intensity_scale)
    )


def assert_integral(values: np.ndarray, field: str, path: Path) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: {field} contains NaN or Inf")
    rounded = np.rint(values)
    if not np.array_equal(values, rounded):
        raise ValueError(f"{path}: {field} must contain integer-valued labels")
    return rounded.astype(np.int64, copy=False)


def map_semantic_labels(tree_ids: np.ndarray) -> np.ndarray:
    """Map raw tree IDs to binary semantic labels: 0=non-tree, 1=tree."""
    return np.where(
        tree_ids == NON_TREE_LABEL, NON_TREE_LABEL, TREE_LABEL
    ).astype(np.int16, copy=False)


def write_scene(source_path: Path, target_dir: Path, args: argparse.Namespace) -> dict:
    semantic_mode = getattr(args, "semantic_mode", "binary")
    species20 = semantic_mode == "species20"
    vertex = open_vertex_data(source_path, semantic_mode)
    point_count = len(vertex)
    if point_count == 0:
        raise ValueError(f"{source_path} has no points")

    temp_dir = target_dir.parent / f".{target_dir.name}.tmp-{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=False)

    try:
        coord = np.lib.format.open_memmap(
            temp_dir / "coord.npy", mode="w+", dtype=np.float32, shape=(point_count, 3)
        )
        strength = np.lib.format.open_memmap(
            temp_dir / "strength.npy", mode="w+", dtype=np.float32, shape=(point_count, 1)
        )
        segment = np.lib.format.open_memmap(
            temp_dir / "segment.npy", mode="w+", dtype=np.int16, shape=(point_count,)
        )
        instance = np.lib.format.open_memmap(
            temp_dir / "instance.npy", mode="w+", dtype=np.int32, shape=(point_count,)
        )

        raw_tree_ids = set()
        semantic_point_counts = np.zeros(20 if species20 else NUM_SEMANTIC_CLASSES, dtype=np.int64)
        ignored_semantic_points = 0
        raw_species_counts = {}
        non_tree_points = 0
        instanced_points = 0
        tree_instances = set()
        intensity_min = float("inf")
        intensity_max = float("-inf")

        for start in range(0, point_count, args.chunk_size):
            end = min(start + args.chunk_size, point_count)
            chunk = vertex[start:end]
            xyz = np.column_stack((chunk["x"], chunk["y"], chunk["z"]))
            intensity = np.asarray(chunk["intensity"])
            tree = assert_integral(np.asarray(chunk["tree"]), "tree", source_path)

            if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(intensity)):
                raise ValueError(f"{source_path}: coordinate or intensity contains NaN or Inf")
            if np.any(intensity < 0):
                raise ValueError(f"{source_path}: intensity must be non-negative")
            tree_mask = tree != 0
            if np.any(tree[tree_mask] > np.iinfo(np.int32).max):
                raise ValueError(f"{source_path}: tree ID does not fit in int32")
            if np.any(tree < 0):
                raise ValueError(
                    f"{source_path}: tree IDs must be non-negative (0 denotes non-tree)"
                )
            if species20:
                species = assert_integral(np.asarray(chunk["label"]), "label", source_path)
                segment_chunk = map_species_labels(tree, species)
                species_ids, counts = np.unique(species, return_counts=True)
                for species_id, count in zip(species_ids, counts):
                    key = str(int(species_id))
                    raw_species_counts[key] = raw_species_counts.get(key, 0) + int(count)
            else:
                segment_chunk = map_semantic_labels(tree)

            coord[start:end] = xyz.astype(np.float32, copy=False)
            strength[start:end, 0] = (intensity / args.intensity_scale).astype(
                np.float32, copy=False
            )
            segment[start:end] = segment_chunk
            instance_chunk = np.full(end - start, -1, dtype=np.int32)
            instance_chunk[tree_mask] = tree[tree_mask].astype(
                np.int32, copy=False
            )
            instance[start:end] = instance_chunk

            raw_tree_ids.update(np.unique(tree).tolist())
            semantic_point_counts += np.bincount(
                segment_chunk[segment_chunk >= 0], minlength=len(semantic_point_counts)
            )
            ignored_semantic_points += int(np.count_nonzero(segment_chunk == -1))
            non_tree_points += int((~tree_mask).sum())
            instanced_points += int(tree_mask.sum())
            tree_instances.update(np.unique(tree[tree_mask]).tolist())
            intensity_min = min(intensity_min, float(np.min(intensity)))
            intensity_max = max(intensity_max, float(np.max(intensity)))

        for array in (coord, strength, segment, instance):
            array.flush()
        del coord, strength, segment, instance

        manifest = dict(
            format_version=OUTPUT_FORMAT_VERSION,
            semantic_mode=semantic_mode,
            num_classes=len(semantic_point_counts),
            source_path=str(source_path.resolve()),
            point_count=point_count,
            raw_fields=list(REQUIRED_FIELDS) + (["label"] if species20 else []),
            raw_tree_ids=sorted(raw_tree_ids),
            semantic_mapping=SEMANTIC_MAPPING if species20 else {"tree == 0": "non-tree (0)", "tree != 0": "tree (1)"},
            instance_mapping="raw tree 0 -> -1; non-zero tree IDs -> raw tree ID",
            semantic_point_counts=semantic_point_counts.tolist(),
            ignored_semantic_points=ignored_semantic_points,
            raw_species_counts=raw_species_counts,
            non_tree_points=non_tree_points,
            instanced_points=instanced_points,
            tree_instance_count=len(tree_instances),
            intensity_scale=args.intensity_scale,
            raw_intensity_min=intensity_min,
            raw_intensity_max=intensity_max,
        )
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temp_dir, target_dir)
        return manifest
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def process_scene(split: str, source_path: Path, args: argparse.Namespace) -> str:
    scene_name = source_path.stem
    target_dir = args.output_root / split / scene_name
    semantic_mode = getattr(args, "semantic_mode", "binary")
    vertex = open_vertex_data(source_path, semantic_mode)
    if args.dry_run:
        print(f"DRY-RUN  {split}/{source_path.name}: {len(vertex):,} points -> {target_dir}")
        return "dry-run"

    if target_dir.exists():
        if is_complete_scene(target_dir, semantic_mode, args.intensity_scale) and not args.force:
            print(f"SKIP     {split}/{source_path.name}: completed output exists")
            return "skipped"
        if not args.force:
            raise FileExistsError(
                f"{target_dir} already exists but is incomplete. Inspect it or rerun with --force."
            )
        shutil.rmtree(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"PROCESS  {split}/{source_path.name} -> {target_dir}", flush=True)
    manifest = write_scene(source_path, target_dir, args)
    print(
        "DONE     {}/{}: {:,} points, {:,} non-tree, {:,} instanced, {:,} tree instances".format(
            split,
            source_path.name,
            manifest["point_count"],
            manifest["non_tree_points"],
            manifest["instanced_points"],
            manifest["tree_instance_count"],
        ),
        flush=True,
    )
    return "processed"


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        raise ValueError("--output-root must differ from --source-root to protect raw data")
    if args.semantic_mode == "species20" and output_root == Path(DEFAULT_OUTPUT_ROOT).resolve():
        raise ValueError("Use a separate output root for species20; the binary dataset is retained")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    groups = list(scene_paths(source_root, args.splits, args.scenes))
    destinations = {}
    for source_split, paths in groups:
        output_split = "train" if args.merge_splits_to_train else source_split
        for path in paths:
            key = (output_split, path.stem)
            if key in destinations:
                raise ValueError(f"Duplicate destination scene {key}: {destinations[key]} and {path}")
            destinations[key] = path

    counts = {"processed": 0, "skipped": 0, "dry-run": 0}
    for (split, _), source_path in destinations.items():
        counts[process_scene(split, source_path, args)] += 1
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    if args.semantic_mode == "species20" and not args.dry_run:
        metadata = class_metadata()
        metadata["source_root"] = str(source_root)
        metadata["merge_splits_to_train"] = args.merge_splits_to_train
        metadata["source_splits"] = {split: len(paths) for split, paths in groups}
        metadata["splits"] = {}
        for split in dict.fromkeys(key[0] for key in destinations):
            paths = [path for (target_split, _), path in destinations.items() if target_split == split]
            total = np.zeros(20, dtype=np.int64)
            ignored, points = 0, 0
            for path in paths:
                manifest = json.loads((output_root / split / path.stem / "manifest.json").read_text())
                total += np.array(manifest["semantic_point_counts"], dtype=np.int64)
                ignored += manifest["ignored_semantic_points"]
                points += manifest["point_count"]
            metadata["splits"][split] = dict(
                scenes=len(paths), points=points,
                semantic_point_counts=total.tolist(), ignored_semantic_points=ignored,
            )
        (output_root / "semantic_classes.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
