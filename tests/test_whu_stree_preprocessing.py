"""Test label remapping and the actual chunked PLY-to-NPY conversion."""

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement


SCRIPT = Path(__file__).parents[1] / "pointcept/datasets/preprocessing/whu_stree/preprocess_whu_stree.py"
spec = importlib.util.spec_from_file_location("whu_preprocess", SCRIPT)
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)


def test_official_species_mapping_including_unknown_trees():
    trees = np.array([0, 7, 8, 9, 10, 11, 12])
    species = np.array([-1, 0, 1, 17, 18, 60, -1])
    result = preprocess.map_species_labels(trees, species)
    assert result.tolist() == [0, 1, 2, 18, 19, 19, -1]
    assert result.dtype == np.int16
    assert len(preprocess.CLASS_NAMES) == 20


@pytest.mark.parametrize("trees,labels", [([0], [2]), ([1], [-2]), ([-1], [0])])
def test_inconsistent_annotations_are_rejected(trees, labels):
    with pytest.raises(ValueError):
        preprocess.map_species_labels(trees, labels)


def test_chunked_conversion_retains_original_order_and_instances(tmp_path):
    vertex = np.zeros(7, dtype=[(k, "f8") for k in ("x", "y", "z", "intensity", "tree", "label")])
    vertex["x"] = np.arange(7) + 100.25
    vertex["intensity"] = 32767.5
    vertex["tree"] = [0, 7, 8, 9, 10, 11, 12]
    vertex["label"] = [-1, 0, 1, 17, 18, 60, -1]
    source = tmp_path / "scene.ply"
    PlyData([PlyElement.describe(vertex, "vertex")]).write(source)
    args = Namespace(semantic_mode="species20", chunk_size=2, intensity_scale=65535.0)
    target = tmp_path / "processed"
    manifest = preprocess.write_scene(source, target, args)
    assert np.load(target / "segment.npy").tolist() == [0, 1, 2, 18, 19, 19, -1]
    assert np.load(target / "instance.npy").tolist() == [-1, 7, 8, 9, 10, 11, 12]
    np.testing.assert_array_equal(np.load(target / "coord.npy")[:, 0], vertex["x"])
    np.testing.assert_array_equal(np.load(target / "strength.npy"), np.full((7, 1), 0.5))
    assert sum(manifest["semantic_point_counts"]) + manifest["ignored_semantic_points"] == 7
    assert manifest["semantic_point_counts"][19] == 2
    assert preprocess.is_complete_scene(target, "species20", 65535.0)
    assert not preprocess.is_complete_scene(target, "binary")
    assert not preprocess.is_complete_scene(target, "species20", 255.0)
    assert json.loads((target / "manifest.json").read_text())["num_classes"] == 20
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_missing_species_field_and_nonintegral_labels_are_rejected(tmp_path):
    vertex = np.zeros(1, dtype=[(k, "f8") for k in preprocess.REQUIRED_FIELDS])
    source = tmp_path / "unlabeled.ply"
    PlyData([PlyElement.describe(vertex, "vertex")]).write(source)
    with pytest.raises(ValueError, match="label"):
        preprocess.open_vertex_data(source, "species20")
    with pytest.raises(ValueError, match="integer-valued"):
        preprocess.assert_integral(np.array([1.5]), "label", source)


def test_binary_mapping_is_retained():
    assert preprocess.map_semantic_labels(np.array([0, 1, 17, 600])).tolist() == [0, 1, 1, 1]


def test_merge_all_source_splits_into_train(tmp_path, monkeypatch):
    source = tmp_path / "raw"
    target = tmp_path / "processed"
    vertex = np.zeros(2, dtype=[(k, "f8") for k in (*preprocess.REQUIRED_FIELDS, "label")])
    vertex["tree"] = [0, 7]
    vertex["label"] = [-1, 17]
    for split, scene in [("train", "a"), ("val", "b")]:
        (source / split).mkdir(parents=True)
        PlyData([PlyElement.describe(vertex, "vertex")]).write(source / split / (scene + ".ply"))
    argv = [str(SCRIPT), "--source-root", str(source), "--output-root", str(target),
            "--semantic-mode", "species20", "--merge-splits-to-train"]
    monkeypatch.setattr(sys, "argv", argv)
    assert preprocess.main() == 0
    assert {p.name for p in (target / "train").iterdir()} == {"a", "b"}
    assert not (target / "val").exists()
    metadata = json.loads((target / "semantic_classes.json").read_text())
    assert list(metadata["splits"]) == ["train"]
    assert metadata["splits"]["train"]["scenes"] == 2
    assert metadata["source_splits"] == {"train": 1, "val": 1}
    # A repeated invocation reuses the completed scenes.
    assert preprocess.main() == 0
    (source / "val" / "b.ply").rename(source / "val" / "a.ply")
    with pytest.raises(ValueError, match="Duplicate destination"):
        preprocess.main()
