"""Check single-tree labels, background exclusion, and raw-point mapping."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocess = load_module("tree_cls_preprocess", "pointcept/datasets/preprocessing/whu_stree/preprocess_tree_classification.py")


def test_species_majority_unknown_and_background():
    assert preprocess.tree_category(np.array([1, 1, 1, 1, 2, -1, 0])) == (0, 0.8)
    assert preprocess.tree_category(np.array([19, 19, -1])) == (18, 1.0)
    assert preprocess.tree_category(np.array([-1, -1])) == (None, 0.0)
    assert preprocess.tree_category(np.array([1, 2]))[0] is None


def test_all_source_splits_are_training_and_shared_ids_retained(tmp_path, monkeypatch):
    root = tmp_path / "source"
    output = tmp_path / "output"
    for name, original_split in (("a", "train"), ("b", "val")):
        scene = root / "train" / name
        scene.mkdir(parents=True)
        (scene / "manifest.json").write_text(json.dumps(dict(
            semantic_mode="species20", source_path=f"/raw/{original_split}/{name}.ply", raw_tree_ids=[0, 7],
        )))
        np.save(scene / "coord.npy", np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32))
        np.save(scene / "strength.npy", np.ones((3, 1), dtype=np.float32))
        np.save(scene / "instance.npy", np.array([-1, 7, 7]))
        np.save(scene / "segment.npy", np.array([0, 1, 1]))
    monkeypatch.setattr("sys.argv", ["preprocess", "train", "--source-root", str(root),
                                    "--output-root", str(output), "--min-points", "2"])
    preprocess.main()
    assert {p.name for p in (output / "train").iterdir()} == {"a__tree_7", "b__tree_7"}
    assert not (output / "val").exists()
    summary = json.loads((output / "train_summary.json").read_text())
    assert set(summary["splits"]) == {"train"}
    assert summary["splits"]["train"]["trees"] == 2


def args(mode, **kwargs):
    return Namespace(mode=mode, max_points=100, min_points=2, purity=0.8, seed=0, **kwargs)


def test_training_excludes_non_tree_unknown_and_shared_ids(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()
    np.save(scene / "coord.npy", np.column_stack((np.arange(9), np.zeros((9, 2)))).astype(np.float32))
    np.save(scene / "strength.npy", np.ones((9, 1), dtype=np.float32))
    np.save(scene / "instance.npy", np.array([-1, 7, 7, 7, 8, 8, 9, 9, 9]))
    np.save(scene / "segment.npy", np.array([0, 1, 1, 0, -1, -1, 19, 19, 19]))
    output = tmp_path / "out"
    record = preprocess.extract_scene(scene, "train", output, args("train"), excluded_ids={9})
    assert len(record["trees"]) == 1
    row = record["trees"][0]
    assert row["instance_id"] == 7 and row["category"] == 0 and row["point_count"] == 2
    assert record["skipped_shared_trees"] == 1
    assert record["skipped_unlabeled_or_mixed_trees"] == 1
    assert np.load(output / "train" / row["sample"] / "category.npy").tolist() == [0]
    assert preprocess.extract_scene(scene, "train", output, args("train"), excluded_ids={9}) == record


def test_test_extraction_uses_semantic_mask_not_largest_instance(tmp_path):
    source = tmp_path / "pred" / "scene"
    source.mkdir(parents=True)
    np.save(source / "semantic.npy", np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.uint8))
    np.save(source / "instance.npy", np.array([99, 99, 99, 7, 7, 8, 8], dtype=np.int32))
    vertex = np.zeros(7, dtype=[(k, "f4") for k in ("x", "y", "z", "intensity")])
    vertex["x"] = np.arange(7)
    vertex["intensity"] = 32767.5
    PlyData([PlyElement.describe(vertex, "vertex")]).write(tmp_path / "scene.ply")
    record = preprocess.extract_scene(source, "test", tmp_path / "out", args("test", input_dir=tmp_path))
    assert [r["instance_id"] for r in record["trees"]] == [7, 8]
    assert record["excluded_non_tree_points"] == 3
    for row in record["trees"]:
        assert row["category"] == -1
        np.testing.assert_allclose(np.load(tmp_path / "out" / "test" / row["sample"] / "strength.npy"), 0.5)


def test_dual_predictions_preserve_background(tmp_path):
    module = load_module("dual_inference", "tools/infer_whu_semantic_instances.py")
    xyz = np.column_stack((np.arange(5), np.zeros((5, 2)))).astype(np.float32)
    semantic = np.array([0, 0, 1, 1, 1], dtype=np.uint8)
    instance = np.array([10, 10, 4, 0, 0], dtype=np.int32)
    count, filled, missing = module.finish_instances(xyz, instance, semantic, 2)
    assert instance.tolist() == [0, 0, 1, 1, 1]
    assert (count, filled, missing) == (1, 2, 0)
    np.save(tmp_path / "semantic.npy", semantic)
    np.save(tmp_path / "instance.npy", instance)
    report = module.validate_scene(tmp_path, 5, 2)
    assert report["non_tree_points"] == 2 and report["tree_points"] == 3


def test_species_mapping_preserves_background_and_unassigned():
    module = load_module("species_inference", "tools/infer_whu_tree_species.py")
    semantic = np.array([0, 0, 1, 1, 1, 1])
    instance = np.array([99, 7, 7, 8, 0, 99])
    trees = {7: {"semantic_class": 0}, 8: {"semantic_class": 18}}
    assert module.species_for_points(semantic, instance, trees).tolist() == [-1, -1, 0, 18, -1, -1]


def test_two_column_output_preserves_instance_ids_and_background(tmp_path):
    module = load_module("species_columns", "tools/infer_whu_tree_species.py")
    semantic = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
    instance = np.array([0, 8, 3, 0, 8], dtype=np.int32)
    trees = {3: {"semantic_class": 18}, 8: {"semantic_class": 0}}
    module.save_scene_result(tmp_path, "scene", semantic, instance, trees, tmp_path / "model.pth", tmp_path)
    result = np.load(tmp_path / "scene.npy")
    assert result.dtype == np.int32 and result.shape == (5, 2)
    assert result[:, 0].tolist() == instance.tolist()
    assert result[:, 1].tolist() == [-1, 0, 18, -1, 0]


def test_direct_ply_classification_and_two_column_mapping(tmp_path):
    import torch
    from types import SimpleNamespace

    module = load_module("species_raw_pipeline", "tools/infer_whu_tree_species.py")
    vertex = np.zeros(7, dtype=[(k, "f4") for k in ("x", "y", "z", "intensity")])
    vertex["x"] = np.arange(7)
    vertex["intensity"] = 32767.5
    path = tmp_path / "raw.ply"
    PlyData([PlyElement.describe(vertex, "vertex")]).write(path)
    semantic = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    instance = np.array([99, 99, 99, 7, 7, 8, 8], dtype=np.int32)
    cfg = SimpleNamespace(seed=0, enable_amp=False, class_names=[str(i) for i in range(19)],
                          data=SimpleNamespace(test=SimpleNamespace(num_points=8192, transform=[
                              dict(type="NormalizeCoord"),
                              dict(type="GridSample", grid_size=0.05, mode="train", return_grid_coord=True),
                              dict(type="ToTensor"),
                              dict(type="Collect", keys=("coord", "grid_coord", "name", "category"),
                                   feat_keys=("coord", "strength")),
                          ])))

    class FakeClassifier:
        def __call__(self, batch):
            assert batch["feat"].shape[1] == 4
            assert "category" not in batch
            logits = torch.zeros((len(batch["offset"]), 19))
            logits[0, 0] = 10
            logits[1, 18] = 10
            return dict(cls_logits=logits)

    predictions, skipped = module.classify_raw_scene(path, semantic, instance, FakeClassifier(), cfg,
                                                     SimpleNamespace(device="cpu", max_saved_points=65536, batch_size=2))
    assert set(predictions) == {7, 8} and skipped == []
    assert predictions[7]["semantic_class"] == 0
    assert predictions[8]["semantic_class"] == 18
    module.save_scene_result(tmp_path, "raw", semantic, instance, predictions, tmp_path / "model.pth", tmp_path)
    result = np.load(tmp_path / "raw.npy")
    np.testing.assert_array_equal(result[:, 0], instance)
    assert result[:, 1].tolist() == [-1, -1, -1, 0, 0, 18, 18]
