"""Treeins transforms matching treeins_rad8.yaml on Pointcept NPY assets."""

import numpy as np
from .transform import TRANSFORMS, index_operator


@TRANSFORMS.register_module()
class TreeinsBinaryLabels:
    """Input uses the preprocessor's 0=ground, 1=wood, 2=leaf IDs."""

    def __call__(self, data):
        labels = data["segment"]
        if not np.isin(labels, [-1, 0, 1, 2]).all():
            raise ValueError(
                "Expected Pointcept semantic IDs -1/0/1/2, not raw PLY IDs"
            )
        data["segment"] = np.where(labels < 0, -1, (labels > 0).astype(np.int64))
        data["instance"] = data["instance"].copy()
        data["instance"][(data["segment"] != 1) | (data["instance"] <= 0)] = -1
        return data


@TRANSFORMS.register_module()
class TreeinsFeatures:
    """Compute relative XYZ + absolute Z before mean-centering coordinates."""

    def __call__(self, data):
        coord = data["coord"]
        relative = coord - coord.mean(axis=0)
        data["treeins_feat"] = np.concatenate([relative, coord[:, 2:3]], axis=1).astype(
            np.float32
        )
        data["coord"] = relative
        keys = list(
            data.get(
                "index_valid_keys",
                ["coord", "color", "normal", "segment", "instance", "strength"],
            )
        )
        if "treeins_feat" not in keys:
            keys.append("treeins_feat")
        data["index_valid_keys"] = keys
        return data


@TRANSFORMS.register_module()
class TreeinsGridSample:
    """Deterministically retain the last input point in each voxel."""

    def __init__(self, grid_size=0.2, return_inverse=False):
        self.grid_size = grid_size
        self.return_inverse = return_inverse

    def __call__(self, data):
        grid = np.floor(data["coord"] / self.grid_size).astype(np.int64)
        grid -= grid.min(0)
        _, inverse = np.unique(grid, axis=0, return_inverse=True)
        last = np.zeros(inverse.max() + 1, dtype=np.int64)
        np.maximum.at(last, inverse, np.arange(len(grid)))
        data = index_operator(data, last)
        data["grid_coord"] = grid[last]
        if "grid_coord" not in data["index_valid_keys"]:
            data["index_valid_keys"] = list(data["index_valid_keys"]) + ["grid_coord"]
        if self.return_inverse:
            data["inverse"] = inverse
        return data


@TRANSFORMS.register_module()
class TreeinsAugment:
    def __call__(self, data):
        coord = data["coord"].copy()
        coord += np.random.normal(0, 0.01, coord.shape)
        angle = np.random.uniform(-np.pi, np.pi)
        c, s = np.cos(angle), np.sin(angle)
        coord = coord @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=coord.dtype)
        coord *= np.random.uniform(0.9, 1.1, 3)
        if np.random.rand() < 0.5:
            coord[:, 0] *= -1
        data["coord"] = coord.astype(np.float32)
        return data
