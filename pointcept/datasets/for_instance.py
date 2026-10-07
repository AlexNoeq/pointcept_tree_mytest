"""FOR-instance dataset and cylindrical cropping for SegmentAnyTreeV2."""

from __future__ import annotations

import os

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import TRANSFORMS, index_operator


@DATASETS.register_module()
class FORInstanceDataset(DefaultDataset):
    """Pointcept layout produced by ``preprocess_for_instance.py``."""

    def get_data_name(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        split = os.path.basename(os.path.dirname(data_path))
        scene = os.path.basename(data_path)
        return f"{split}-{scene}"


@TRANSFORMS.register_module()
class CylinderCrop:
    """Crop an unbounded-height XY cylinder and optionally cap raw points.

    ``mode='random'`` follows the training protocol and chooses a source point
    as the cylinder centre. ``mode='center'`` uses the XY bounding-box centre
    and makes validation deterministic.
    """

    def __init__(
        self,
        radius: float = 16.0,
        point_max: int | None = 650_000,
        mode: str = "random",
    ):
        if radius <= 0:
            raise ValueError("radius must be positive")
        if point_max is not None and point_max <= 0:
            raise ValueError("point_max must be positive or None")
        if mode not in {"random", "center"}:
            raise ValueError("mode must be 'random' or 'center'")
        self.radius = float(radius)
        self.point_max = point_max
        self.mode = mode

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        if len(coord) == 0:
            return {
                "_skip_sample": True,
                "_skip_reason": "empty_input_before_cylinder_crop",
                "_skip_name": data_dict.get("name"),
                "_skip_point_count": 0,
            }
        if self.mode == "random":
            center = coord[np.random.randint(len(coord)), :2]
        else:
            center = (coord[:, :2].min(axis=0) + coord[:, :2].max(axis=0)) / 2
        distance_squared = ((coord[:, :2] - center) ** 2).sum(axis=1)
        indices = np.flatnonzero(distance_squared <= self.radius**2)
        if len(indices) == 0:
            # Defensive fallback for non-finite/custom coordinate containers.
            indices = np.array([distance_squared.argmin()])
        if self.point_max is not None and len(indices) > self.point_max:
            rng = np.random if self.mode == "random" else np.random.default_rng(0)
            indices = rng.choice(indices, self.point_max, replace=False)
            indices.sort()
        return index_operator(data_dict, indices)


@TRANSFORMS.register_module()
class TreeInstanceParser:
    """Compact positive tree IDs while preserving unlabelled tree voxels.

    Pointcept's generic ``InstanceParser`` treats every ID on a thing-class
    point as an instance, including non-positive IDs.  SaTv2 instead excludes
    seeds and embedding voxels whose source instance ID is not positive.
    """

    def __init__(self, segment_ignore_index=(0,), instance_ignore_index=-1):
        self.segment_ignore_index = tuple(segment_ignore_index)
        self.instance_ignore_index = int(instance_ignore_index)

    def __call__(self, data_dict):
        segment = np.asarray(data_dict["segment"])
        instance = np.asarray(data_dict["instance"])
        valid = (~np.isin(segment, self.segment_ignore_index)) & (instance > 0)
        parsed = np.full(instance.shape, self.instance_ignore_index, dtype=np.int64)
        if valid.any():
            _, inverse = np.unique(instance[valid], return_inverse=True)
            parsed[valid] = inverse
        data_dict["instance"] = parsed
        return data_dict
