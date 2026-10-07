"""Object classification dataset: one WHU-STree tree per sample."""

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class WHUStreeTreeDataset(DefaultDataset):
    VALID_ASSETS = ["coord", "strength", "category"]

    def __init__(self, num_points=8192, random_sampling=False, seed=0, **kwargs):
        self.num_points = num_points
        self.random_sampling = random_sampling
        self.seed = seed
        if num_points < 2:
            raise ValueError("num_points must be at least 2")
        if kwargs.get("test_mode", False):
            raise ValueError("Use test_mode=False with ClsTester or the tree inference script")
        super().__init__(**kwargs)
        self.data_list = sorted(p for p in self.data_list if not p.split("/")[-1].startswith("."))
        if not self.data_list:
            raise ValueError(f"No tree samples in {self.data_root}/{self.split}")

    def get_data(self, idx):
        data = super().get_data(idx)
        category = data["category"].reshape(-1).astype(np.int64)
        if category.shape != (1,) or not (0 <= category[0] < 19 or (self.split == "test" and category[0] == -1)):
            raise ValueError(f"Invalid tree category for {data['name']}: {category}")
        data["category"] = category
        count = len(data["coord"])
        if count < 2 or not np.isfinite(data["coord"]).all():
            raise ValueError(f"Invalid coordinates for {data['name']}")
        if np.ptp(data["coord"], axis=0).max() <= 0:
            raise ValueError(f"Degenerate tree for {data['name']}")
        if count > self.num_points:
            rng = np.random if self.random_sampling else np.random.default_rng(self.seed + idx % len(self.data_list))
            indices = rng.choice(count, self.num_points, replace=False)
            for key in ("coord", "strength", "segment", "instance"):
                if key in data:
                    data[key] = data[key][indices]
        return data
