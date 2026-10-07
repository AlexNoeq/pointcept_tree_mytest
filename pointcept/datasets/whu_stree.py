"""WHU-Stree dataset registration.

The preprocessing script stores every scene as ``<root>/<split>/<scene>/`` with
Pointcept's standard ``coord``, ``strength``, ``segment``, and ``instance``
assets. Binary preprocessing uses ``segment`` 0=non-tree and 1=tree.
The optional species20 preprocessing uses 0=non-tree, 1..18=major species,
19=Others, and -1=unknown tree species; its root semantic_classes.json records
the complete mapping.
"""

import os

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class WHUStreeDataset(DefaultDataset):
    """Pointcept dataset wrapper for the preprocessed WHU-Stree scenes."""

    def get_data_name(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        split = os.path.basename(os.path.dirname(data_path))
        scene = os.path.basename(data_path)
        return f"{split}-{scene}"
