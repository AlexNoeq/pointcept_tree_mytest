"""WHU-STree's official 18 species plus Others, with non-tree background.

Source: https://github.com/WHU-USI3DV/WHU-STree
Table: https://raw.githubusercontent.com/WHU-USI3DV/WHU-STree/main/media/species.png
"""

import numpy as np


SPECIES_NAMES = (
    "Platanus × acerifolia",
    "Cinnamomum camphora",
    "Lagerstroemia indica",
    "Osmanthus fragrans",
    "Koelreuteria paniculata",
    "Prunus cerasifera 'atropurpurea'",
    "Zelkova serrata",
    "Cedrus deodara",
    "Ginkgo biloba",
    "Acer palmatum var. atropurpureum",
    "Prunus serrulata",
    "Ligustrum lucidum",
    "Malus × micromalus",
    "Magnolia grandiflora",
    "Metasequoia glyptostroboides",
    "Sophora japonica",
    "Acer pictum subsp. mono",
    "Photinia × fraseri",
)
CLASS_NAMES = ("non-tree", *SPECIES_NAMES, "Others")
SEMANTIC_MAPPING = {
    "tree == 0": "non-tree (0)",
    "tree > 0 and label in [0, 17]": "label + 1 (1..18)",
    "tree > 0 and label >= 18": "Others (19)",
    "tree > 0 and label == -1": "unlabeled species, ignore (-1)",
}


def map_species_labels(tree_ids, species_labels):
    """Retain unknown tree species as ignored, rather than background."""
    tree_ids = np.asarray(tree_ids)
    species_labels = np.asarray(species_labels)
    if tree_ids.shape != species_labels.shape:
        raise ValueError("tree and label shapes differ")
    if np.any(tree_ids < 0) or np.any(species_labels < -1):
        raise ValueError("tree must be nonnegative and label must be >= -1")
    if np.any((tree_ids == 0) & (species_labels >= 0)):
        raise ValueError("non-tree points unexpectedly have species annotations")
    result = np.full(tree_ids.shape, -1, dtype=np.int16)
    result[tree_ids == 0] = 0
    known = (tree_ids > 0) & (species_labels >= 0)
    result[known] = np.minimum(species_labels[known], 18) + 1
    return result


def class_metadata():
    return {
        "num_classes": len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
        "ignore_index": -1,
        "tree_class_indices": list(range(1, 20)),
        "mapping": SEMANTIC_MAPPING,
        "official_source": "https://github.com/WHU-USI3DV/WHU-STree",
        "official_species_table": "https://raw.githubusercontent.com/WHU-USI3DV/WHU-STree/main/media/species.png",
        "classes": [
            {
                "id": index,
                "name": name,
                "raw_label": index - 1 if 1 <= index <= 18 else None,
            }
            for index, name in enumerate(CLASS_NAMES)
        ],
    }
