"""PTV3 tree-species classifier following the ModelNet40 PTV3 baseline.

Categories 0..18 correspond to semantic classes 1..19 (no non-tree category).
One sample is one ground-truth tree instance with XYZ + normalized intensity.
All labeled source train/val scenes are used for training, without evaluation.
"""

_base_ = ["../modelnet40/cls-ptv3-v1m1-0-base.py"]

custom_imports = dict(imports=["pointcept.datasets.whu_stree_tree"], allow_failed_imports=False)
save_path = "exp/whu_stree/cls-ptv3-tree-species"
seed = 0
batch_size = 32
batch_size_val = 8
batch_size_test = 8
num_worker = 16
mix_prob = 0.0
enable_amp = False  # Match the ModelNet40 classifier's stable spconv execution.
clip_grad = 1.0
evaluate = False
evaluate_test = False

num_classes = 19
class_names = [
    "Platanus × acerifolia", "Cinnamomum camphora", "Lagerstroemia indica",
    "Osmanthus fragrans", "Koelreuteria paniculata", "Prunus cerasifera 'atropurpurea'",
    "Zelkova serrata", "Cedrus deodara", "Ginkgo biloba",
    "Acer palmatum var. atropurpureum", "Prunus serrulata", "Ligustrum lucidum",
    "Malus × micromalus", "Magnolia grandiflora", "Metasequoia glyptostroboides",
    "Sophora japonica", "Acer pictum subsp. mono", "Photinia × fraseri", "Others",
]

model = dict(
    num_classes=num_classes,
    backbone=dict(in_channels=4, enc_mode=True),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

epoch = 500
eval_epoch = 100  # One logged epoch is one pass through the training trees.
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR", max_lr=[0.001, 0.0001], pct_start=0.05,
    anneal_strategy="cos", div_factor=10.0, final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

dataset_type = "WHUStreeTreeDataset"
data_root = __import__("os").environ.get("WHU_TREECLS_ROOT", "data/whu_stree_treecls")
num_points = 81920
grid_size = 0.05  # Coordinates are normalized to the unit sphere first.

train_transform = [
    dict(type="NormalizeCoord"),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.7, 1.5], anisotropic=True),
    dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="ShufflePoint"),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "category", "name"), feat_keys=("coord", "strength")),
]
eval_transform = [
    dict(type="NormalizeCoord"),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="ToTensor"),
    dict(type="Collect", keys=("coord", "grid_coord", "category", "name"), feat_keys=("coord", "strength")),
]

data = dict(
    _delete_=True,
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(type=dataset_type, split="train", data_root=data_root, num_points=num_points,
               random_sampling=True, seed=seed, transform=train_transform, test_mode=False, loop=1),
    test=dict(type=dataset_type, split="test", data_root=data_root, num_points=num_points,
              random_sampling=False, seed=seed, transform=eval_transform, test_mode=False),
)

hooks = [
    dict(type="CheckpointLoader"), dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2), dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=25),
]
test = dict(_delete_=True, type="ClsTester", verbose=True)
