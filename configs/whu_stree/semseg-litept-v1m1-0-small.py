"""LitePT-S semantic segmentation on WHU-STree's 20-class training set.

All 206 scenes are in train/. There is no validation or end-of-training test.
Features are XYZ plus normalized intensity; labels -1 are ignored.
"""

_base_ = ["../_base_/default_runtime.py"]

save_path = "exp/whu_stree/semseg-litept-v1m1-0-small"
seed = 0
batch_size = 1
num_worker = 4
mix_prob = 0.0
empty_cache = True
enable_amp = True
clip_grad = 1.0
evaluate = False
evaluate_test = False

num_classes = 20
semantic_ignore_index = -1
class_names = [
    "non-tree",
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
    "Others",
]

# Architecture follows configs/scannet/semseg-litept-v1m1-0-small.py.
model = dict(
    type="DefaultSegmentorV2",
    num_classes=num_classes,
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT-v1",
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=semantic_ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=semantic_ignore_index),
    ],
)

# Equal values keep data.train.loop=1: each logged epoch is one dataset pass.
epoch = 1000
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0005, 0.00005],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.00005)]

dataset_type = "WHUStreeDataset"
data_root = __import__("os").environ.get(
    "WHU_STREE_ROOT", "/home/wyj/225/fgl/data/WHU-Stree-pointcept-20class"
)
grid_size = 0.12
cylinder_radius = 16.0
point_max = 1000000

data = dict(
    num_classes=num_classes,
    ignore_index=semantic_ignore_index,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        loop=1,
        transform=[
            dict(type="CylinderCrop", radius=cylinder_radius, point_max=640000, mode="random"),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="SphereCrop", point_max=point_max, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "grid_size"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
    ),
)

# No evaluator hooks: checkpoints are saved without a validation metric.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=50),
]
