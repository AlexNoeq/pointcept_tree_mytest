"""LitePT PointGroup baseline for WHU-Stree instance segmentation.

This is the WHU-Stree counterpart of
``configs/scannet/insseg-pointgroup-v1m2-0-litept-small.py``.  It uses the
binary preprocessing convention: raw ``tree == 0`` is ``non-tree`` (class 0)
and every non-zero raw tree ID is ``tree`` (class 1).  Only the tree class
participates in instance segmentation.  By default the input is xyz. Set
``WHU_STREE_USE_INTENSITY=1`` to append the normalized intensity channel.
"""

_base_ = ["../_base_/default_runtime.py"]

# WHU-Stree scenes are substantially larger than ScanNet scenes.  These values
# target one GPU and retain the existing WHU-Stree 0.25 m voxelization.
batch_size = 1
batch_size_val = 1
batch_size_test = 1
num_worker = 4
mix_prob = 0.0
empty_cache = True
enable_amp = True
evaluate = True

use_intensity = __import__("os").environ.get("WHU_STREE_USE_INTENSITY", "0").lower() in (
    "1",
    "true",
    "yes",
)
feature_keys = ("coord", "strength") if use_intensity else ("coord",)
in_channels = 4 if use_intensity else 3

class_names = ["non-tree", "tree"]
num_classes = 2
semantic_ignore_index = -1
# Non-tree is learned by the semantic head, but is excluded from instance
# offsets, clustering, and instance mAP; tree is the sole instance class.
segment_ignore_index = (0,)
grid_size = 0.25

model = dict(
    type="PG-v1m2",
    backbone=dict(
        type="LitePT-v1",
        in_channels=in_channels,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(True, True, True, False),
        dec_attn=(False, False, False, True),
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
    backbone_out_channels=72,
    semantic_num_classes=num_classes,
    semantic_ignore_index=semantic_ignore_index,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    cluster_thresh=1.5,
    cluster_closed_points=300,
    cluster_propose_points=100,
    cluster_min_points=100,
    voxel_size=grid_size,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=semantic_ignore_index),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=semantic_ignore_index,
        ),
    ],
)

# LitePT uses the same optimizer/schedule family as the ScanNet LitePT config.
# The maximum LR is linearly scaled from ScanNet's 12-scene global batch to the
# one-scene WHU-Stree batch.
epoch = 1000
eval_epoch = 50
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
    "WHU_STREE_ROOT", "/home/wyj/225/fgl/data/WHU-Stree-pointcept"
)

data = dict(
    num_classes=num_classes,
    ignore_index=semantic_ignore_index,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="SphereCrop", sample_rate=1, mode="random"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "instance_centroid",
                    "bbox",
                    "grid_size",
                ),
                feat_keys=feature_keys,
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "name",
                ),
                feat_keys=feature_keys,
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "name",
                ),
                feat_keys=feature_keys,
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="InsSegEvaluator", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

test = dict(
    type="InsSegTester",
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    verbose=False,
)
