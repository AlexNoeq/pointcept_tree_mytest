"""PointGroup baseline for binary WHU-Stree tree instance segmentation.

Use the paired ``preprocess_whu_stree.py`` script first.  The default network
uses xyz only. Set ``WHU_STREE_USE_INTENSITY=1`` to add normalized intensity as
the fourth input channel while keeping every other setting unchanged.
"""

_base_ = ["../_base_/default_runtime.py"]

# The raw scenes are much larger than indoor S3DIS rooms. These conservative
# defaults are for one GPU; scale batch_size with available GPU memory.
batch_size = 1
batch_size_val = 1
batch_size_test = 1
num_worker = 4
mix_prob = 0.0
empty_cache = True
enable_amp = True
evaluate = True

# Keep intensity available in preprocessing but optional in the model. This is
# intentionally environment-controlled so xyz and xyz+intensity use identical
# data, augmentations, and PointGroup settings.
use_intensity = __import__("os").environ.get("WHU_STREE_USE_INTENSITY", "0").lower() in (
    "1",
    "true",
    "yes",
)
feature_keys = ("coord", "strength") if use_intensity else ("coord",)
in_channels = 4 if use_intensity else 3

class_names = ["non-tree", "tree"]
num_classes = 2
# The raw tree ID determines the semantic class: 0 is non-tree; every non-zero
# value is tree.  Non-tree remains in the semantic loss but is excluded from
# instance targets, clustering, and instance mAP.
semantic_ignore_index = -1
segment_ignore_index = (0,)
grid_size = 0.2

model = dict(
    type="PG-v1m1",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=in_channels,
        num_classes=0,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    backbone_out_channels=96,
    semantic_num_classes=num_classes,
    semantic_ignore_index=semantic_ignore_index,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    cluster_thresh=1.5,
    cluster_closed_points=300,
    cluster_propose_points=100,
    cluster_min_points=100,
    voxel_size=grid_size,
)

epoch = 1000
eval_epoch = 200
optimizer = dict(type="SGD", lr=0.05, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type="PolyLR")

dataset_type = "WHUStreeDataset"
data_root = __import__("os").environ.get(
    "WHU_STREE_ROOT", "/media/ubuntu/新加卷/FGL/WHU-Stree-pointcept"
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
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            # dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            # Keep most of each outdoor scene to reduce all-ignore crops.
            dict(type="SphereCrop", sample_rate=1, mode="random"),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "instance", "instance_centroid", "bbox"),
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
            dict(type="Copy", keys_dict={"coord": "origin_coord", "segment": "origin_segment", "instance": "origin_instance"}),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "instance", "origin_coord", "origin_segment", "origin_instance", "instance_centroid", "bbox", "name"),
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
            dict(type="Copy", keys_dict={"coord": "origin_coord", "segment": "origin_segment", "instance": "origin_instance"}),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="InstanceParser", segment_ignore_index=segment_ignore_index, instance_ignore_index=-1),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "instance", "origin_coord", "origin_segment", "origin_instance", "instance_centroid", "bbox", "name"),
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
