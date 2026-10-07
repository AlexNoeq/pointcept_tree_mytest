"""Standalone SaTv2 SpUNet-v2m1 local aggregation + Spatial Mamba config."""

# Runtime defaults (kept here so this config does not inherit another file).
weight = None
resume = False
evaluate = True
evaluate_test = True
test_only = False
seed = None
save_path = "exp/for_instance/satv2_mamba_aggregation"
num_worker = 4
batch_size = 1
gradient_accumulation_steps = 12
batch_size_val = 1
batch_size_test = 1
epoch = 3000
eval_epoch = 100
clip_grad = 1.0
sync_bn = False
enable_amp = True
amp_dtype = "float16"
empty_cache = False
empty_cache_per_epoch = True
find_unused_parameters = True
enable_wandb = False
wandb_project = "pointcept"
wandb_key = None
mix_prob = 0.0
param_dicts = [dict(keyword="block", lr=0.0003)]

custom_imports = dict(
    imports=["pointcept.models.myself", "pointcept.engines.hooks.satv2_evaluator"],
    allow_failed_imports=False,
)

num_classes = 3
class_names = ["ground", "wood", "leaf"]
semantic_ignore_index = -1
tree_class_indices = (1, 2)
instance_ignore_index = -1
grid_size = 0.2
cylinder_radius = 16

model = dict(
    type="SaTv2-myself-v1m1",
    backbone=dict(
        type="SpUNet-v2m1",
        in_channels=3,
        num_channels=32,
        num_planes=(32, 64, 96, 128, 256),
        return_blocks=True,
    ),
    backbone_out_channels=32,
    semantic_num_classes=num_classes,
    semantic_ignore_index=semantic_ignore_index,
    tree_class_indices=tree_class_indices,
    embedding_dim=5,
    warmup_epochs=10,
    decoder=dict(
        type="mamba",
        d_model=128,
        num_queries=400,
        num_layers=6,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        cross_mode="aggregation",
        cross_k=32,
        mamba_d_state=64,
        mamba_d_conv=4,
        mamba_expand=2,
        slab_thickness=2.0,
        use_relative_pos=True,
    ),
    criteria=dict(
        class_weights=(1.0, 2.0, 1.0),
        semantic_ce_weight=0.5,
        semantic_lovasz_weight=0.5,
        dice_weight=2.0,
        focal_weight=2.0,
        score_weight=0.5,
        decoder_loss_weight=1.0,
        discriminative_weight=1.0,
        auxiliary_weight=0.5,
        focal_alpha=0.25,
        focal_gamma=2.0,
        delta_var=0.5,
        delta_dist=2.5,
        discriminative_reg_weight=1e-3,
        discriminative_clip=10.0,
    ),
    score_threshold=0.15,
    mask_threshold=0.5,
    min_points=0,
)

optimizer = dict(type="AdamW", lr=0.003, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.003, 0.0003],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

dataset_type = "FORInstanceDataset"
data_root = "/home/wyj/225/fgl/data/forinstance_pointcept"

train_transform = [
    dict(type="CylinderCrop", radius=cylinder_radius, point_max=480000, mode="random"),
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", center=[0, 0, 0], p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
    dict(type="TreeInstanceParser", segment_ignore_index=(0,), instance_ignore_index=instance_ignore_index),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "instance"),
        feat_keys=("coord",),
    ),
]

eval_transform = [
    dict(type="CylinderCrop", radius=cylinder_radius, point_max=480000, mode="center"),
    dict(type="CenterShift", apply_z=True),
    dict(type="Copy", keys_dict={"coord": "origin_coord", "segment": "origin_segment", "instance": "origin_instance"}),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_inverse=True, return_grid_coord=True),
    dict(type="TreeInstanceParser", segment_ignore_index=(0,), instance_ignore_index=instance_ignore_index),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "segment", "instance", "origin_coord", "origin_segment", "origin_instance", "inverse", "name"),
        feat_keys=("coord",),
        offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
    ),
]

data = dict(
    num_classes=num_classes,
    ignore_index=semantic_ignore_index,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=train_transform,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=eval_transform,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=eval_transform,
        test_mode=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SaTV2Evaluator", iou_threshold=0.5, evaluate_test=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

train = dict(type="DefaultTrainer")
test = dict(type="SaTV2Tester", iou_threshold=0.5, verbose=False)
