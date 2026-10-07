"""SegmentAnyTreeV2 reproduction on FOR-instance v3.

The model and optimization values follow Table A.5 and Sections 3.2--3.8.
Set FOR_INSTANCE_ROOT to the output of preprocess_for_instance.py.
"""

_base_ = ["../_base_/default_runtime.py"]

save_path = "exp/for_instance/insseg-satv2-v1m1-0-ptv3"
batch_size = 1
batch_size_val = 1
batch_size_test = 1
num_worker = 4
gradient_accumulation_steps = 24
mix_prob = 0.0
empty_cache = False
empty_cache_per_epoch = True
enable_amp = True
amp_dtype = "float16"
find_unused_parameters = True
evaluate = True
evaluate_test = True  # evaluate both val and test after each training epoch
clip_grad = 1.0

num_classes = 3
class_names = ["ground", "wood", "leaf"]
semantic_ignore_index = -1
tree_class_indices = (1, 2)
instance_ignore_index = -1
grid_size = 0.2
cylinder_radius = 16

model = dict(
    type="SaTv2-v1m1",
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 192),
        enc_num_head=(3, 6, 12, 24, 12),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        # Pointcept orders decoder stages fine-to-coarse; Table A.5 reports
        # them coarse-to-fine as 384, 256, 128, 128.
        dec_channels=(128, 128, 256, 384),
        dec_num_head=(8, 8, 16, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
    ),
    backbone_out_channels=128,
    semantic_num_classes=num_classes,
    semantic_ignore_index=semantic_ignore_index,
    tree_class_indices=tree_class_indices,
    embedding_dim=5,
    warmup_epochs=10,
    decoder=dict(
        dim=128,
        num_queries=400,
        num_layers=6,
        num_heads=4,
        hidden_dim=512,
        dropout=0.0,
    ),
    criteria=dict(
        class_weights=(1.0, 1.0, 0.5),
        semantic_ce_weight=0.5,
        semantic_lovasz_weight=0.5,
        dice_weight=2.0,
        focal_weight=2.0,
        score_weight=0.5,
        # The paper names but does not publish decoder_loss_weight. Keep it
        # explicit for checkpoint/config provenance; 1.0 is the neutral value.
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

# Pointcept represents validation frequency by folding logical data passes into
# dataset.loop. 100 trainer epochs x loop 10 = 1000 paper epochs, with
# validation after every 10 logical epochs.
epoch = 1000
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.003, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.003, 0.0003],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0003)]

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

test = dict(
    type="SaTV2Tester",
    iou_threshold=0.5,
    verbose=False,
)
