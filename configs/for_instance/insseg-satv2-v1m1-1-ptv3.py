weight = None
resume = False
evaluate = True
test_only = False
seed = 42562517
save_path = 'exp/for_instance/satv2_ptv3_eval_test'
num_worker = 4
batch_size = 1
gradient_accumulation_steps = 24
batch_size_val = 1
batch_size_test = 1
epoch = 1000
eval_epoch = 100
clip_grad = 1.0
sync_bn = False
enable_amp = True
amp_dtype = 'float16'
empty_cache = False
empty_cache_per_epoch = True
find_unused_parameters = True
enable_wandb = False
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.0
param_dicts = [dict(keyword='block', lr=0.0003)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SaTV2Evaluator', iou_threshold=0.5, evaluate_test=True),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SaTV2Tester', verbose=False, iou_threshold=0.5)
evaluate_test = True
num_classes = 3
class_names = ['ground', 'wood', 'leaf']
semantic_ignore_index = -1
tree_class_indices = (1, 2)
instance_ignore_index = -1
grid_size = 0.2
cylinder_radius = 8
model = dict(
    type='SaTv2-v1m1',
    backbone=dict(
        type='PT-v3m1',
        in_channels=3,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 192),
        enc_num_head=(3, 6, 12, 24, 12),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
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
        pdnorm_ln=False),
    backbone_out_channels=128,
    semantic_num_classes=3,
    semantic_ignore_index=-1,
    tree_class_indices=(1, 2),
    embedding_dim=5,
    warmup_epochs=10,
    decoder=dict(
        dim=128,
        num_queries=300,
        num_layers=6,
        num_heads=4,
        hidden_dim=512,
        dropout=0.0),
    criteria=dict(
        class_weights=(1.0, 10.0, 2.0),
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
        discriminative_reg_weight=0.001,
        discriminative_clip=10.0),
    score_threshold=0.15,
    mask_threshold=0.5,
    min_points=0)
optimizer = dict(type='AdamW', lr=0.003, weight_decay=0.05)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.003, 0.0003],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'FORInstanceDataset'
data_root = '/home/wyj/225/fgl/data/forinstance_pointcept'
train_transform = [
    dict(type='CylinderCrop', radius=8, point_max=480000, mode='random'),
    dict(type='CenterShift', apply_z=True),
    dict(
        type='RandomRotate', angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
    dict(
        type='RandomRotate',
        angle=[-0.015625, 0.015625],
        axis='x',
        center=[0, 0, 0],
        p=0.5),
    dict(
        type='RandomRotate',
        angle=[-0.015625, 0.015625],
        axis='y',
        center=[0, 0, 0],
        p=0.5),
    dict(type='RandomScale', scale=[0.9, 1.1]),
    dict(type='RandomFlip', p=0.5),
    dict(
        type='GridSample',
        grid_size=0.2,
        hash_type='fnv',
        mode='train',
        return_grid_coord=True),
    dict(
        type='TreeInstanceParser',
        segment_ignore_index=(0, ),
        instance_ignore_index=-1),
    dict(type='ToTensor'),
    dict(
        type='Collect',
        keys=('coord', 'grid_coord', 'segment', 'instance'),
        feat_keys=('coord', ))
]
eval_transform = [
    dict(type='CylinderCrop', radius=8, point_max=480000, mode='center'),
    dict(type='CenterShift', apply_z=True),
    dict(
        type='Copy',
        keys_dict=dict(
            coord='origin_coord',
            segment='origin_segment',
            instance='origin_instance')),
    dict(
        type='GridSample',
        grid_size=0.2,
        hash_type='fnv',
        mode='train',
        return_inverse=True,
        return_grid_coord=True),
    dict(
        type='TreeInstanceParser',
        segment_ignore_index=(0, ),
        instance_ignore_index=-1),
    dict(type='ToTensor'),
    dict(
        type='Collect',
        keys=('coord', 'grid_coord', 'segment', 'instance', 'origin_coord',
              'origin_segment', 'origin_instance', 'inverse', 'name'),
        feat_keys=('coord', ),
        offset_keys_dict=dict(offset='coord', origin_offset='origin_coord'))
]
data = dict(
    num_classes=3,
    ignore_index=-1,
    names=['ground', 'wood', 'leaf'],
    train=dict(
        type='FORInstanceDataset',
        split='train',
        data_root='/home/wyj/225/fgl/data/forinstance_pointcept',
        transform=[
            dict(
                type='CylinderCrop', radius=8, point_max=480000,
                mode='random'),
            dict(type='CenterShift', apply_z=True),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='x',
                center=[0, 0, 0],
                p=0.5),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='y',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(
                type='GridSample',
                grid_size=0.2,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(
                type='TreeInstanceParser',
                segment_ignore_index=(0, ),
                instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'instance'),
                feat_keys=('coord', ))
        ],
        test_mode=False,
        loop=30),
    val=dict(
        type='FORInstanceDataset',
        split='val',
        data_root='/home/wyj/225/fgl/data/forinstance_pointcept',
        transform=[
            dict(
                type='CylinderCrop', radius=8, point_max=480000,
                mode='center'),
            dict(type='CenterShift', apply_z=True),
            dict(
                type='Copy',
                keys_dict=dict(
                    coord='origin_coord',
                    segment='origin_segment',
                    instance='origin_instance')),
            dict(
                type='GridSample',
                grid_size=0.2,
                hash_type='fnv',
                mode='train',
                return_inverse=True,
                return_grid_coord=True),
            dict(
                type='TreeInstanceParser',
                segment_ignore_index=(0, ),
                instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'instance',
                      'origin_coord', 'origin_segment', 'origin_instance',
                      'inverse', 'name'),
                feat_keys=('coord', ),
                offset_keys_dict=dict(
                    offset='coord', origin_offset='origin_coord'))
        ],
        test_mode=False),
    test=dict(
        type='FORInstanceDataset',
        split='test',
        data_root='/home/wyj/225/fgl/data/forinstance_pointcept',
        transform=[
            dict(
                type='CylinderCrop', radius=8, point_max=480000,
                mode='center'),
            dict(type='CenterShift', apply_z=True),
            dict(
                type='Copy',
                keys_dict=dict(
                    coord='origin_coord',
                    segment='origin_segment',
                    instance='origin_instance')),
            dict(
                type='GridSample',
                grid_size=0.2,
                hash_type='fnv',
                mode='train',
                return_inverse=True,
                return_grid_coord=True),
            dict(
                type='TreeInstanceParser',
                segment_ignore_index=(0, ),
                instance_ignore_index=-1),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'instance',
                      'origin_coord', 'origin_segment', 'origin_instance',
                      'inverse', 'name'),
                feat_keys=('coord', ),
                offset_keys_dict=dict(
                    offset='coord', origin_offset='origin_coord'))
        ],
        test_mode=False))
