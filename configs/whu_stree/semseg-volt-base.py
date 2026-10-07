"""Volt semantic segmentation on WHU-STree's 20-class training set.

Training, data transforms, losses, and checkpoint settings follow the WHU-STree
LitePT config. The Volt architecture follows configs/scannetpp/semseg-volt-base.py
with four input channels for XYZ plus normalized intensity.
"""

_base_ = ["./semseg-litept-v1m1-0-small.py"]

save_path = "exp/whu_stree/semseg-volt-base"

model = dict(
    backbone_out_channels=128,
    backbone=dict(
        _delete_=True,
        type="Volt",
        in_channels=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        init_values=None,
        qk_norm=True,
        drop_path=0.3,
        stride=5,
        kernel_size=5,
        increase_drop_path=True,
        up_mlp_dim=128,
    ),
)
