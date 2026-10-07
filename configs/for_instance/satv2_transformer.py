"""SpUNet-v2m1 SaTv2 Transformer baseline using the restored myself model."""

_base_ = ["./insseg-satv2-v1m1-0-ptv3.py"]
custom_imports = dict(imports=["pointcept.models.myself"], allow_failed_imports=False)
save_path = "exp/for_instance/satv2_transformer"
model = dict(
    type="SaTv2-myself-v1m1",
    backbone=dict(
        _delete_=True,
        type="SpUNet-v2m1",
        in_channels=3,
        num_channels=128,
        num_planes=(128, 128, 256, 128),
        return_blocks=True,
    ),
    backbone_out_channels=128,
    decoder=dict(type="transformer"),
)
