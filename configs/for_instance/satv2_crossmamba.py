"""SpUNet-v2m1 SaTv2 local Cross-Mamba + Spatial Mamba ablation."""

_base_ = ["./satv2_mamba_aggregation.py"]
save_path = "exp/for_instance/satv2_crossmamba"
model = dict(decoder=dict(cross_mode="mamba"))
