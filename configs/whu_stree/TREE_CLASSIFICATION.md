# WHU-STree：实例分割后逐树分类

训练配置为 `cls-ptv3-v1m1-0-tree-species.py`，参考 ModelNet40 的 PTV3 分类配置。
使用 `DefaultClassifier`、PTV3 encoder 和全局均值池化。输入是 XYZ＋归一化强度，
19 个输出类别为 18 个主要树种＋Others；分类编号 0–18 对应原语义编号 1–19。
分类编号 0 是第一个树种，不是非树。非树不会成为分类训练样本。

每棵树离线最多均匀随机保留 65536 点，训练时再从整棵树随机抽取最多 8192 点，
避免局部球裁剪只保留树的一部分。坐标归一化到单位球，之后用 0.01 的网格采样。
强度与已有预处理保持一致，原始强度除以 65535。训练 300 epoch，batch size 16，
AdamW 基础学习率 0.001、block 学习率 0.0001、weight decay 0.01，使用 OneCycleLR。
与 ModelNet40 分类基线一样关闭 AMP，兼容当前 spconv 的训练和推理执行路径。

## 1. 训练数据预处理

复用已经生成的 `WHU-Stree-pointcept-20class`，按真实实例拆成单树。
一棵树的类别取已标注树点的多数树种，至少占 80%；全部未知或混合树种样本跳过。
`segment == 0` 和非正实例 ID 的点被排除；未知树种点不参与类别投票。
原始 train 和 val 的全部 206 个场景统一用于训练，不剔除两个来源中重复出现的树 ID。
不划分验证集，不进行训练期间或训练结束后的评估。

在仓库根目录运行：

```bash
python pointcept/datasets/preprocessing/whu_stree/preprocess_tree_classification.py train \
  --source-root /home/wyj/225/fgl/data/WHU-Stree-pointcept-20class \
  --output-root data/whu_stree_treecls_all
```

输出为 `data/whu_stree_treecls_all/train/<scene>__tree_<id>/`，
每棵树含 `coord.npy`、`strength.npy`、`category.npy` 和 `metadata.json`。
`classes.json` 记录类别映射；`train_summary.json` 记录各类树的数量。
同参数重复运行会跳过已完成的场景；改变采样或标签参数时应使用新的输出目录。
使用新的 `_all` 目录，避免复用先前按 train/val 分开的样本。

## 2. 在物理卡 1 上训练

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
WHU_TREECLS_ROOT=data/whu_stree_treecls_all \
python tools/train.py \
  --config-file configs/whu_stree/cls-ptv3-v1m1-0-tree-species.py \
  --num-gpus 1
```

训练输出在 `exp/whu_stree/cls-ptv3-tree-species-all/`。
每 epoch 更新 `model_last.pth`，每 25 epoch 额外保存一次快照；没有验证集，因此不生成 `model_best.pth`。
分类配置中的 `data.test` 供后续单树推理脚本读取无标签测试树，不在训练期间加载。

## 3. 用 Mamba 最佳模型保存两个标签

旧的 `tools/infer_whu_instances.py` 只输出实例，并把背景填入最近实例。
新的双标签推理入口为：

```bash
OMP_NUM_THREADS=4 python -u -m tools.infer_whu_semantic_instances --device cuda:1
```

默认加载 `exp/whu_stree/satv2_mamba_aggregation/model/model_best.pth`，
使用该实验保存的 `config.py`，对 `/home/wyj/225/fgl/data/test` 逐场景推理。
输出在 `exp/whu_stree/satv2_mamba_aggregation/mamba_best_semantic_instance/<scene>/`：

- `semantic.npy`：`uint8`，0=非树、1=树。
- `instance.npy`：`int32`，正数为树实例 ID；非树点固定为 0。
- `manifest.json`：原始点数、非树点数、实例数、推理参数及未分配树点数。

两份数组都与原始 PLY 的 vertex 顺序一致。重叠圆柱累加语义概率、合并实例掩码；
仅在语义为树的点中填充未分配实例。若一个场景没有可用实例掩码，它的树点保持 ID 0，
记录为未分配，不进入单树分类。已有原始推理结果保留。

## 4. 准备测试树并分类

完成全部双标签推理后，只提取 `semantic == 1` 且 `instance > 0` 的点。
不使用最大实例猜测背景，也不需要 Volt 树点掩码。

```bash
python pointcept/datasets/preprocessing/whu_stree/preprocess_tree_classification.py test \
  --prediction-root exp/whu_stree/satv2_mamba_aggregation/mamba_best_semantic_instance \
  --input-dir /home/wyj/225/fgl/data/test \
  --output-root data/whu_stree_treecls_all
```

分类器训练完之后可直接读取原始 PLY 和 Mamba 双标签推理，无需预处理测试树：

```bash
OMP_NUM_THREADS=4 python -u -m tools.infer_whu_tree_species \
  --config exp/whu_stree/cls-ptv3-tree-species/config.py \
  --weight exp/whu_stree/cls-ptv3-tree-species/model/model_last.pth \
  --input-dir /home/wyj/225/fgl/data/test \
  --prediction-root exp/whu_stree/satv2_mamba_aggregation/mamba_best_semantic_instance \
  --output-dir exp/whu_stree/cls-ptv3-tree-species/instance_species \
  --device cuda:1
```

结果在 `exp/whu_stree/cls-ptv3-tree-species/instance_species/<scene>.npy`。
每个数组形状为 `(N, 2)`，类型为 `int32`，与原始 PLY 点顺序一致。
第一列保留 Mamba 的原始实例 ID；第二列为树种语义：0–18=分类器树种编号、
-1=原非树点或未分类树点。`<scene>_trees.json` 保存每棵树的树种和置信度，
`manifest.json` 记录列名和类别名称。输出编号 0 对应第一个树种，不代表非树。
