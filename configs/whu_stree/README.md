# WHU-Stree 预处理与训练

## 官方树种标签与 20 类预处理

官方定义：`tree` 是树实例 ID，`label` 是树种 ID。官方基准使用 18 个主要
树种（原始 ID 0–17）和 Others（原始 ID ≥18 合并）。加入非树背景后，共 20
个语义类别。来源：[官方说明](https://github.com/WHU-USI3DV/WHU-STree#2-folder-structure-and-format)
与[官方编号表](https://raw.githubusercontent.com/WHU-USI3DV/WHU-STree/main/media/species.png)。

| 处理后 ID | 含义 | 原始条件 |
|---|---|---|
| 0 | non-tree | `tree == 0` |
| 1–18 | 官方 18 个主要树种，保持原表顺序 | `tree > 0` 且 `label` 为 0–17，取 `label + 1` |
| 19 | Others | `tree > 0` 且 `label >= 18` |
| -1 | 树种未标注，语义训练/评估忽略 | `tree > 0` 且 `label == -1` |

原始数据中存在 `tree > 0, label == -1` 的点，因此不能把所有原始 `label == -1`
都作为非树。这里保持未知树种为忽略标签，实例 ID 仍保留。

```bash
python pointcept/datasets/preprocessing/whu_stree/preprocess_whu_stree.py \
  --source-root /home/wyj/225/fgl/data/WHU-Stree_random50 \
  --output-root /home/wyj/225/fgl/data/WHU-Stree-pointcept-20class \
  --semantic-mode species20 \
  --merge-splits-to-train
```

输出仍是每个场景的 `coord.npy`、`strength.npy`、`segment.npy`、`instance.npy`，
保留全部点和原始顺序；`segment.npy` 为 int16，`instance.npy` 为 int32。
场景 `manifest.json` 包含 20 类点数与忽略点数，根目录 `semantic_classes.json`
包含完整类别名称、官方来源、映射和各划分统计。脚本默认仍使用原来的二分类模式。
上述命令将原始 train 的 152 个场景与 val 的 54 个场景全部写入新目录的
`train/`，共 206 个训练场景，不创建验证集；文件名中的原始后缀保持不变。

使用新数据训练时，需将 `data_root` 指向新目录，语义类别数设为 20、
`ignore_index=-1`、树语义类别设为 `tuple(range(1, 20))`；类别名称读取
`semantic_classes.json` 的 `names`。现有二分类权重的语义头与新类别数不匹配。

## LitePT 20 类语义分割训练

配置：`configs/whu_stree/semseg-litept-v1m1-0-small.py`。
使用 LitePT-S + 20 类语义头，XYZ 和归一化强度共 4 个输入通道，损失为
CrossEntropy + Lovasz，忽略标签为 -1。数据默认指向上述 20 类目录的全部 206
个训练场景，关闭验证及训练结束后的测试。

默认单 GPU、batch size 1、AMP、体素 0.2 m，随机圆柱半径 16 m，
体素采样后最多保留 102400 个点。AdamW 的主学习率为 0.0005，使用 OneCycleLR。
`epoch=eval_epoch=1000` 保证数据 loop 为 1，每个日志 epoch 对应一次数据集遍历。

```bash
cd /home/wyj/225/fgl/Pointcept
conda activate pointcept
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --config-file configs/whu_stree/semseg-litept-v1m1-0-small.py \
  --num-gpus 1
```

训练输出到 `exp/whu_stree/semseg-litept-v1m1-0-small`，每轮更新
`model/model_last.pth`，每 50 轮另存一次 checkpoint。关闭验证时不生成
`model_best.pth`。若修改训练轮数，应同时修改 `epoch` 与 `eval_epoch`；
使用双 GPU 时将全局 `batch_size` 至少设为 2。

## 原有二分类预处理

该配置用于 WHU-Stree 树木实例分割。原始 PLY 位于 `train/` 和 `val/` 子目录，
包含 `x y z intensity tree label` 字段。预处理使用 `tree` 字段：`tree == 0`
映射为非树语义类别 0，`tree != 0` 映射为树语义类别 1；树点保留原始 `tree`
实例 ID，非树点的实例 ID 设为 -1。`label` 字段不参与此二分类实例分割配置。

## 预处理

在仓库根目录运行：

```bash
python pointcept/datasets/preprocessing/whu_stree/preprocess_whu_stree.py \
  --source-root /home/wyj/225/fgl/data/WHU-Stree_random50 \
  --output-root /home/wyj/225/fgl/data/WHU-Stree-pointcept \
  --splits train val
```

输出结构如下，原始 PLY 不会被修改：

```text
/home/wyj/225/fgl/data/WHU-Stree-pointcept/
├── train/<scene>/{coord,strength,segment,instance}.npy
└── val/<scene>/{coord,strength,segment,instance}.npy
```

脚本按块读取点云。已完成的场景会跳过；若要重做已有输出，可追加 `--force`，
它只替换对应的预处理场景目录。

## 训练

默认使用 xyz 特征，单 GPU 运行 LitePT + PointGroup：

```bash
WHU_STREE_ROOT=/home/wyj/225/fgl/data/WHU-Stree-pointcept \
python tools/train.py \
  --config-file configs/whu_stree/insseg-pointgroup-v1m2-0-litept-small.py \
  --num-gpus 1 \
  --options save_path=exp/whu_stree/pointgroup-litept-xyz
```

配置把 `tree` 作为唯一实例类别；非树点仍参与语义分类损失，但不参与实例偏移、
聚类和实例 mAP。若要把归一化强度作为第 4 个输入通道，可设置
`WHU_STREE_USE_INTENSITY=1`，并为实验指定不同的 `save_path`：

```bash
WHU_STREE_ROOT=/home/wyj/225/fgl/data/WHU-Stree-pointcept \
WHU_STREE_USE_INTENSITY=1 \
python tools/train.py \
  --config-file configs/whu_stree/insseg-pointgroup-v1m2-0-litept-small.py \
  --num-gpus 1 \
  --options save_path=exp/whu_stree/pointgroup-litept-xyz-intensity
```

## SaTv2 + SpUNet-v2m1

也可以使用原生 `SaTv2-v1m1` 实例分割头和 SpUNet-v2m1 骨干网络。优化器、学习率计划、
训练轮数和数据增强参数参照 `configs/for_instance/insseg-satv2-v1m1-1-spunetv2m1.py`；
类别数和类别权重适配 WHU-Stree 的二分类标签。SpUNet 输出通道和 SaTv2 decoder
维度均设为 32，以匹配原生 SaTv2 的维度要求。配置使用 xyz 特征、`grid_size=0.2`、
batch size 1，并启用 24 步梯度累积：

```bash
WHU_STREE_ROOT=/home/wyj/225/fgl/data/WHU-Stree-pointcept \
python tools/train.py \
  --config-file configs/whu_stree/insseg-satv2-v1m1-1-spunetv2m1.py \
  --num-gpus 1 \
  --options save_path=exp/whu_stree/satv2-spunetv2m1
```

该配置训练 3000 epoch、每 100 epoch 验证一次；显存不足时可先减小配置中的
`cylinder_radius` 或提高 `grid_size`。
