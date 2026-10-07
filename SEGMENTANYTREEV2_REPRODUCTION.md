# SegmentAnyTreeV2：基于 Pointcept 的复现说明

本目录实现了论文 [*SegmentAnyTreeV2: A generalist model for tree instance
segmentation in complex forest point clouds*](https://arxiv.org/abs/2606.08206)
的可训练复现。
PTv3 使用 Pointcept 原生实现；语义头、ISA、树点过滤、掩码解码器、论文损失、
FOR-instance 数据适配、指标和整图滑窗推理均在 Pointcept 内完成。

这不是作者官方代码。论文截至 2026-09-04 仍说明代码、FOR-instance v3 和权重将
在接收后公开，因此当前实现以论文正文、公式和附录表为依据。

## 实现范围

| 模块 | 位置 | 对齐内容 |
|---|---|---|
| SaTv2 模型 | `pointcept/models/segment_any_tree_v2/` | PTv3、3 类语义头、5-D embedding、400-query ISA、6 层掩码解码器、NMS |
| 数据集与裁块 | `pointcept/datasets/for_instance.py` | 16 m 圆柱、650k 点上限、正实例 ID 过滤 |
| 数据预处理 | `pointcept/datasets/preprocessing/for_instance/preprocess_for_instance.py` | PLY 到 Pointcept scene layout，标签 1/2/3 转 0/1/2 |
| 训练配置 | `configs/for_instance/insseg-satv2-v1m1-0-ptv3.py` | Table A.5 网络、1000 epoch、AdamW、OneCycle、梯度累积与数据增强 |
| 训练期验证 | `pointcept/engines/hooks/satv2_evaluator.py` | 全局 P/R/F1、逐图 MWCov 后平均、语义 IoU/mIoU |
| 独立测试 | `pointcept/engines/test_satv2.py` | SaTv2 专用 Pointcept tester，输出 `satv2_metrics.json` |
| 整图推理 | `tools/infer_satv2.py` | 16 m/4 m 滑窗、语义投票、局部/全局合并、PLY 和 JSON 输出 |

## 1. 准备数据

输入 PLY 须至少含 `x/y/z`、`semantic_seg` 和 `treeID`。论文数据约定为：

- `semantic_seg`: 1=ground、2=wood、3=leaf；
- `treeID`: 正整数表示有效树实例，0 或负数表示背景/无实例标注。

源目录可按 split 建目录：

```text
/data/FOR-instance/
├── train/*.ply
├── val/*.ply
└── test/*.ply
```

也可把文件平铺放置，但文件名必须以 `_train`、`_val` 或 `_test` 结尾。转换：
在 v3 尚未公开时，可以先用公开的
[FOR-instance v2](https://zenodo.org/records/16742708) 验证训练和测试链路。

```bash
cd /home/wyj/225/fgl/Pointcept
python pointcept/datasets/preprocessing/for_instance/preprocess_for_instance.py \
  --source-root /home/wyj/225/fgl/data/forinstance \
  --output-root /home/wyj/225/fgl/data/forinstance_pointcept
```

若已有数据标签为 0/1/2，额外传入 `--semantic-offset 0`。脚本默认使用论文的
`--semantic-offset 1`，并逐文件生成 `manifest.json` 记录类别数和树实例数。

## 2. 训练

严格论文配置需要大显存；论文使用单张 A100 80 GB、batch size 6、累积 4 步，
有效 batch size 为 24。启动命令：

```bash
cd /home/wyj/225/fgl/Pointcept
FOR_INSTANCE_ROOT=/data/for_instance_pointcept \
bash scripts/train.sh \
  -g 1 \
  -d for_instance \
  -c insseg-satv2-v1m1-0-ptv3 \
  -n satv2_ptv3
```

Pointcept 会把 10 次逻辑数据遍历折叠为一个 trainer epoch，因此配置中的
`epoch=1000, eval_epoch=100` 实际仍是论文的 1000 次数据遍历、每 10 次验证。
ISA 在逻辑 epoch 30 启用；此前仅训练语义头和判别 embedding。

检查点写入：

```text
exp/for_instance/satv2_ptv3/model/model_last.pth
exp/for_instance/satv2_ptv3/model/model_best.pth
```

## 3. 评估

下面的 Pointcept tester 是单个 16 m 中心圆柱的快速验证路径：

```bash
cd /home/wyj/225/fgl/Pointcept
FOR_INSTANCE_ROOT=/data/for_instance_pointcept \
bash scripts/test.sh \
  -g 1 \
  -d for_instance \
  -n satv2_ptv3 \
  -w model_best
```

论文表格的 plot-level 数值应使用下一节的整图推理；快速 tester 不会替代论文的
跨圆柱全局合并。

## 4. 整图滑窗推理

```bash
cd /home/wyj/225/fgl/Pointcept
PYTHONPATH=. python tools/infer_satv2.py \
  /data/FOR-instance/test/plot.ply \
  output/plot_satv2.ply \
  --weight exp/for_instance/satv2_ptv3/model/model_best.pth
```

输出 PLY 保留原字段和坐标，并新增：

- `semantic_pred`：1/2/3；
- `instance_pred`：0 为非树，正数为连续树 ID；
- `instance_confidence`：该点所属实例的置信度。

若输入含真值，工具还会生成 `plot_satv2.ply.json`，报告 P/R/F1、逐图 MWCov
（`coverage`）及语义 IoU/mIoU。

## 5. 论文未公开或存在冲突的部分

以下选择被显式隔离，作者代码公开后可直接替换：

1. 论文只说明 mask memory projection 比 attention projection 更深，没有发布精确
   层序列。本实现采用 `Linear` attention projection，以及
   `Linear-LN-ReLU-Linear-ReLU` mask projection。
2. 论文未给出 FPS 的首点选择规则；本实现使用 `torch_cluster.fps` 在 5-D
   embedding 中采样，并以首个候选点作为确定性起点（`random_start=False`）。
3. 论文没有逐项给出 PTv3 的 stride 和各 stage head 数；本实现保持 Pointcept PTv3
   的每头 16 维设计，并严格使用论文公开的 channel/depth/patch 配置。
4. 公式给出 `decoder_loss_weight` 名称但未给数值，本实现使用中性值 1.0。
5. 正文推理阈值为 confidence=0.5667、global overlap=0.1473、size=20，附录
   B.7–B.9 则为 0.15、0.1、10。本实现默认采用附录配置表的 0.15/0.1/10，所有值
   均可通过 `tools/infer_satv2.py --help` 覆盖。
6. 附录给出局部 ground-height filter 参数，但没有完整公开人工地面网格和插值算法；
   当前整图工具没有猜测实现该过滤器，只执行论文明确的 ground semantic overwrite。
   因此在作者代码、权重和 FOR-instance v3 未公开前，不能声称数值级完全复现论文
   85.0 F1 / 90.7 Cov。

## 6. 自检

```bash
cd /home/wyj/225/fgl/Pointcept
python -m pytest -q tests/test_satv2.py
```

测试覆盖 embedding-space FPS、无树边界、ISA 解码器、所有损失反向传播、NMS、
正实例过滤及论文主指标。
