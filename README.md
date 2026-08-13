# CityViMD-Net

[![CI](https://github.com/forever-ivy/CityViMD-Net/actions/workflows/ci.yml/badge.svg)](https://github.com/forever-ivy/CityViMD-Net/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

面向 AIC2026 全球校园人工智能算法精英大赛「面向城市场景的视觉多模态目标检测」
赛题的 RGB、红外与深度公开基线。

A reproducible RGB–infrared–depth early-fusion baseline for urban multimodal
object detection.

> 维护团队：**都是同龄人队**
>
> 本仓库用于维护可复现的公开基线，不代表全部内部实验，也不保证与任何最终参赛版本完全一致。

## 项目定位

CityViMD-Net 使用已对齐的 RGB、红外和深度图像组成五通道输入，通过单个共享
CSPDarknet 风格骨干提取特征，再由 PAN 和解耦检测头完成分类与边界框回归。

公开基线重点关注：

- 三模态文件、尺寸和数值范围的一致性校验；
- 简单、低开销的五通道早期融合；
- 可重复的训练、验证、推理和结果打包流程；
- 不依赖比赛数据的合成输入冒烟测试。

当前实现不包含跨模态注意力、动态模态加权或多模型集成。完整边界与设计理由见
[`docs/architecture.md`](docs/architecture.md)。

## 架构

```text
RGB (3ch) ─────┐
Infrared (1ch) ├─ concat (5ch) ─ CSPDarknet ─ PAN ─ YOLOv8-style head
Depth (1ch) ───┘                                      │
                                                      └─ boxes/classes
```

默认输入为 `640 × 640`，检测头使用 P3、P4、P5 三个尺度以及 DFL 边界框回归。

## 快速开始

推荐使用 Python 3.10 及独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

先运行不依赖比赛数据的检查：

```bash
python tools/smoke_test.py
```

开发者可额外安装测试依赖：

```bash
python -m pip install -r requirements-dev.txt
pytest
```

## 数据目录

三种模态必须使用相同文件名，并保持原始高宽一致：

```text
data/
├── train/
│   ├── rgb/*.png
│   ├── infrared/*.png
│   ├── depth/*.png
│   └── labels/*.txt
├── val/
│   ├── rgb/*.png
│   ├── infrared/*.png
│   ├── depth/*.png
│   └── labels/*.txt
└── test/
    ├── rgb/*.png
    ├── infrared/*.png
    └── depth/*.png
```

图像约定：

- RGB：3 通道 8 位 PNG；
- Infrared：以单通道灰度读取的 8 位 PNG；
- Depth：1 通道 16 位 PNG，按 20,000 mm 截断并归一化；
- 标签：YOLO 格式 `class_id cx cy w h`，坐标归一化到 `[0, 1]`。

训练前执行：

```bash
python tools/validate_dataset.py --split train
python tools/validate_dataset.py --split val
```

## 训练与推理

训练：

```bash
python train.py --config configs/default.yaml
```

恢复训练：

```bash
python train.py \
  --config configs/default.yaml \
  --resume runs/train/weights/last.pt
```

生成并校验预测：

```bash
python test.py \
  --config configs/default.yaml \
  --weights runs/train/weights/best.pt \
  --input data/test \
  --output runs/test/predictions \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --zip

python tools/validate_predictions.py \
  --images data/test/rgb \
  --predictions runs/test/predictions
```

预测文件每行格式为：

```text
class_id cx cy w h confidence
```

完整操作说明见 [`SUBMISSION.md`](SUBMISSION.md)，复现要求见
[`docs/reproducibility.md`](docs/reproducibility.md)。

## 基准结果

仓库目前不发布无法由公开产物验证的排行榜成绩。任何后续基准必须同时记录配置、
代码提交、随机种子、运行环境和原始日志；记录格式见
[`BENCHMARKS.md`](BENCHMARKS.md)。

## 类别与指标

基线默认包含 12 类目标：`person`、`boat`、`animal`、`seat`、`sign`、
`bicycle`、`car`、`ball`、`light`、`garbage_can`、`uav` 和 `tricycle`。

主评估指标为 mAP@50-95，同时报告 mAP@50 与 mAP@75。

## 项目结构

```text
CityViMD-Net/
├── configs/                 # 公开基线配置
├── datasets/                # 多模态数据读取
├── docs/                    # 架构与复现说明
├── models/                  # Backbone、PAN 与检测头
├── tests/                   # 自动化测试
├── tools/                   # 校验、冒烟和打包工具
├── utils/                   # 损失、指标、EMA 和可视化
├── train.py
├── test.py
├── BENCHMARKS.md
└── SUBMISSION.md
```

## 限制

- 公开基线假设三种模态已经完成像素级空间对齐；
- 当前仅实现早期融合，无法单独处理缺失模态；
- 仓库不包含比赛数据、训练权重或测试集预测；
- 配置中的阈值是工程默认值，不应被视为对任意数据集的最优值。

## 贡献

提交 Issue 或 Pull Request 前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。
请勿上传比赛数据、受限制权重、凭据或无法公开验证的成绩。

## 许可证与引用

代码按 [MIT License](LICENSE) 发布。引用信息见 [`CITATION.cff`](CITATION.cff)。
