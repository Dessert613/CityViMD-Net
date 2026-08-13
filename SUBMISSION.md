# CityViMD-Net 运行与打包说明

本仓库提供公开基线的训练、验证、推理、预测校验和源码打包流程。仓库不包含数据、
训练权重或测试集预测。使用比赛数据、预训练权重和提交文件前，请先确认对应赛事
规则及数据许可。

## 1. 环境

推荐 Python 3.10、PyTorch 2.x。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tools/smoke_test.py
```

开发与测试环境：

```bash
python -m pip install -r requirements-dev.txt
pytest
```

## 2. 数据准备

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

三种模态必须具有完全相同的文件名和原始高宽。训练前执行：

```bash
python tools/validate_dataset.py --split train
python tools/validate_dataset.py --split val
```

数据校验只确认目录、文件对应关系、图像尺寸和标签格式，不代替人工检查数据许可、
类别定义和标注质量。

## 3. 训练

```bash
python train.py --config configs/default.yaml
```

默认输出：

```text
runs/train/
├── config.yaml
├── logs/
├── checkpoints/
└── weights/
    ├── last.pt
    └── best.pt
```

恢复训练：

```bash
python train.py \
  --config configs/default.yaml \
  --resume runs/train/weights/last.pt
```

默认配置启用 AMP 和 EMA。随机种子、优化器、损失权重及验证间隔均记录在
`configs/default.yaml` 中。正式记录基准时还应保存代码提交、环境和原始日志，
详见 [`docs/reproducibility.md`](docs/reproducibility.md)。

## 4. 推理与结果校验

```bash
python test.py \
  --config configs/default.yaml \
  --weights runs/train/weights/best.pt \
  --input data/test \
  --output runs/test/predictions \
  --conf-thres 0.001 \
  --iou-thres 0.7 \
  --max-det 100 \
  --zip

python tools/validate_predictions.py \
  --images data/test/rgb \
  --predictions runs/test/predictions
```

输出 ZIP 位于 `runs/test/predictions.zip`。每张输入图像必须存在同名 TXT；
未检测到目标时生成空文件。每行格式为：

```text
class_id cx cy w h confidence
```

`--tta` 可启用简单水平翻转测试时增强。该选项只是公开基线能力，不表示它在任意
数据集上都会提升指标。

## 5. 组合校验

基础链路：

```bash
python tools/validate_pipeline.py --split train
```

包含训练、推理、预测校验和源码打包的完整链路：

```bash
python tools/validate_pipeline.py \
  --split train \
  --train \
  --test \
  --package
```

完整链路需要本地数据和训练权重，运行成本较高。持续集成只运行不依赖比赛数据的
单元测试、模型冒烟测试和源码打包检查。

## 6. 打包源码

```bash
python tools/package_submission.py
```

源码压缩包生成于 `runs/submission/cityvimd_source.zip`。打包脚本不会包含
`data/`、`runs/`、权重或预测结果。

## 发布或提交前检查表

- 已确认赛事规则允许对应代码、数据和权重的使用方式；
- 数据校验、单元测试和冒烟测试全部通过；
- 训练日志中没有 NaN 或 Inf；
- 记录了配置、代码提交、随机种子、依赖版本和硬件环境；
- 测试图像与预测 TXT 数量完全一致；
- 预测文件字段、范围与 ZIP 层级通过校验；
- 源码包中不含数据、权重、凭据、私有日志或测试集预测；
- 文档中的结构和成绩均可由公开产物验证。
