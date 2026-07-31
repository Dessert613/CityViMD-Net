# CityViMD-Net 提交说明

本仓库提供训练、验证、推理、结果校验和源码打包流程。比赛最终提交仍需要使用
官方训练数据训练得到的 `best.pt`，并在官方测试集上生成预测文件。本仓库不包含
数据或训练权重，性能成绩详见项目记录。

## 1. 环境

推荐 Python 3.10、PyTorch 2.x 和 CUDA 11.8 以上版本。

```bash
python -m pip install -r requirements.txt
python tools/smoke_test.py
```

## 2. 数据目录

三种模态必须具有完全相同的文件名和原始高宽：

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

训练前执行：

```bash
python tools/validate_dataset.py --split train
python tools/validate_dataset.py --split val
```

## 3. 训练

```bash
python train.py --config configs/default.yaml
```

最佳权重默认保存至 `runs/train/weights/best.pt`。恢复训练：

```bash
python train.py --config configs/default.yaml \
  --resume runs/train/weights/last.pt
```

## 4. 生成和验证预测

```bash
python test.py --config configs/default.yaml \
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

输出 ZIP 位于 `runs/test/predictions.zip`。脚本会为每张测试图创建同名 TXT；
未检测到目标时生成空文件，每张图最多保留 100 个检测框。

## 5. 打包源码

```bash
python tools/package_submission.py
```

源码压缩包生成于 `runs/submission/cityvimd_source.zip`。半决赛提交时还需按官方
要求附上预测 ZIP、训练权重、运行环境说明和技术报告。

## 提交前检查表

- 数据校验脚本通过。
- 冒烟测试通过。
- 训练日志中无 NaN/Inf。
- 使用最佳验证集 mAP@50-95 对应的权重。
- 测试图与预测 TXT 数量完全一致。
- 预测校验脚本通过。
- ZIP 根目录直接包含 TXT 文件，没有多余目录层级。
- 技术报告中的模型、参数和成绩均与最终提交一致。
