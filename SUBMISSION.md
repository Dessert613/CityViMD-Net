# CityViMD-Net 提交说明

本仓库提供训练、验证、推理、结果校验和源码打包流程。比赛最终提交仍需要使用
官方训练数据训练得到的 `best.pt`，并在官方测试集上生成预测文件。仓库不包含
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

默认训练配置已启用 `AMP` 和 `EMA`，对应开关见 `configs/default.yaml` 中的 `train.amp`、`train.use_ema` 和 `train.ema_decay`。

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

如果需要更稳的提交结果，可以在 `test.py` 中附加 `--tta` 启用简单水平翻转测试时增强。

如果想把整条链路一次性跑完，可以使用：

```bash
python tools/validate_pipeline.py --train --test --package
```

如果需要连同 TTA、数据校验和预测文件校验一起跑完，可以使用：

```bash
python tools/validate_pipeline.py --split train --train --test --package --tta
```

常用参数还包括：

- `--split train|val`：选择先校验哪个数据划分
- `--weights runs/train/weights/best.pt`：指定测试权重
- `--test-input data/test`：指定测试输入目录
- `--test-output runs/test/predictions`：指定预测输出目录
- `--tta`：测试时启用翻转增强
- `--skip-smoke`：跳过冒烟测试
- `--skip-predictions`：跳过预测结果文件校验

## 5. 打包源码

```bash
python tools/package_submission.py
```

源码压缩包生成于 `runs/submission/cityvimd_source.zip`。半决赛提交时还需按官方
要求附上预测 ZIP、训练权重、运行环境说明和技术报告。

## 提交前检查表

- 数据校验脚本通过
- 冒烟测试通过
- 训练日志中无 NaN/Inf
- 使用最佳验证集 mAP@50-95 对应的权重
- 测试图与预测 TXT 数量完全一致
- 预测校验脚本通过
- ZIP 根目录直接包含 TXT 文件，没有多余目录层级
- 技术报告中的模型、参数和成绩均与最终提交一致

## 6. 工程规划

下面这份规划用于把当前方案拆成可执行、可验证的工程步骤，避免只停留在概念层面。

| 阶段 | 目标 | 主要工作 | 产出 |
|---|---|---|---|
| 阶段 A | 数据与链路稳定 | 校验三模态文件名、高宽、深度读取、标签格式、空预测和 ZIP 结构 | 可重复的训练/推理闭环 |
| 阶段 B | 基线收敛 | 固化五通道早期融合、共享骨干、PAN 和检测头的主干配置 | 可复现的基线权重 |
| 阶段 C | 表征增强 | 增加模态对齐与轻量融合控制 | 更稳定的跨模态特征 |
| 阶段 D | 训练提效 | 调整损失权重、学习率策略、warmup、EMA 与混合精度 | 更优的收敛质量 |
| 阶段 E | 后处理优化 | 类别感知 NMS、阈值搜索、TTA 和最大检测数控制 | 更稳的提交结果 |
| 阶段 F | 冲分验证 | 做消融、复核随机种子、输出格式和提交包 | 最终提交版本 |

### 当前状态

- 阶段 A：已完成
- 阶段 B：已完成
- 阶段 C：已完成基础版，增强版待做
- 阶段 D：已完成主要调参，EMA/AMP 待评估
- 阶段 E：已完成主要后处理策略，TTA 待评估
- 阶段 F：已完成复核提交版本，最高分记录为 **64.233**

### 下一步建议

- 如果目标是继续冲分，优先做阶段 C 和阶段 E 的小步迭代。
- 如果目标是正式提交，优先冻结配置、复查输出并固定最终权重。
- 如果目标是写技术报告，优先整理每个阶段的实验依据和消融结论。
