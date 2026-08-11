# CityViMD-Net — 都是同龄人队

> **面向城市场景的视觉多模态目标检测**
> 
> City Visual Multimodal Detection Network
>
> 参赛队伍：**都是同龄人队**
>
> 作者：**Codex / Claude**

## 📖 项目简介

CityViMD-Net 是针对**全球校园人工智能算法精英大赛**「面向城市场景的视觉多模态目标检测」赛题的解决方案。

项目采用**可见光(RGB) + 热红外(Infrared) + 深度(Depth)** 五通道早期融合输入，使用单个共享骨干完成目标检测。

## 🏗️ 网络架构

```
输入: RGB(3ch) + IR(1ch) + Depth(1ch)
    ↓
┌─────────────────────────────────────┐
│  五通道输入直接拼接                  │
│  RGB(3) + IR(1) + Depth(1)          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  单个共享 CSPDarknet 骨干            │
└─────────────────────────────────────┘
    ↓ P2/P3/P4/P5 多尺度特征
┌─────────────────────────────────────┐
│  Neck: YOLOv8 PAN                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Head: YOLOv8 解耦检测头 + DFL      │
└─────────────────────────────────────┘
    ↓
输出: 12类检测框 + 置信度
```

### 方案特点

1. **早期融合**：三种模态在输入端直接组成五通道张量。
2. **共享骨干**：使用一个 CSPDarknet 提取 P2/P3/P4/P5 多尺度特征。
3. **统一检测接口**：PAN 与 YOLOv8 风格检测头完成分类和边界框回归。

## 📁 项目结构

```
CityViMD/
├── configs/
│   └── default.yaml          # 默认配置文件
├── data/                     # 数据集目录
│   ├── train/
│   │   ├── rgb/
│   │   ├── infrared/
│   │   ├── depth/
│   │   └── labels/
│   ├── val/
│   └── test/
├── datasets/
│   ├── __init__.py
│   └── multimodal_dataset.py # 多模态数据集加载
├── models/
│   ├── __init__.py
│   ├── model.py              # 完整模型组装
│   ├── backbone.py           # 骨干网络 (CSPDarknet)
│   ├── neck.py               # 特征金字塔 (PAN)
│   └── head.py               # 检测头 (YOLOv8)
├── utils/
│   ├── __init__.py
│   ├── loss.py               # 损失函数
│   ├── metrics.py            # 评估指标 (mAP)
│   └── visualize.py          # 可视化工具
├── runs/                     # 训练/测试输出
├── tools/                    # 工具脚本
├── train.py                  # 训练脚本
├── test.py                   # 测试/推理脚本
├── requirements.txt          # 依赖包
└── README.md
```

## 🚀 快速开始

最短上手路径见 [`SUBMISSION.md`](SUBMISSION.md)。

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 数据准备

三种模态必须文件名一致、原始高宽一致。训练前建议执行：

```bash
python tools/validate_dataset.py --split train
python tools/validate_dataset.py --split val
python tools/smoke_test.py
```

一键执行：

```bash
python tools/validate_pipeline.py --split train --train --test --package --tta
```

训练过程中，`runs/train/weights/` 保存最终权重，`runs/train/checkpoints/` 保存按轮快照。

### 3. 训练

```bash
# 使用默认配置训练
python train.py --config configs/default.yaml

# 指定参数
python train.py --config configs/default.yaml \
    --epochs 300 \
    --batch-size 16 \
    --data data/ \
    --device 0

# 恢复训练
python train.py --config configs/default.yaml \
    --resume runs/train/weights/last.pt
```

### 4. 测试与校验

```bash
python test.py --config configs/default.yaml \
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

输出格式：`class_id cx cy w h confidence`

## 📊 数据集说明

### 类别信息（12类）

| ID | 类别 | 说明 |
|----|------|------|
| 0 | person | 行人 |
| 1 | boat | 船 |
| 2 | animal | 动物 |
| 3 | seat | 座椅 |
| 4 | sign | 标识（路牌、标语、标志） |
| 5 | bicycle | 双轮车（自行车、电动车） |
| 6 | car | 四轮汽车 |
| 7 | ball | 球 |
| 8 | light | 灯（路灯、照明灯） |
| 9 | garbage_can | 垃圾箱 |
| 10 | uav | 无人机 |
| 11 | tricycle | 三轮车 |

### 数据规模

- 训练集：2,000 组
- 初赛测试集：1,000 组
- 复赛测试集：1,000 组
- 半决赛测试集：1,000 组

### 数据格式

- **RGB**：3通道 8位 PNG，[0, 255]
- **Infrared**：3通道 8位 PNG（实际为灰度），[0, 255]，值越高温度越高
- **Depth**：1通道 16位 PNG，单位毫米，有效范围 [0, 19999]

## 📈 评估指标

- **主指标**：mAP@50-95（IoU 阈值 0.50~0.95，步长 0.05，共10个阈值的平均 AP）
- **其他指标**：mAP@50、mAP@75

### 调参记录
共 11 次提交。最高分为 **64.233**，时间为 **2026-08-10 00:00:00**。

## ⚙️ 配置说明

主要配置项见 `configs/default.yaml`：

- `data`：数据集路径、模态选择、图像尺寸、类别
- `model`：骨干网络、融合模块、Neck、检测头参数
- `train`：训练轮数、批次大小、优化器、学习率、数据增强
- `test`：推理阈值、最大检测数

## 🎯 赛题约束

✅ **允许**：
- 使用 ImageNet、COCO、Objects365 等公开预训练权重
- 自行设计网络架构和融合方法

❌ **禁止**：
- 调用任何在线服务或 API（必须离线）
- 手工标注测试集
- 模型集成（投票法、平均法等）
- 使用外部训练数据
- 将测试集用于训练
