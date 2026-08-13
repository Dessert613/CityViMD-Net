# 公开基线架构

## 设计目标

CityViMD-Net 的公开实现是一条低复杂度、容易检查的早期融合基线。它用于验证
RGB、红外和深度数据能够经过统一的训练与推理链路，而不是宣称早期融合必然优于
其他多模态结构。

设计约束：

- 三种模态已经完成像素级空间对齐；
- 每个样本的三种图像具有相同文件名和原始高宽；
- RGB 为 3 通道，红外和深度各转换为 1 通道；
- 不依赖在线 API；
- 单个模型完成一次前向推理。

## 数据流

```text
RGB uint8 [H,W,3] ─────────── /255 ───────┐
IR uint8 [H,W] ────────────── /255 ───────┼─ [B,5,H,W]
Depth uint16 [H,W] ─ clip(/20000, 0, 1) ──┘
                                                │
                                                ▼
                                    CSPDarknet-style backbone
                                         P2 / P3 / P4 / P5
                                                │
                                                ▼
                                         PAN (P3 / P4 / P5)
                                                │
                                                ▼
                               decoupled classification/regression head
                                                │
                                                ▼
                                 DFL decode + per-class NMS + max_det
```

所有几何增强必须同步应用于三种模态。RGB 颜色增强和红外 gamma 扰动只改变对应
模态的像素值，不改变空间位置。

可选配置开关（均有默认值，默认行为与基线一致）：

- `data.depth_validity_mask`：官方数据约定深度 0 或过小为无效。开启后深度
  输出两通道 `[归一化深度, 0/1 有效掩码]`，输入变为 `[B, 6, H, W]`，同时须将
  `model.in_channels.depth` 设为 2（模型构建时会校验一致性）；
- `data.depth_encoding`：深度编码方式，`linear`（默认，官方线性截断）/
  `inverse`（逆深度）/ `log` / `minmax`（逐图归一化）。所有编码下无效深度
  统一输出 0；
- `data.ir_encoding`：红外编码方式，`raw`（默认）/ `clahe` / `percentile`；
- `data.test_blacklist`：测试集哈希黑名单路径（`tools/build_test_blacklist.py`
  生成）。文件存在时训练/验证加载器拒绝任何测试集图像；
- `train.augment.modality_dropout`：训练时以给定概率随机将红外或深度整幅
  置零，模拟传感器失效，用于模态劣化鲁棒性。仅作用于训练增强，不改变推理。

训练与推理共享 `datasets/multimodal_dataset.py` 中的编码实现，
保证两侧预处理一致。

## 模块映射

| 模块 | 实现 | 责任 |
|---|---|---|
| 数据读取 | `datasets/multimodal_dataset.py` | 文件对齐、图像读取、增强、缩放与归一化 |
| 骨干 | `models/backbone.py` | 生成 P2、P3、P4、P5 多尺度特征 |
| Neck | `models/neck.py` | 将 P3、P4、P5 进行自顶向下及自底向上融合 |
| 检测头 | `models/head.py` | 分类分支、DFL 回归分支和边界框解码 |
| 损失 | `utils/loss.py` | Task-Aligned 分配、BCE、CIoU 和 DFL |
| 推理 | `test.py` | 预处理、推理、坐标恢复、TTA 和结果写出 |

## 明确不包含的能力

公开基线当前没有实现：

- 动态模态权重；
- 跨模态注意力或 Transformer 融合；
- 独立模态骨干；
- 缺失模态推理；
- 时序信息；
- 模型集成；
- 自动阈值搜索。

模型 `forward` 只返回三个检测尺度的原始预测。任何文档、图表或基准若声称包含
上述能力，都应先有对应代码和测试。

## 形状约定

默认 `640 × 640` 输入下：

- 输入：`[B, 5, 640, 640]`；
- P3 检测尺度：步长约为 8；
- P4 检测尺度：步长约为 16；
- P5 检测尺度：步长约为 32；
- 每尺度输出通道：`num_classes + 4 × (reg_max + 1)`。

训练标签格式为 `[batch_index, class_id, cx, cy, width, height]`，其中框坐标均
归一化。推理内部使用像素坐标 `xyxy`，写出时恢复为归一化 YOLO 格式。

## 扩展原则

新增融合模块时应同时提供：

1. 配置开关和默认行为；
2. 输入、输出形状测试；
3. 单步反向传播测试；
4. 与当前基线使用相同划分和训练协议的消融；
5. 更新后的架构说明；
6. 对参数量与预测行为的影响。
