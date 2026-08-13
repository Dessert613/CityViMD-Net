# 复现规范

本规范用于区分“代码能够运行”和“实验结果能够复核”。只有满足记录要求的结果
才应加入 `BENCHMARKS.md`。

## 最小复现信息

每次正式实验应保存：

- Git 提交哈希和工作区是否存在未提交修改；
- 完整 YAML 配置；
- 数据划分标识、样本数量和数据版本校验值；
- Python、PyTorch 和关键依赖版本；
- 随机种子；
- 启动命令；
- 每轮损失、学习率和验证指标；
- 最佳轮次及停止原因；
- 推理阈值、最大检测数和 TTA 状态。

数据本身、比赛凭据和受限制的权重不应提交到公开仓库。数据校验值应使用不会泄露
样本内容的清单摘要，并确认赛事规则允许公开。

## 推荐流程

### 1. 记录代码状态

```bash
git rev-parse HEAD
git status --short
```

存在未提交修改时，实验记录必须额外保存补丁；否则提交哈希无法唯一确定代码。

### 2. 建立隔离环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip freeze > runs/environment.txt
```

`runs/` 默认不会进入 Git。若要公开环境清单，应先检查其中是否包含本地路径或
私有包地址。

### 3. 验证代码与数据

```bash
python tools/smoke_test.py
python tools/validate_dataset.py --split train
python tools/validate_dataset.py --split val
```

### 4. 启动训练

```bash
python train.py --config configs/default.yaml
```

训练脚本会把生效配置写入 `runs/train/config.yaml`。命令行覆盖参数也必须记录。

### 5. 固定评估条件

比较实验时应保持以下条件一致：

- 训练集与验证集划分；
- 输入预处理与训练配置；
- 优化器、学习率策略和 warmup；
- 随机种子；
- 置信度、NMS IoU、`max_det` 和 TTA；
- mAP 实现版本。

如果只能运行一个随机种子，应明确标注结果不能代表方差。条件允许时至少运行三个
种子并报告均值和标准差。

## 公开结果门禁

一项结果只有在以下条件全部满足时才能发布：

- 对应代码和配置仍可取得；
- 原始日志无 NaN、Inf 或中途人为改写；
- 指标可由保存的预测或权重复算；
- 文档中的模型描述与实际配置一致；
- 数据与权重的公开方式符合许可和比赛规则；
- 不包含测试集人工标注、平台凭据或受限制材料。

无法满足门禁的数字可以保留在团队私有笔记中，但不应写入公开基准表。
