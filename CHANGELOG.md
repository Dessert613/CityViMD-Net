# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 的基本格式。
版本号在创建正式 Release 时确定。

## Unreleased

### Added

- MIT 许可证和引用信息；
- 架构边界、复现要求与基准记录规范；
- 自动化测试与持续集成；
- 开发依赖和源码包内容测试；
- 深度有效性掩码配置项 `data.depth_validity_mask`（区分无效深度与极近距离，
  默认关闭）；
- 模态 dropout 增强 `train.augment.modality_dropout`（训练时随机置零红外或
  深度，默认关闭）；
- 评测器与 pycocotools 的对拍测试（随机场景 + 密集匹配与类平均口径回归锁）；
- 深度编码 `data.depth_encoding`（linear/inverse/log/minmax）与红外编码
  `data.ir_encoding`（raw/clahe/percentile），训练与推理共享同一实现；
- 分层交叉验证：`tools/make_folds.py` 生成折划分，`train.py --folds --fold`
  按折训练与验证；
- 数据审计工具 `tools/audit_dataset.py`（类别频次、尺寸分桶、深度无效比例）；
- 合规守卫：`tools/build_test_blacklist.py` 生成测试集哈希黑名单，
  训练/验证加载器发现测试集图像立即报错；
- 模态鲁棒性评测 `tools/eval_robustness.py`（完整输入 vs 置零红外/深度）；
- 多尺度 TTA（`test.py --tta-scales`）与视角间加权框融合（WBF），
  仍为单模型多次前向；
- Stage A 实验矩阵生成器 `tools/gen_stage_a_configs.py`；
- 训练指标落盘（`metrics.jsonl` / `summary.json`）与多种子结果汇总工具
  `tools/aggregate_results.py`；
- 合成数据过拟合收敛测试与 CLI 级端到端链路测试
  （make_folds → train → test → validate_predictions）。

### Fixed

- mAP 计算与 COCO 协议对齐：贪心匹配改为在未占用的 GT 中取最大 IoU；
  数据集中无 GT 的类别不再以 AP=0 计入平均；
- DataLoader 多 worker 下 Python/NumPy 随机状态重复的问题
  （补充 `worker_init_fn`，保证各 worker 增强序列独立且可复现）。

### Changed

- 将仓库明确定位为可复现的五通道早期融合公开基线；
- 移除无法由公开产物验证的排行榜成绩；
- 文档不再暗示已实现动态模态加权或跨模态注意力；
- 源码包改为包含基准记录和复现文档；
- 移除面向开源库的许可证、贡献指南、引用文件与 README 展示。
