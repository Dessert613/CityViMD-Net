# 贡献指南

感谢对 CityViMD-Net 公开基线的改进。

## 提交前

1. 不要上传比赛数据、测试集预测、受限制权重、访问凭据或私有日志；
2. 新功能必须与公开基线定位一致，并同步更新相关文档；
3. 修复缺陷时应添加能够复现问题的测试；
4. 新增基准必须满足 `BENCHMARKS.md` 和 `docs/reproducibility.md` 的记录要求；
5. 确认贡献代码及依赖允许按本仓库许可证公开。

## 本地检查

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
pytest
python tools/smoke_test.py
python tools/package_submission.py
```

## Pull Request

Pull Request 应说明：

- 要解决的问题；
- 主要实现变化；
- 已执行的测试；
- 对模型形状、训练行为或推理输出的影响；
- 是否改变公开配置或基准条件。

请不要用无法公开验证的排行榜数字作为修改依据。
