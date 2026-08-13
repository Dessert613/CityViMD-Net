"""「能学会」测试：验证 TAL 分配器 + 损失 + 反向传播能在合成样本上收敛。

训练循环存在隐性缺陷（分配器失配、损失符号、梯度断链等）时，
损失不会随过拟合显著下降；该测试在烧真实 GPU 时长之前拦住这类问题。
"""

from pathlib import Path

import torch
import yaml

from models.model import build_model
from utils.loss import build_loss


ROOT = Path(__file__).resolve().parents[1]


def tiny_config():
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    cfg["data"]["img_size"] = [64, 64]
    cfg["model"]["backbone"]["depth_multiple"] = 0.34
    cfg["model"]["backbone"]["width_multiple"] = 0.25
    return cfg


def synthetic_batch():
    """带明显目标结构的固定样本（batch=2，同图复制以稳定 BN）。"""
    images = torch.zeros(2, 5, 64, 64)
    images[:, :3, 16:48, 16:48] = 1.0   # RGB 亮块
    images[:, 3, 16:48, 16:48] = 0.8    # 红外热块
    images[:, 4, 16:48, 16:48] = 0.3    # 深度近块
    targets = torch.tensor([
        [0, 6, 0.5, 0.5, 0.5, 0.5],
        [1, 6, 0.5, 0.5, 0.5, 0.5],
    ], dtype=torch.float32)
    return images, targets


def test_loss_decreases_when_overfitting_single_batch():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = build_model(cfg).train()
    loss_fn = build_loss(model, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    images, targets = synthetic_batch()
    losses = []
    for _ in range(60):
        predictions = model(images)
        loss, loss_items = loss_fn(predictions, targets, img_size=(64, 64))
        assert torch.isfinite(loss), f"non-finite loss: {loss_items}"
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())

    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    # 过拟合单一样本，损失应显著下降
    assert final < initial * 0.6, (
        f"loss failed to converge: initial={initial:.3f}, final={final:.3f}"
    )
