from pathlib import Path

import pytest
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


def test_forward_loss_backward_and_predict():
    cfg = tiny_config()
    model = build_model(cfg).train()
    images = torch.rand(2, 5, 64, 64)
    targets = torch.tensor(
        [
            [0, 0, 0.50, 0.50, 0.25, 0.30],
            [1, 6, 0.35, 0.40, 0.20, 0.20],
        ],
        dtype=torch.float32,
    )

    predictions = model(images)

    assert len(predictions) == 3
    assert [tuple(item.shape[-2:]) for item in predictions] == [
        (8, 8),
        (4, 4),
        (2, 2),
    ]
    expected_channels = cfg["data"]["num_classes"] + 4 * (
        cfg["model"]["head"]["reg_max"] + 1
    )
    assert all(item.shape[1] == expected_channels for item in predictions)

    loss, loss_items = build_loss(model, cfg)(
        predictions,
        targets,
        img_size=(64, 64),
    )
    assert torch.isfinite(loss)
    assert set(loss_items) == {"loss", "loss_box", "loss_cls", "loss_dfl"}
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    detections = model.predict(images[:1], conf_thres=0.99, max_det=10)
    assert len(detections) == 1
    assert detections[0].ndim == 2
    assert detections[0].shape[1] == 6
    assert len(detections[0]) <= 10


def test_forward_rejects_wrong_channel_count():
    model = build_model(tiny_config())

    try:
        model(torch.rand(1, 4, 64, 64))
    except ValueError as error:
        assert "Expected 5 early-fusion channels" in str(error)
    else:
        raise AssertionError("wrong channel count should fail")


def test_model_accepts_depth_validity_mask_six_channels():
    cfg = tiny_config()
    cfg["data"]["depth_validity_mask"] = True
    cfg["model"]["in_channels"]["depth"] = 2

    model = build_model(cfg).eval()
    predictions = model(torch.rand(1, 6, 64, 64))

    assert len(predictions) == 3


def test_model_rejects_mismatched_depth_channels():
    cfg = tiny_config()
    cfg["data"]["depth_validity_mask"] = True  # in_channels.depth 仍为 1

    with pytest.raises(ValueError, match="in_channels.depth"):
        build_model(cfg)
