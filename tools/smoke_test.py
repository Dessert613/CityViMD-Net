"""CityViMD-Net forward/loss/backward smoke test."""

import copy
import os
import sys

import torch
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.model import build_model
from utils.loss import build_loss


def main():
    with open(os.path.join(ROOT, "configs", "default.yaml"), encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    cfg = copy.deepcopy(cfg)
    cfg["data"]["img_size"] = [64, 64]
    cfg["model"]["backbone"]["depth_multiple"] = 0.34
    cfg["model"]["backbone"]["width_multiple"] = 0.25
    cfg["model"]["neck"]["type"] = "pan"

    model = build_model(cfg).train()
    images = torch.rand(2, 5, 64, 64)
    targets = torch.tensor([
        [0, 0, 0.50, 0.50, 0.25, 0.30],
        [1, 6, 0.35, 0.40, 0.20, 0.20],
    ], dtype=torch.float32)

    predictions, modality_weights = model(images)
    loss_fn = build_loss(model, cfg)
    loss, loss_items = loss_fn(predictions, targets, img_size=(64, 64))
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss_items}")
    loss.backward()

    model.eval()
    detections = model.predict(images[:1], conf_thres=0.99, max_det=100)
    print("SMOKE_TEST_OK")
    print("prediction_shapes:", [tuple(item.shape) for item in predictions])
    print("modality_weight_shapes:", [tuple(item.shape) for item in modality_weights])
    print("loss:", loss_items)
    print("detections:", tuple(detections[0].shape))


if __name__ == "__main__":
    main()
