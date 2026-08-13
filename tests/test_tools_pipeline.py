import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.multimodal_dataset import MultimodalDataset, collate_fn
from models.model import build_model
from tools.audit_dataset import audit_split
from tools.eval_robustness import (
    iterate_batches,
    modality_channel_slices,
    run_scenarios,
)
from tools.gen_stage_a_configs import DEPTH_VARIANTS, IR_VARIANTS
from tools.gen_stage_a_configs import main as gen_stage_a_main


ROOT = Path(__file__).resolve().parents[1]


def write_split(root: Path, labels_by_id, depth_zero_right_half=False):
    split = root / "train"
    for directory in ("rgb", "infrared", "depth", "labels"):
        (split / directory).mkdir(parents=True, exist_ok=True)
    for sample_id, label_lines in labels_by_id.items():
        rgb = np.full((4, 8, 3), 128, dtype=np.uint8)
        infrared = np.full((4, 8), 64, dtype=np.uint8)
        depth = np.full((4, 8), 10_000, dtype=np.uint16)
        if depth_zero_right_half:
            depth[:, 4:] = 0
        assert cv2.imwrite(str(split / "rgb" / f"{sample_id}.png"), rgb)
        assert cv2.imwrite(str(split / "infrared" / f"{sample_id}.png"), infrared)
        assert cv2.imwrite(str(split / "depth" / f"{sample_id}.png"), depth)
        (split / "labels" / f"{sample_id}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )


def test_audit_split_reports_classes_sizes_and_depth(tmp_path):
    write_split(
        tmp_path,
        {
            "a": ["0 0.5 0.5 0.25 0.5", "1 0.4 0.4 0.2 0.2"],
            "b": [],
        },
        depth_zero_right_half=True,
    )

    report = audit_split(
        str(tmp_path), "train", num_classes=2, class_names=["person", "boat"]
    )

    assert report["samples"] == 2
    assert report["boxes"]["total"] == 2
    assert report["boxes"]["images_without_boxes"] == 1
    assert report["classes"]["person"]["instances"] == 1
    assert report["classes"]["boat"]["instances"] == 1
    # 4x8 图上的 2x2 像素框属于 small 桶
    assert report["classes"]["person"]["size_buckets"]["small"] == 1
    assert report["image_sizes"] == {"4x8": 2}
    # 右半深度为 0
    assert report["depth"]["zero_ratio_mean"] == 0.5


def test_gen_stage_a_config_matrix(tmp_path):
    output = tmp_path / "stage_a"
    gen_stage_a_main([
        "--base", str(ROOT / "configs" / "default.yaml"),
        "--output", str(output),
        "--seeds", "1,2",
        "--epochs", "5",
    ])

    config_paths = sorted(glob.glob(str(output / "*.yaml")))
    assert len(config_paths) == len(DEPTH_VARIANTS) * len(IR_VARIANTS) * 2
    assert os.path.exists(output / "commands.sh")

    mask_config = next(path for path in config_paths if "-mask" in path)
    with open(mask_config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    assert cfg["data"]["depth_validity_mask"] is True
    assert cfg["model"]["in_channels"]["depth"] == 2
    assert cfg["train"]["epochs"] == 5

    plain_config = next(path for path in config_paths if "-mask" not in path)
    with open(plain_config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    assert cfg["model"]["in_channels"]["depth"] == 1


def test_modality_channel_slices_and_zeroing():
    cfg = {
        "data": {"modalities": ["rgb", "infrared", "depth"],
                 "depth_validity_mask": True},
        "model": {"in_channels": {"rgb": 3, "infrared": 1, "depth": 2}},
    }
    slices = modality_channel_slices(cfg)
    assert slices["rgb"] == slice(0, 3)
    assert slices["infrared"] == slice(3, 4)
    assert slices["depth"] == slice(4, 6)

    batch = {"images": torch.ones(2, 6, 8, 8), "labels": torch.zeros((0, 6))}
    zeroed = next(iter(iterate_batches([batch], slices["infrared"])))
    assert zeroed["images"][:, 3].abs().max().item() == 0.0
    assert zeroed["images"][:, 0].min().item() == 1.0
    assert zeroed["images"][:, 4].min().item() == 1.0
    # 原 batch 不被就地修改
    assert batch["images"][:, 3].min().item() == 1.0


def test_run_scenarios_tiny_end_to_end(tmp_path):
    write_split(tmp_path, {"a": ["6 0.5 0.5 0.25 0.5"]})

    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    cfg["data"]["img_size"] = [64, 64]
    cfg["data"]["root"] = str(tmp_path)
    cfg["model"]["backbone"]["depth_multiple"] = 0.34
    cfg["model"]["backbone"]["width_multiple"] = 0.25

    dataset = MultimodalDataset(
        root_dir=str(tmp_path), split="train", img_size=(64, 64), augment=False
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    model = build_model(cfg).eval()

    results = run_scenarios(model, dataloader, torch.device("cpu"), cfg)

    assert set(results) == {"full", "zero-infrared", "zero-depth"}
    for metrics in results.values():
        assert np.isfinite(metrics["map50_95"])
        assert 0.0 <= metrics["map50_95"] <= 1.0
