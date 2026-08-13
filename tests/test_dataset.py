from pathlib import Path

import cv2
import numpy as np
import pytest

from datasets.multimodal_dataset import MultimodalDataset


def write_sample(root: Path, sample_id="sample", depth_shape=(4, 8)):
    split = root / "train"
    for directory in ("rgb", "infrared", "depth", "labels"):
        (split / directory).mkdir(parents=True, exist_ok=True)

    rgb = np.full((4, 8, 3), 128, dtype=np.uint8)
    infrared = np.full((4, 8), 64, dtype=np.uint8)
    depth = np.full(depth_shape, 10_000, dtype=np.uint16)

    assert cv2.imwrite(str(split / "rgb" / f"{sample_id}.png"), rgb)
    assert cv2.imwrite(str(split / "infrared" / f"{sample_id}.png"), infrared)
    assert cv2.imwrite(str(split / "depth" / f"{sample_id}.png"), depth)
    (split / "labels" / f"{sample_id}.txt").write_text(
        "6 0.5 0.5 0.25 0.5\n",
        encoding="utf-8",
    )


def test_dataset_builds_five_channel_tensor(tmp_path):
    write_sample(tmp_path)
    dataset = MultimodalDataset(
        root_dir=str(tmp_path),
        split="train",
        img_size=(8, 8),
        augment=False,
    )

    sample = dataset[0]

    assert sample["sample_id"] == "sample"
    assert tuple(sample["images"].shape) == (5, 8, 8)
    assert tuple(sample["labels"].shape) == (1, 6)
    assert sample["labels"][0, 1].item() == 6
    assert sample["images"][3].max().item() == pytest.approx(64 / 255)
    assert sample["images"][4].max().item() == pytest.approx(0.5)


def test_dataset_rejects_spatially_unaligned_modalities(tmp_path):
    write_sample(tmp_path, depth_shape=(3, 8))
    dataset = MultimodalDataset(
        root_dir=str(tmp_path),
        split="train",
        img_size=(8, 8),
        augment=False,
    )

    with pytest.raises(ValueError, match="Unaligned modality sizes"):
        dataset[0]


def test_dataset_rejects_invalid_class_id(tmp_path):
    write_sample(tmp_path)
    label_path = tmp_path / "train" / "labels" / "sample.txt"
    label_path.write_text("12 0.5 0.5 0.25 0.5\n", encoding="utf-8")
    dataset = MultimodalDataset(
        root_dir=str(tmp_path),
        split="train",
        img_size=(8, 8),
        augment=False,
    )

    with pytest.raises(ValueError, match="Invalid class id"):
        dataset[0]


def test_dataset_depth_validity_mask_adds_channel(tmp_path):
    write_sample(tmp_path)
    depth = np.zeros((4, 8), dtype=np.uint16)
    depth[:, :4] = 10_000  # 左半有效，右半为无效深度 0
    assert cv2.imwrite(str(tmp_path / "train" / "depth" / "sample.png"), depth)

    dataset = MultimodalDataset(
        root_dir=str(tmp_path),
        split="train",
        img_size=(8, 8),
        augment=False,
        depth_validity_mask=True,
    )

    images = dataset[0]["images"]

    assert tuple(images.shape) == (6, 8, 8)
    mask = images[5]
    assert set(mask.unique().tolist()) == {0.0, 1.0}
    assert mask.sum() > 0
    # 无效深度与 letterbox 填充处掩码为 0，且归一化深度同为 0
    assert images[4][mask == 0].abs().max().item() == 0.0
    # 有效区域归一化深度为 10000/20000
    assert images[4][mask == 1].max().item() == pytest.approx(0.5)


def test_dataset_modality_dropout_zeroes_one_modality(tmp_path):
    write_sample(tmp_path)
    dataset = MultimodalDataset(
        root_dir=str(tmp_path),
        split="train",
        img_size=(8, 8),
        augment=True,
        augment_cfg={
            "fliplr": 0.0,
            "color_prob": 0.0,
            "ir_gamma_prob": 0.0,
            "modality_dropout": 1.0,
        },
    )

    images = dataset[0]["images"]

    ir_zeroed = images[3].abs().max().item() == 0.0
    depth_zeroed = images[4].abs().max().item() == 0.0
    assert ir_zeroed or depth_zeroed
    assert not (ir_zeroed and depth_zeroed)
    # RGB 永不参与 dropout
    assert images[:3].abs().max().item() > 0.0
