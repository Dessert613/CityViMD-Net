import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from datasets.multimodal_dataset import MultimodalDataset, load_fold_assignments
from tools.build_test_blacklist import main as blacklist_main
from tools.make_folds import build_assignments, main as make_folds_main


def write_split(root: Path, labels_by_id):
    split = root / "train"
    for directory in ("rgb", "infrared", "depth", "labels"):
        (split / directory).mkdir(parents=True, exist_ok=True)
    for index, (sample_id, label_lines) in enumerate(labels_by_id.items()):
        rgb = np.full((4, 8, 3), 100 + index, dtype=np.uint8)
        infrared = np.full((4, 8), 60 + index, dtype=np.uint8)
        depth = np.full((4, 8), 9_000 + index, dtype=np.uint16)
        assert cv2.imwrite(str(split / "rgb" / f"{sample_id}.png"), rgb)
        assert cv2.imwrite(str(split / "infrared" / f"{sample_id}.png"), infrared)
        assert cv2.imwrite(str(split / "depth" / f"{sample_id}.png"), depth)
        (split / "labels" / f"{sample_id}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )


def test_build_assignments_balanced_and_deterministic():
    sample_classes = {f"s{i:02d}": [i % 3] for i in range(20)}

    first, sizes, instances = build_assignments(sample_classes, 4, seed=42)
    second, _, _ = build_assignments(sample_classes, 4, seed=42)

    assert first == second
    assert sum(sizes) == 20
    assert max(sizes) - min(sizes) <= 1
    # 最稀有的类别（6 个实例）在 4 折间尽量均衡
    rare_counts = [instances[f][2] for f in range(4)]
    assert max(rare_counts) - min(rare_counts) <= 1


def test_make_folds_end_to_end_and_dataset_subset(tmp_path):
    labels = {
        "a": ["0 0.5 0.5 0.25 0.5"],
        "b": ["1 0.5 0.5 0.25 0.5"],
        "c": ["0 0.5 0.5 0.25 0.5", "1 0.4 0.4 0.2 0.2"],
        "d": [],
    }
    write_split(tmp_path, labels)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump({"data": {"root": str(tmp_path), "train_dir": "train"}}),
        encoding="utf-8",
    )
    folds_path = tmp_path / "folds.json"

    make_folds_main([
        "--config", str(config_path),
        "--num-folds", "2",
        "--seed", "7",
        "--output", str(folds_path),
    ])

    assignments = load_fold_assignments(str(folds_path))
    assert set(assignments) == set(labels)
    assert set(assignments.values()) <= {0, 1}

    fold0_ids = sorted(sid for sid, fold in assignments.items() if fold == 0)
    dataset = MultimodalDataset(
        root_dir=str(tmp_path),
        split="train",
        img_size=(8, 8),
        augment=False,
        sample_ids=fold0_ids,
    )
    assert len(dataset) == len(fold0_ids)
    assert sorted(dataset.sample_ids) == fold0_ids


def test_dataset_rejects_unknown_sample_ids(tmp_path):
    write_split(tmp_path, {"a": ["0 0.5 0.5 0.25 0.5"]})

    with pytest.raises(RuntimeError, match="not found in split"):
        MultimodalDataset(
            root_dir=str(tmp_path),
            split="train",
            img_size=(8, 8),
            augment=False,
            sample_ids=["missing"],
        )


def test_blacklist_guard_blocks_test_images(tmp_path):
    write_split(tmp_path, {"a": ["0 0.5 0.5 0.25 0.5"]})

    # 模拟测试集：包含与训练集完全相同的一张图像
    fake_test = tmp_path / "fake_test"
    fake_test.mkdir()
    shutil.copy(
        tmp_path / "train" / "rgb" / "a.png", fake_test / "leaked.png"
    )
    blacklist_path = tmp_path / "blacklist.json"
    blacklist_main([
        "--dirs", str(fake_test),
        "--output", str(blacklist_path),
    ])

    with pytest.raises(RuntimeError, match="COMPLIANCE VIOLATION"):
        MultimodalDataset(
            root_dir=str(tmp_path),
            split="train",
            img_size=(8, 8),
            augment=False,
            forbidden_hashes_path=str(blacklist_path),
        )


def test_blacklist_guard_passes_for_clean_data(tmp_path):
    write_split(tmp_path, {"a": ["0 0.5 0.5 0.25 0.5"]})

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    unrelated = np.full((6, 6, 3), 250, dtype=np.uint8)
    assert cv2.imwrite(str(other_dir / "clean.png"), unrelated)
    blacklist_path = tmp_path / "blacklist.json"
    blacklist_main([
        "--dirs", str(other_dir),
        "--output", str(blacklist_path),
    ])

    dataset = MultimodalDataset(
        root_dir=str(tmp_path),
        split="train",
        img_size=(8, 8),
        augment=False,
        forbidden_hashes_path=str(blacklist_path),
    )
    assert len(dataset) == 1
