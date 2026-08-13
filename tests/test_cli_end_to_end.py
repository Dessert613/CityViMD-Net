"""CLI 级端到端链路测试（合成数据，CPU）。

真实以子进程运行：make_folds → train.py（交叉验证折 + EMA）→ test.py
（多尺度 TTA + WBF + zip）→ validate_predictions，专抓单元测试覆盖不到的
接线问题（参数解析、checkpoint 读写、输出产物布局）。
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def run_cli(args):
    result = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"command failed: {' '.join(args)}\n"
        f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-3000:]}"
    )
    return result.stdout


def write_modalities(split_dir: Path, sample_id: str, with_label: bool):
    rgb = np.full((64, 64, 3), 120, dtype=np.uint8)
    rgb[16:48, 16:48] = 250
    infrared = np.full((64, 64), 60, dtype=np.uint8)
    infrared[16:48, 16:48] = 220
    depth = np.full((64, 64), 12_000, dtype=np.uint16)
    depth[16:48, 16:48] = 4_000
    assert cv2.imwrite(str(split_dir / "rgb" / f"{sample_id}.png"), rgb)
    assert cv2.imwrite(str(split_dir / "infrared" / f"{sample_id}.png"), infrared)
    assert cv2.imwrite(str(split_dir / "depth" / f"{sample_id}.png"), depth)
    if with_label:
        (split_dir / "labels" / f"{sample_id}.txt").write_text(
            "6 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )


def build_workspace(root: Path):
    train_dir = root / "train"
    test_dir = root / "test"
    for directory in ("rgb", "infrared", "depth", "labels"):
        (train_dir / directory).mkdir(parents=True, exist_ok=True)
    for directory in ("rgb", "infrared", "depth"):
        (test_dir / directory).mkdir(parents=True, exist_ok=True)
    for index in range(4):
        write_modalities(train_dir, f"train_{index}", with_label=True)
    for index in range(2):
        write_modalities(test_dir, f"test_{index}", with_label=False)


def build_config(root: Path) -> Path:
    with (ROOT / "configs" / "default.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    cfg["data"]["root"] = str(root)
    cfg["data"]["img_size"] = [64, 64]
    cfg["model"]["backbone"]["depth_multiple"] = 0.34
    cfg["model"]["backbone"]["width_multiple"] = 0.25
    cfg["train"].update({
        "epochs": 1,
        "batch_size": 2,
        "workers": 0,
        "amp": False,
        "use_ema": True,
        "val_interval": 1,
        "save_interval": 1,
    })
    cfg["test"]["batch_size"] = 2
    cfg["paths"]["output_dir"] = str(root / "runs_train")
    config_path = root / "config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.dump(cfg, file, allow_unicode=True)
    return config_path


def test_full_cli_pipeline(tmp_path):
    build_workspace(tmp_path)
    config_path = build_config(tmp_path)
    folds_path = tmp_path / "folds.json"

    stdout = run_cli([
        "tools/make_folds.py",
        "--config", str(config_path),
        "--num-folds", "2",
        "--output", str(folds_path),
    ])
    assert "FOLDS_OK" in stdout

    run_cli([
        "train.py",
        "--config", str(config_path),
        "--folds", str(folds_path),
        "--fold", "0",
    ])
    weights = tmp_path / "runs_train" / "weights" / "last.pt"
    assert weights.exists()
    assert (tmp_path / "runs_train" / "metrics.jsonl").exists()
    summary = json.loads(
        (tmp_path / "runs_train" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["epochs_completed"] == 1

    predictions_dir = tmp_path / "preds"
    run_cli([
        "test.py",
        "--config", str(config_path),
        "--weights", str(weights),
        "--input", str(tmp_path / "test"),
        "--output", str(predictions_dir),
        "--device", "cpu",
        "--zip",
        "--tta",
        "--tta-scales", "0.75,1.0",
    ])
    assert (predictions_dir / "test_0.txt").exists()
    assert (predictions_dir / "test_1.txt").exists()
    assert (tmp_path / "predictions.zip").exists() or (
        predictions_dir.parent / "predictions.zip"
    ).exists()

    stdout = run_cli([
        "tools/validate_predictions.py",
        "--images", str(tmp_path / "test" / "rgb"),
        "--predictions", str(predictions_dir),
    ])
    assert "PREDICTIONS_OK" in stdout
