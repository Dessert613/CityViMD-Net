"""Run the main local validation workflow end to end."""

import argparse
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable


def run(cmd):
    print(f"[validate_pipeline] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    parser.add_argument("--predictions-dir", default=None)
    parser.add_argument("--images-dir", default="data/test/rgb")
    parser.add_argument("--weights", default="runs/train/weights/best.pt")
    parser.add_argument("--test-input", default="data/test")
    parser.add_argument("--test-output", default="runs/test/predictions")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--conf-thres", default="0.001")
    parser.add_argument("--iou-thres", default="0.7")
    parser.add_argument("--max-det", default="100")
    return parser.parse_args()


def main():
    args = parse_args()
    ran_train = False
    ran_test = False
    ran_validate_predictions = False
    ran_package = False

    run([PYTHON, "tools/validate_dataset.py", "--config", args.config, "--split", args.split])
    if args.split != "val":
        run([PYTHON, "tools/validate_dataset.py", "--config", args.config, "--split", "val"])

    if not args.skip_smoke:
        run([PYTHON, "tools/smoke_test.py"])

    if args.train:
        print("[validate_pipeline] phase=train")
        run([PYTHON, "train.py", "--config", args.config])
        ran_train = True

    if args.test:
        print("[validate_pipeline] phase=test")
        test_cmd = [
            PYTHON,
            "test.py",
            "--config", args.config,
            "--weights", args.weights,
            "--input", args.test_input,
            "--output", args.test_output,
            "--conf-thres", args.conf_thres,
            "--iou-thres", args.iou_thres,
            "--max-det", args.max_det,
            "--zip",
        ]
        if args.tta:
            test_cmd.append("--tta")
        run(test_cmd)
        ran_test = True

    predictions_dir = args.predictions_dir or args.test_output
    if args.test and not args.skip_predictions and os.path.isdir(predictions_dir):
        print("[validate_pipeline] phase=validate_predictions")
        run([
            PYTHON,
            "tools/validate_predictions.py",
            "--images", args.images_dir,
            "--predictions", predictions_dir,
        ])
        ran_validate_predictions = True

    if args.package:
        print("[validate_pipeline] phase=package")
        run([PYTHON, "tools/package_submission.py"])
        ran_package = True

    if args.package:
        package_path = os.path.join(ROOT, "runs", "submission", "cityvimd_source.zip")
        if not os.path.exists(package_path):
            raise FileNotFoundError(package_path)
        print(f"[validate_pipeline] package={package_path}")

    if args.train:
        best_path = os.path.join(ROOT, "runs", "train", "weights", "best.pt")
        if os.path.exists(best_path):
            print(f"[validate_pipeline] best_weights={best_path}")

    summary_flags = " ".join([
        f"split={args.split}",
        f"train={'yes' if ran_train else 'no'}",
        f"test={'yes' if ran_test else 'no'}",
        f"check={'yes' if ran_validate_predictions else 'no'}",
        f"pack={'yes' if ran_package else 'no'}",
    ])
    summary_outputs = " ".join(
        part for part in [
            "train_output=runs/train" if args.train else "",
        f"test_output={predictions_dir}" if args.test else "",
            "package_output=runs/submission/cityvimd_source.zip" if args.package else "",
        ] if part
    )

    print(f"[validate_pipeline] summary: {summary_flags}")
    if summary_outputs:
        print(f"[validate_pipeline] outputs: {summary_outputs}")

    print("PIPELINE_OK")


if __name__ == "__main__":
    main()
