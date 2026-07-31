"""Validate competition TXT prediction files before packaging."""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Test RGB image directory")
    parser.add_argument("--predictions", required=True, help="Prediction TXT directory")
    parser.add_argument("--num-classes", type=int, default=12)
    parser.add_argument("--max-det", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    image_ids = {
        os.path.splitext(name)[0]
        for name in os.listdir(args.images)
        if name.lower().endswith(".png")
    }
    prediction_ids = {
        os.path.splitext(name)[0]
        for name in os.listdir(args.predictions)
        if name.lower().endswith(".txt")
    }
    missing = sorted(image_ids - prediction_ids)
    extra = sorted(prediction_ids - image_ids)
    if missing or extra:
        raise RuntimeError(
            f"Prediction file mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    total = 0
    for sample_id in sorted(image_ids):
        path = os.path.join(args.predictions, f"{sample_id}.txt")
        with open(path, encoding="utf-8") as file:
            rows = [line.strip() for line in file if line.strip()]
        if len(rows) > args.max_det:
            raise ValueError(f"{path} contains {len(rows)} detections")
        for line_number, row in enumerate(rows, start=1):
            fields = row.split()
            if len(fields) != 6:
                raise ValueError(f"{path}:{line_number} must contain 6 fields")
            class_id = int(fields[0])
            values = [float(value) for value in fields[1:]]
            if not 0 <= class_id < args.num_classes:
                raise ValueError(f"{path}:{line_number} invalid class {class_id}")
            if not all(0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"{path}:{line_number} value outside [0, 1]")
        total += len(rows)
    print(f"PREDICTIONS_OK files={len(image_ids)} detections={total}")


if __name__ == "__main__":
    main()
