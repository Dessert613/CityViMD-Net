"""Audit the raw dataset before training decisions.

产出规划文档要求的数据审计报告：类别频次、目标尺寸分桶（COCO 口径）、
深度无效区域比例、红外亮度统计、图像尺寸分布。
结果写 JSON 并打印摘要；不依赖模型，拿到数据即可运行。
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml


SMALL_AREA = 32 ** 2
MEDIUM_AREA = 96 ** 2
DEPTH_MIN_VALID_MM = 300  # 官方量程下界约 30cm


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default=None,
                        help="默认 runs/audit/audit_<split>.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="只审计前 N 个样本（0 = 全量）")
    return parser.parse_args(argv)


def bucket_of(area_pixels):
    if area_pixels < SMALL_AREA:
        return "small"
    if area_pixels < MEDIUM_AREA:
        return "medium"
    return "large"


def audit_split(root_dir, split_dir, num_classes, class_names=None, limit=0):
    split_path = os.path.join(root_dir, split_dir)
    rgb_dir = os.path.join(split_path, "rgb")
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    sample_ids = sorted(
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(rgb_dir, "*.png"))
    )
    if limit:
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise RuntimeError(f"No PNG samples found in: {rgb_dir}")

    class_names = class_names or [f"class_{idx}" for idx in range(num_classes)]
    class_instances = [0] * num_classes
    class_images = [set() for _ in range(num_classes)]
    class_buckets = [
        {"small": 0, "medium": 0, "large": 0} for _ in range(num_classes)
    ]
    boxes_per_image = []
    image_sizes = {}
    depth_zero_ratios = []
    depth_below_min_ratios = []
    ir_means = []

    has_labels = os.path.isdir(os.path.join(split_path, "labels"))

    for sample_id in sample_ids:
        rgb = cv2.imread(os.path.join(rgb_dir, f"{sample_id}.png"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"Failed to decode RGB image: {sample_id}")
        height, width = rgb.shape[:2]
        image_sizes[f"{height}x{width}"] = image_sizes.get(f"{height}x{width}", 0) + 1

        box_count = 0
        if has_labels:
            label_path = os.path.join(split_path, "labels", f"{sample_id}.txt")
            if os.path.exists(label_path):
                with open(label_path, encoding="utf-8") as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        cls = int(float(parts[0]))
                        norm_w, norm_h = float(parts[3]), float(parts[4])
                        area = (norm_w * width) * (norm_h * height)
                        if 0 <= cls < num_classes:
                            class_instances[cls] += 1
                            class_images[cls].add(sample_id)
                            class_buckets[cls][bucket_of(area)] += 1
                        box_count += 1
        boxes_per_image.append(box_count)

        depth_path = os.path.join(split_path, "depth", f"{sample_id}.png")
        if os.path.exists(depth_path):
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is not None:
                if depth.ndim == 3:
                    depth = depth[..., 0]
                total = depth.size
                depth_zero_ratios.append(float((depth == 0).sum()) / total)
                depth_below_min_ratios.append(
                    float((depth < DEPTH_MIN_VALID_MM).sum()) / total
                )

        ir_path = os.path.join(split_path, "infrared", f"{sample_id}.png")
        if os.path.exists(ir_path):
            ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
            if ir is not None:
                ir_means.append(float(ir.mean()))

    boxes = np.array(boxes_per_image)
    report = {
        "split": split_dir,
        "samples": len(sample_ids),
        "image_sizes": image_sizes,
        "boxes": {
            "total": int(boxes.sum()),
            "per_image_mean": float(boxes.mean()),
            "per_image_p50": float(np.percentile(boxes, 50)),
            "per_image_max": int(boxes.max()),
            "images_without_boxes": int((boxes == 0).sum()),
        },
        "classes": {
            class_names[idx]: {
                "id": idx,
                "instances": class_instances[idx],
                "images": len(class_images[idx]),
                "size_buckets": class_buckets[idx],
            }
            for idx in range(num_classes)
        },
        "depth": {
            "zero_ratio_mean": float(np.mean(depth_zero_ratios)) if depth_zero_ratios else None,
            "zero_ratio_p90": float(np.percentile(depth_zero_ratios, 90)) if depth_zero_ratios else None,
            "below_min_ratio_mean": float(np.mean(depth_below_min_ratios)) if depth_below_min_ratios else None,
            "min_valid_mm": DEPTH_MIN_VALID_MM,
        },
        "infrared": {
            "brightness_mean": float(np.mean(ir_means)) if ir_means else None,
            "brightness_std": float(np.std(ir_means)) if ir_means else None,
        },
    }
    return report


def main(argv=None):
    args = parse_args(argv)
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    data_cfg = cfg["data"]
    split_dir = data_cfg.get(f"{args.split}_dir", args.split)
    report = audit_split(
        data_cfg["root"],
        split_dir,
        data_cfg["num_classes"],
        class_names=data_cfg.get("class_names"),
        limit=args.limit,
    )

    output = args.output or os.path.join("runs", "audit", f"audit_{args.split}.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"samples={report['samples']} boxes={report['boxes']['total']}")
    ranked = sorted(
        report["classes"].items(), key=lambda item: item[1]["instances"]
    )
    for name, item in ranked:
        buckets = item["size_buckets"]
        print(
            f"  {name:<12} instances={item['instances']:>5} "
            f"images={item['images']:>5} "
            f"S/M/L={buckets['small']}/{buckets['medium']}/{buckets['large']}"
        )
    if report["depth"]["zero_ratio_mean"] is not None:
        print(
            f"depth zero_ratio mean={report['depth']['zero_ratio_mean']:.4f} "
            f"p90={report['depth']['zero_ratio_p90']:.4f}"
        )
    print(f"AUDIT_OK output={output}")


if __name__ == "__main__":
    main()
