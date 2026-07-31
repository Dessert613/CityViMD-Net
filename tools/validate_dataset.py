"""Validate modality alignment, image decoding and YOLO labels."""

import argparse
import os
import sys

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.multimodal_dataset import MultimodalDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    data_cfg = cfg["data"]
    split_dir = data_cfg.get(f"{args.split}_dir", args.split)
    dataset = MultimodalDataset(
        root_dir=data_cfg["root"],
        split=split_dir,
        img_size=tuple(data_cfg["img_size"]),
        num_classes=data_cfg["num_classes"],
        augment=False,
        modalities=data_cfg["modalities"],
        strict_modalities=True,
    )
    total_boxes = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample["images"].shape[0] != 5:
            raise RuntimeError(
                f"Unexpected channel count for {sample['sample_id']}: "
                f"{sample['images'].shape}"
            )
        total_boxes += len(sample["labels"])
    print(f"DATASET_OK split={args.split} samples={len(dataset)} boxes={total_boxes}")


if __name__ == "__main__":
    main()
