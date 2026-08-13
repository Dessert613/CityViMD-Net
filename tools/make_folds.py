"""Create stratified cross-validation folds for the training split.

按「样本最稀有类别」贪心分层：优先安置含稀有类的样本，使每折的
稀有类实例数尽量均衡；无标注样本按折大小均衡分配。
输出 JSON 供 train.py --folds/--fold 使用。
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/folds.json")
    return parser.parse_args(argv)


def collect_sample_classes(split_path):
    """返回 {sample_id: [instance class ids]}；无标签文件视为空列表。"""
    rgb_dir = os.path.join(split_path, "rgb")
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    sample_ids = sorted(
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(rgb_dir, "*.png"))
    )
    if not sample_ids:
        raise RuntimeError(f"No PNG samples found in: {rgb_dir}")

    sample_classes = {}
    for sample_id in sample_ids:
        label_path = os.path.join(split_path, "labels", f"{sample_id}.txt")
        classes = []
        if os.path.exists(label_path):
            with open(label_path, encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line:
                        classes.append(int(float(line.split()[0])))
        sample_classes[sample_id] = classes
    return sample_classes


def build_assignments(sample_classes, num_folds, seed):
    """稀有类优先的贪心分层折划分（确定性，由 seed 控制并列顺序）。"""
    totals = Counter()
    for classes in sample_classes.values():
        totals.update(classes)

    sample_ids = sorted(sample_classes)
    random.Random(seed).shuffle(sample_ids)

    def rarity(sample_id):
        classes = set(sample_classes[sample_id])
        if not classes:
            return float("inf")
        rare = min(classes, key=lambda cls: (totals[cls], cls))
        return totals[rare]

    # 稳定排序：稀有样本先安置，等稀有度时保留 seed 打乱后的顺序
    sample_ids.sort(key=rarity)

    fold_sizes = [0] * num_folds
    fold_instances = [Counter() for _ in range(num_folds)]
    assignments = {}
    for sample_id in sample_ids:
        classes = sample_classes[sample_id]
        class_set = set(classes)
        if class_set:
            rare = min(class_set, key=lambda cls: (totals[cls], cls))
            fold = min(
                range(num_folds),
                key=lambda f: (fold_instances[f][rare], fold_sizes[f], f),
            )
        else:
            fold = min(range(num_folds), key=lambda f: (fold_sizes[f], f))
        assignments[sample_id] = fold
        fold_sizes[fold] += 1
        fold_instances[fold].update(classes)
    return assignments, fold_sizes, fold_instances


def main(argv=None):
    args = parse_args(argv)
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    data_cfg = cfg["data"]
    split_dir = data_cfg.get(f"{args.split}_dir", args.split)
    split_path = os.path.join(data_cfg["root"], split_dir)

    sample_classes = collect_sample_classes(split_path)
    assignments, fold_sizes, fold_instances = build_assignments(
        sample_classes, args.num_folds, args.seed
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    payload = {
        "num_folds": args.num_folds,
        "seed": args.seed,
        "split": args.split,
        "assignments": assignments,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)

    class_ids = sorted({cls for c in sample_classes.values() for cls in c})
    print(f"fold sizes: {fold_sizes}")
    for cls in class_ids:
        row = [fold_instances[f][cls] for f in range(args.num_folds)]
        print(f"class {cls:>2} instances per fold: {row}")
    print(
        f"FOLDS_OK samples={len(assignments)} folds={args.num_folds} "
        f"seed={args.seed} output={args.output}"
    )


if __name__ == "__main__":
    main()
