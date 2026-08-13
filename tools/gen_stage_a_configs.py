"""Generate the Stage A encoding-search config matrix.

规划文档 Stage A：深度编码 6 变体 × 红外编码 3 变体 × 3 种子 = 54 个
短程实验配置。输出 YAML 与启动命令清单；配置写入 runs/（不进 Git）。
"""

import argparse
import copy
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml


# (depth_encoding, depth_validity_mask)
DEPTH_VARIANTS = [
    ("linear", False),
    ("inverse", False),
    ("log", False),
    ("minmax", False),
    ("inverse", True),
    ("log", True),
]
IR_VARIANTS = ["raw", "clahe", "percentile"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="configs/default.yaml")
    parser.add_argument("--output", default="runs/experiments/stage_a")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=60,
                        help="短程实验轮数（写入生成的配置）")
    parser.add_argument("--folds", default="data/folds.json",
                        help="写入命令清单的折文件路径")
    parser.add_argument("--fold", type=int, default=0,
                        help="Stage A 统一使用的验证折")
    return parser.parse_args(argv)


def variant_name(depth_encoding, mask, ir_encoding, seed):
    mask_tag = "-mask" if mask else ""
    return f"d-{depth_encoding}{mask_tag}_ir-{ir_encoding}_seed{seed}"


def build_variant(base_cfg, depth_encoding, mask, ir_encoding, seed,
                  epochs, output_dir):
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["depth_encoding"] = depth_encoding
    cfg["data"]["depth_validity_mask"] = mask
    cfg["model"]["in_channels"]["depth"] = 2 if mask else 1
    cfg["data"]["ir_encoding"] = ir_encoding
    cfg["device"]["seed"] = seed
    cfg["train"]["epochs"] = epochs
    cfg["paths"]["output_dir"] = output_dir
    return cfg


def main(argv=None):
    args = parse_args(argv)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]

    with open(args.base, encoding="utf-8") as file:
        base_cfg = yaml.safe_load(file)

    os.makedirs(args.output, exist_ok=True)
    commands = []
    count = 0
    for depth_encoding, mask in DEPTH_VARIANTS:
        for ir_encoding in IR_VARIANTS:
            for seed in seeds:
                name = variant_name(depth_encoding, mask, ir_encoding, seed)
                run_dir = os.path.join(args.output, name)
                cfg = build_variant(
                    base_cfg, depth_encoding, mask, ir_encoding, seed,
                    args.epochs, run_dir,
                )
                config_path = os.path.join(args.output, f"{name}.yaml")
                with open(config_path, "w", encoding="utf-8") as file:
                    yaml.dump(cfg, file, allow_unicode=True, sort_keys=False)
                commands.append(
                    f"python train.py --config {config_path} "
                    f"--folds {args.folds} --fold {args.fold}"
                )
                count += 1

    commands_path = os.path.join(args.output, "commands.sh")
    with open(commands_path, "w", encoding="utf-8") as file:
        file.write("#!/bin/sh\n# Stage A 编码搜索启动命令（可整体交给调度器）\n")
        file.write("\n".join(commands) + "\n")

    print(f"STAGE_A_CONFIGS_OK count={count} dir={args.output}")
    print(f"commands: {commands_path}")


if __name__ == "__main__":
    main()
