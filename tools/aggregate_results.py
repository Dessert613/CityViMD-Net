"""Aggregate sweep results: group runs by variant, report mean/std ranking.

扫描根目录下各运行目录的 summary.json（train.py 训练结束时写出），
将目录名中 `_seed<N>` 剥离后视为同一变体，输出按均值排序的
mean ± std 排名表。这是 Stage A 编码搜索与后续消融的决策看板。
"""

import argparse
import glob
import json
import os
import re
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np


SEED_PATTERN = re.compile(r"_seed\d+")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/experiments/stage_a",
                        help="包含各运行目录的根目录")
    parser.add_argument("--metric", default="best_map50_95")
    parser.add_argument("--output", default=None,
                        help="可选：聚合结果 JSON 输出路径")
    parser.add_argument("--min-seeds", type=int, default=1,
                        help="少于该种子数的变体标记为 INCOMPLETE")
    return parser.parse_args(argv)


def collect_runs(root, metric):
    """返回 {variant: [(seed_dir, value)]}。"""
    grouped = {}
    for summary_path in sorted(
        glob.glob(os.path.join(root, "*", "summary.json"))
    ):
        run_dir = os.path.basename(os.path.dirname(summary_path))
        with open(summary_path, encoding="utf-8") as file:
            summary = json.load(file)
        if metric not in summary:
            print(f"[aggregate] skip {run_dir}: missing metric '{metric}'")
            continue
        variant = SEED_PATTERN.sub("", run_dir)
        grouped.setdefault(variant, []).append((run_dir, float(summary[metric])))
    return grouped


def build_ranking(grouped, min_seeds=1):
    """返回按均值降序的 [{variant, n, mean, std, min, max, complete}]。"""
    ranking = []
    for variant, entries in grouped.items():
        values = np.array([value for _, value in entries], dtype=np.float64)
        ranking.append({
            "variant": variant,
            "n": len(values),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "complete": len(values) >= min_seeds,
            "runs": [name for name, _ in entries],
        })
    ranking.sort(key=lambda item: item["mean"], reverse=True)
    return ranking


def main(argv=None):
    args = parse_args(argv)
    grouped = collect_runs(args.root, args.metric)
    if not grouped:
        raise RuntimeError(f"No summary.json found under: {args.root}")

    ranking = build_ranking(grouped, args.min_seeds)

    print(f"{'rank':<5} {'variant':<40} {'n':>3} "
          f"{'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    for index, item in enumerate(ranking, start=1):
        flag = "" if item["complete"] else "  INCOMPLETE"
        print(
            f"{index:<5} {item['variant']:<40} {item['n']:>3} "
            f"{item['mean']:>8.4f} {item['std']:>8.4f} "
            f"{item['min']:>8.4f} {item['max']:>8.4f}{flag}"
        )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(
                {"metric": args.metric, "root": args.root, "ranking": ranking},
                file, ensure_ascii=False, indent=2,
            )
        print(f"written: {args.output}")
    print(f"AGGREGATE_OK variants={len(ranking)} metric={args.metric}")
    return ranking


if __name__ == "__main__":
    main()
