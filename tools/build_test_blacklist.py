"""Hash test-set images into a blacklist enforced by the training dataloader.

比赛合规守卫：将测试集所有 PNG 的 SHA-256 写入黑名单文件；
配置 data.test_blacklist 指向该文件后，训练/验证数据加载器发现
任何测试集图像会立即报错终止（防止测试数据混入训练环节）。
"""

import argparse
import glob
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.multimodal_dataset import file_sha256


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirs", nargs="+", default=["data/test"],
        help="测试集目录（递归收集 *.png）",
    )
    parser.add_argument("--output", default="data/test_blacklist.json")
    return parser.parse_args(argv)


def collect_hashes(directories):
    hashes = set()
    file_count = 0
    for directory in directories:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        pattern = os.path.join(directory, "**", "*.png")
        for path in glob.glob(pattern, recursive=True):
            hashes.add(file_sha256(path))
            file_count += 1
    if file_count == 0:
        raise RuntimeError(f"No PNG files found under: {directories}")
    return sorted(hashes), file_count


def main(argv=None):
    args = parse_args(argv)
    hashes, file_count = collect_hashes(args.dirs)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    payload = {
        "sources": args.dirs,
        "files": file_count,
        "hashes": hashes,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(
        f"BLACKLIST_OK files={file_count} unique_hashes={len(hashes)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
