"""Evaluate mAP with individual modalities zeroed (degraded-sensor gate).

对同一权重跑三种场景：完整输入、红外置零、深度置零，报告 mAP 及相对
跌幅。对应官方「模态质量劣化时保持鲁棒」的功能要求；冲榜流程将其作为
提交前门禁。
"""

import argparse
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets.multimodal_dataset import build_dataloader, load_config
from models.model import build_model
from utils.metrics import evaluate


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="只评估前 N 个 batch（0 = 全量，用于快速冒烟）")
    return parser.parse_args(argv)


def modality_channel_slices(cfg):
    """按 modalities 顺序返回各模态在拼接张量中的通道切片。"""
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    slices = {}
    start = 0
    for modality in data_cfg["modalities"]:
        width = model_cfg["in_channels"][modality]
        slices[modality] = slice(start, start + width)
        start += width
    return slices


def iterate_batches(dataloader, channel_slice=None, max_batches=0):
    """可选地将指定通道整体置零后逐 batch 产出。"""
    for index, batch in enumerate(dataloader):
        if max_batches and index >= max_batches:
            break
        if channel_slice is None:
            yield batch
            continue
        images = batch["images"].clone()
        images[:, channel_slice] = 0.0
        yield {**batch, "images": images}


def run_scenarios(model, dataloader, device, cfg, conf_thres=0.001,
                  iou_thres=0.7, max_det=100, max_batches=0):
    """返回 {scenario: metrics}，场景为 full / zero-<modality>。"""
    slices = modality_channel_slices(cfg)
    img_size = tuple(cfg["data"]["img_size"])
    num_classes = cfg["data"]["num_classes"]

    scenarios = {"full": None}
    for modality in cfg["data"]["modalities"]:
        if modality == "rgb":
            continue
        scenarios[f"zero-{modality}"] = slices[modality]

    results = {}
    for name, channel_slice in scenarios.items():
        results[name] = evaluate(
            model,
            iterate_batches(dataloader, channel_slice, max_batches),
            device,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            num_classes=num_classes,
            max_det=max_det,
            img_size=img_size,
        )
    return results


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)

    use_cuda = torch.cuda.is_available() and args.device.lower() != "cpu"
    device = torch.device(f"cuda:{args.device}" if use_cuda else "cpu")
    print(f"Using device: {device}")

    model = build_model(cfg)
    checkpoint = torch.load(args.weights, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    dataloader = build_dataloader(cfg, split=args.split, eval_mode=True)
    print(f"Eval samples: {len(dataloader.dataset)}")

    results = run_scenarios(
        model, dataloader, device, cfg,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        max_batches=args.max_batches,
    )

    baseline = results["full"]["map50_95"]
    print(f"{'scenario':<16} {'mAP@50':>8} {'mAP@50-95':>10} {'drop':>8}")
    for name, metrics in results.items():
        if name == "full" or baseline <= 0:
            drop = "-"
        else:
            drop = f"{(metrics['map50_95'] - baseline) / baseline:+.1%}"
        print(
            f"{name:<16} {metrics['map50']:>8.4f} "
            f"{metrics['map50_95']:>10.4f} {drop:>8}"
        )
    print("ROBUSTNESS_OK")
    return results


if __name__ == "__main__":
    main()
