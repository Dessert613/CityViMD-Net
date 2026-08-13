"""utils.metrics 与 pycocotools 官方协议的对拍测试。

官方评测为 COCO 式 mAP@50-95（10 档 IoU × 101 点插值 × 逐类平均）。
本文件用合成数据把自实现评测器与 pycocotools 对拍到 1e-4，
并用定向场景锁死两处协议细节：
1. 贪心匹配必须在「未占用」的 GT 中取最大 IoU；
2. 数据集中没有 GT 的类别不参与 mAP 平均。
"""

import contextlib
import io

import numpy as np
import pytest

pytest.importorskip("pycocotools")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.metrics import compute_map


IMG_SIZE = 640


def coco_eval_scores(predictions, targets, num_classes):
    """用 pycocotools 计算 (map50_95, map50, map75)。"""
    images = []
    annotations = []
    ann_id = 1
    for img_idx, gt in enumerate(targets):
        images.append({"id": img_idx + 1, "width": IMG_SIZE, "height": IMG_SIZE})
        for row in gt:
            cls, x1, y1, x2, y2 = row.tolist()
            width, height = x2 - x1, y2 - y1
            annotations.append({
                "id": ann_id,
                "image_id": img_idx + 1,
                "category_id": int(cls) + 1,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
            })
            ann_id += 1

    detections = []
    for img_idx, pred in enumerate(predictions):
        for row in pred:
            x1, y1, x2, y2, score, cls = row.tolist()
            detections.append({
                "image_id": img_idx + 1,
                "category_id": int(cls) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })
    assert detections, "parity scenarios must contain at least one detection"

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": cls + 1, "name": str(cls)} for cls in range(num_classes)
            ],
        }
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator.stats[0], evaluator.stats[1], evaluator.stats[2]


def random_scenario(seed, num_images=30, num_classes=6):
    """带扰动 TP、随机 FP 和漏检的可复现随机场景。"""
    rng = np.random.default_rng(seed)
    predictions, targets = [], []
    for _ in range(num_images):
        gt_rows = []
        for _ in range(int(rng.integers(0, 7))):
            cls = int(rng.integers(0, num_classes))
            x1 = rng.uniform(0, IMG_SIZE - 130)
            y1 = rng.uniform(0, IMG_SIZE - 130)
            width = rng.uniform(12, 120)
            height = rng.uniform(12, 120)
            gt_rows.append([cls, x1, y1, x1 + width, y1 + height])
        targets.append(np.array(gt_rows, dtype=np.float64).reshape(-1, 5))

        pred_rows = []
        for cls, x1, y1, x2, y2 in gt_rows:
            if rng.uniform() < 0.75:
                jitter = rng.uniform(-12, 12, size=4)
                bx1, by1, bx2, by2 = np.clip(
                    [x1 + jitter[0], y1 + jitter[1], x2 + jitter[2], y2 + jitter[3]],
                    0, IMG_SIZE,
                )
                if bx2 - bx1 > 2 and by2 - by1 > 2:
                    pred_rows.append(
                        [bx1, by1, bx2, by2, rng.uniform(0.05, 1.0), cls]
                    )
        for _ in range(int(rng.integers(0, 5))):
            cls = int(rng.integers(0, num_classes))
            x1 = rng.uniform(0, IMG_SIZE - 110)
            y1 = rng.uniform(0, IMG_SIZE - 110)
            width = rng.uniform(8, 100)
            height = rng.uniform(8, 100)
            pred_rows.append(
                [x1, y1, x1 + width, y1 + height, rng.uniform(0.05, 1.0), cls]
            )
        predictions.append(np.array(pred_rows, dtype=np.float64).reshape(-1, 6))
    return predictions, targets


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_map_matches_pycocotools_on_random_scenarios(seed):
    predictions, targets = random_scenario(seed)
    ours = compute_map(predictions, targets, num_classes=6)
    coco_map, coco_map50, coco_map75 = coco_eval_scores(predictions, targets, 6)

    assert ours["map50_95"] == pytest.approx(coco_map, abs=1e-4)
    assert ours["map50"] == pytest.approx(coco_map50, abs=1e-4)
    assert ours["map75"] == pytest.approx(coco_map75, abs=1e-4)


def test_second_detection_matches_remaining_overlapping_gt():
    """密集场景回归锁：高分预测占用 GT1 后，低分预测必须能匹配空闲的 GT2。"""
    targets = [np.array([
        [0, 100, 100, 200, 200],
        [0, 110, 110, 210, 210],
    ], dtype=np.float64)]
    predictions = [np.array([
        [101, 101, 201, 201, 0.9, 0],   # 与 GT1 的 IoU 最大，占用 GT1
        [104, 104, 204, 204, 0.8, 0],   # argmax 也是 GT1，但 GT2 空闲且 IoU>=0.5
    ], dtype=np.float64)]

    ours = compute_map(predictions, targets, num_classes=1)
    coco_map, coco_map50, coco_map75 = coco_eval_scores(predictions, targets, 1)

    assert ours["map50"] == pytest.approx(1.0)
    assert ours["map50_95"] == pytest.approx(coco_map, abs=1e-4)
    assert ours["map50"] == pytest.approx(coco_map50, abs=1e-4)
    assert ours["map75"] == pytest.approx(coco_map75, abs=1e-4)


def test_classes_without_gt_are_excluded_from_mean():
    """无 GT 类别口径锁：类别 1 只有误检没有 GT，不得拉低 mAP。"""
    targets = [np.array([[0, 50, 50, 150, 150]], dtype=np.float64)]
    predictions = [np.array([
        [50, 50, 150, 150, 0.9, 0],
        [300, 300, 400, 400, 0.8, 1],
    ], dtype=np.float64)]

    ours = compute_map(predictions, targets, num_classes=2)
    coco_map, coco_map50, _ = coco_eval_scores(predictions, targets, 2)

    assert ours["map50"] == pytest.approx(1.0)
    assert ours["map50"] == pytest.approx(coco_map50, abs=1e-4)
    assert ours["map50_95"] == pytest.approx(coco_map, abs=1e-4)
    # 展示用的逐类 AP 仍保留全长
    assert len(ours["ap_per_class_50"]) == 2
