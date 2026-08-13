import numpy as np
import pytest

from test import _round_to_stride, preprocess, weighted_box_fusion


def aligned_images():
    return {
        "rgb": np.full((4, 8, 3), 255, dtype=np.uint8),
        "infrared": np.full((4, 8), 127, dtype=np.uint8),
        "depth": np.full((4, 8), 10_000, dtype=np.uint16),
    }


def test_round_to_stride():
    assert _round_to_stride(640) == 640
    assert _round_to_stride(640 * 0.75) == 480
    assert _round_to_stride(100) == 96
    assert _round_to_stride(10) == 32  # 不低于单个步长


def test_wbf_fuses_overlapping_boxes_and_rewards_consensus():
    detections = np.array([
        [100, 100, 200, 200, 0.8, 0],
        [104, 104, 204, 204, 0.6, 0],   # 与上一框 IoU≈0.85，应融合
        [400, 400, 500, 500, 0.9, 1],   # 孤立框，另一类别
    ], dtype=np.float64)

    fused = weighted_box_fusion(detections, iou_thres=0.55, num_views=2, max_det=100)

    assert len(fused) == 2
    cls0 = fused[fused[:, 5] == 0][0]
    weight_sum = 0.8 + 0.6
    np.testing.assert_allclose(
        cls0[:4],
        [
            (100 * 0.8 + 104 * 0.6) / weight_sum,
            (100 * 0.8 + 104 * 0.6) / weight_sum,
            (200 * 0.8 + 204 * 0.6) / weight_sum,
            (200 * 0.8 + 204 * 0.6) / weight_sum,
        ],
        rtol=1e-6,
    )
    # 双视角共识：score = mean(0.8, 0.6) × min(2,2)/2
    assert cls0[4] == pytest.approx(0.7)

    cls1 = fused[fused[:, 5] == 1][0]
    np.testing.assert_allclose(cls1[:4], [400, 400, 500, 500])
    # 仅单视角命中：score = 0.9 × 1/2
    assert cls1[4] == pytest.approx(0.45)


def test_wbf_respects_max_det_and_empty_input():
    empty = np.zeros((0, 6))
    assert len(weighted_box_fusion(empty, 0.55, num_views=2, max_det=10)) == 0

    detections = np.array([
        [0, 0, 10, 10, 0.9, 0],
        [100, 100, 110, 110, 0.8, 0],
        [200, 200, 210, 210, 0.7, 0],
    ], dtype=np.float64)
    fused = weighted_box_fusion(detections, 0.55, num_views=1, max_det=2)
    assert len(fused) == 2
    # 按融合后分数降序截断
    assert fused[0, 4] >= fused[1, 4]


def test_preprocess_applies_depth_and_ir_encodings():
    tensor, _ = preprocess(aligned_images(), (8, 8), depth_encoding="inverse")
    assert tensor[4].max().item() == pytest.approx(1000 / 11000, rel=1e-5)

    tensor, _ = preprocess(aligned_images(), (8, 8), depth_encoding="log")
    expected = np.log1p(10_000.0) / np.log1p(20_000.0)
    assert tensor[4].max().item() == pytest.approx(expected, rel=1e-5)


def test_preprocess_depth_validity_mask_channel():
    tensor, _ = preprocess(aligned_images(), (8, 8), depth_validity_mask=True)

    assert tuple(tensor.shape) == (6, 8, 8)
    mask = tensor[5]
    # 上下各 2 行是 letterbox 填充，深度 0 → 掩码 0
    assert mask[:2].max().item() == 0.0
    assert mask[-2:].max().item() == 0.0
    assert mask[2:6].min().item() == 1.0
