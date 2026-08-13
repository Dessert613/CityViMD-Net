import numpy as np
import pytest

from utils.metrics import compute_ap, compute_map


def test_compute_ap_for_perfect_precision_recall():
    ap, recall_points, precision = compute_ap(
        np.array([1.0]),
        np.array([1.0]),
    )

    assert ap == pytest.approx(1.0)
    assert len(recall_points) == 101
    np.testing.assert_allclose(precision, np.ones(101))


def test_compute_map_for_perfect_detection():
    predictions = [
        np.array([[10, 10, 30, 30, 0.9, 0]], dtype=np.float32),
    ]
    targets = [
        np.array([[0, 10, 10, 30, 30]], dtype=np.float32),
    ]

    results = compute_map(predictions, targets, num_classes=1)

    assert results["map50"] == pytest.approx(1.0)
    assert results["map75"] == pytest.approx(1.0)
    assert results["map50_95"] == pytest.approx(1.0)
    assert results["ap_per_class_50"] == pytest.approx([1.0])


def test_compute_map_penalizes_missing_detection():
    results = compute_map(
        [np.zeros((0, 6), dtype=np.float32)],
        [np.array([[0, 10, 10, 30, 30]], dtype=np.float32)],
        num_classes=1,
    )

    assert results["map50"] == 0.0
    assert results["map75"] == 0.0
    assert results["map50_95"] == 0.0
