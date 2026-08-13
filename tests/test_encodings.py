import numpy as np
import pytest

from datasets.multimodal_dataset import encode_depth, encode_infrared


DEPTH = np.array([[0.0, 10_000.0, 20_000.0]], dtype=np.float32)


def test_encode_depth_linear():
    np.testing.assert_allclose(encode_depth(DEPTH, "linear"), [[0.0, 0.5, 1.0]])


def test_encode_depth_inverse_keeps_invalid_zero():
    value = encode_depth(DEPTH, "inverse")
    np.testing.assert_allclose(
        value, [[0.0, 1000 / 11000, 1000 / 21000]], rtol=1e-6
    )


def test_encode_depth_log_keeps_invalid_zero():
    value = encode_depth(DEPTH, "log")
    expected = np.log1p(10_000.0) / np.log1p(20_000.0)
    np.testing.assert_allclose(value, [[0.0, expected, 1.0]], rtol=1e-6)


def test_encode_depth_minmax_per_image():
    value = encode_depth(DEPTH, "minmax")
    # 有效像素 [10000, 20000] 映射到 [0, 1]，无效像素保持 0
    np.testing.assert_allclose(value, [[0.0, 0.0, 1.0]])


def test_encode_depth_minmax_constant_image():
    depth = np.full((2, 2), 5000.0, dtype=np.float32)
    np.testing.assert_allclose(encode_depth(depth, "minmax"), np.zeros((2, 2)))


def test_encode_depth_rejects_unknown_encoding():
    with pytest.raises(ValueError, match="depth encoding"):
        encode_depth(DEPTH, "wavelet")


def test_encode_infrared_raw():
    image = np.array([[0, 127, 255]], dtype=np.uint8)
    np.testing.assert_allclose(
        encode_infrared(image, "raw"), [[0.0, 127 / 255, 1.0]]
    )


def test_encode_infrared_clahe_shape_and_range():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    value = encode_infrared(image, "clahe")
    assert value.shape == image.shape
    assert value.dtype == np.float32
    assert 0.0 <= value.min() <= value.max() <= 1.0


def test_encode_infrared_percentile_stretches_contrast():
    image = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    value = encode_infrared(image, "percentile")
    assert value.min() == pytest.approx(0.0)
    assert value.max() == pytest.approx(1.0)
    # 单调不减
    assert np.all(np.diff(value[0]) >= 0)


def test_encode_infrared_rejects_unknown_encoding():
    with pytest.raises(ValueError, match="infrared encoding"):
        encode_infrared(np.zeros((2, 2), dtype=np.uint8), "fourier")
