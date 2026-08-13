import numpy as np
import pytest

from test import nms_per_class, postprocess, preprocess, xyxy_to_yolo


def aligned_images():
    return {
        "rgb": np.full((4, 8, 3), 255, dtype=np.uint8),
        "infrared": np.full((4, 8), 127, dtype=np.uint8),
        "depth": np.full((4, 8), 10_000, dtype=np.uint16),
    }


def test_preprocess_preserves_modalities_and_scale_information():
    tensor, scale_info = preprocess(aligned_images(), (8, 8))

    assert tuple(tensor.shape) == (5, 8, 8)
    assert scale_info == {
        "scale": 1.0,
        "pad_h": 2,
        "pad_w": 0,
        "orig_h": 4,
        "orig_w": 8,
    }
    assert tensor[:3].max().item() == pytest.approx(1.0)
    assert tensor[3].max().item() == pytest.approx(127 / 255)
    assert tensor[4].max().item() == pytest.approx(0.5)


def test_preprocess_rejects_unaligned_modalities():
    images = aligned_images()
    images["depth"] = np.zeros((3, 8), dtype=np.uint16)

    with pytest.raises(ValueError, match="not spatially aligned"):
        preprocess(images, (8, 8))


def test_postprocess_restores_original_coordinates():
    detections = np.array([[1.0, 2.0, 7.0, 6.0, 0.9, 6.0]], dtype=np.float32)
    scale_info = {
        "scale": 1.0,
        "pad_h": 2,
        "pad_w": 0,
        "orig_h": 4,
        "orig_w": 8,
    }

    restored = postprocess(detections, scale_info)
    np.testing.assert_allclose(restored[0, :4], [1.0, 0.0, 7.0, 4.0])

    yolo = xyxy_to_yolo(restored[:, :4], img_w=8, img_h=4)
    np.testing.assert_allclose(yolo[0], [0.5, 0.5, 0.75, 1.0])


def test_nms_is_class_aware():
    detections = np.array(
        [
            [0, 0, 10, 10, 0.9, 0],
            [0, 0, 10, 10, 0.8, 0],
            [0, 0, 10, 10, 0.7, 1],
        ],
        dtype=np.float32,
    )

    kept = nms_per_class(detections, iou_thres=0.5, max_det=10)

    assert len(kept) == 2
    assert set(kept[:, 5].astype(int)) == {0, 1}
