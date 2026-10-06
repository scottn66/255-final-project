"""Foreground mIoU: background is not part of the six-class mean."""

from __future__ import annotations

import numpy as np
import torch

from sw.metrics import ConfusionMeter, fg_miou


def test_perfect_prediction_scores_one():
    target = np.arange(7)
    score, iou = fg_miou(target, target)
    assert score == 1.0
    assert np.allclose(iou, 1.0)


def test_background_perfect_but_foreground_wrong_scores_zero():
    """Every background pixel is right, every foreground pixel is wrong.

    Background IoU is 1. The reported mIoU is the mean of classes 1..6, so it is 0.
    A mean over all 7 classes would be 1/7 and is the wrong number.
    """
    target = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    pred = np.array([0, 0, 0, 2, 2, 1, 1, 4, 4, 3, 3, 6, 6, 5, 5])
    score, iou = fg_miou(pred, target)
    assert iou[0] == 1.0
    assert np.all(iou[1:] == 0.0)
    assert score == 0.0
    seven_class_mean = float(np.mean(iou))
    assert abs(seven_class_mean - (1.0 / 7.0)) < 1e-12


def test_known_partial_overlap():
    # One class-1 pixel is predicted as class 2.
    # IoU_1 = 1/2, IoU_2 = 2/3, IoU_3..6 = 1 -> mean = 31/36.
    target = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    pred = target.copy()
    pred[1] = 2
    score, iou = fg_miou(pred, target)
    assert abs(iou[1] - 0.5) < 1e-12
    assert abs(iou[2] - (2.0 / 3.0)) < 1e-12
    assert abs(score - (31.0 / 36.0)) < 1e-12


def test_meter_accumulates_batches_including_logits():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 7, size=(4, 8, 8))
    pred = rng.integers(0, 7, size=(4, 8, 8))
    whole, whole_iou = fg_miou(pred, target)

    meter = ConfusionMeter()
    meter.update(pred[:2], target[:2])
    meter.update(torch.as_tensor(pred[2:]), torch.as_tensor(target[2:]))
    assert meter.fg_miou() == whole
    assert np.allclose(meter.per_class_iou(), whole_iou, equal_nan=True)

    logits = torch.zeros(4, 7, 8, 8)
    for cls in range(7):
        logits[:, cls] = torch.as_tensor(pred == cls, dtype=torch.float32)
    from_logits, _ = fg_miou(logits, target)
    assert from_logits == whole
