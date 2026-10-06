"""Foreground mIoU for SpectralWaste.

The score is the mean IoU of classes 1..6. Class 0 (background) is not in
that mean. A class with no pixels in the target and none in the prediction
is skipped (its IoU is NaN). On a full split every class shows up, so this
matches a plain mean of the six foreground IoUs.

``ConfusionMeter`` adds batches into one matrix. Call ``update`` as you loop,
then ``fg_miou`` once at the end.
"""

from __future__ import annotations

import numpy as np
import torch

NUM_CLASSES = 7
BACKGROUND = 0


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def as_class_map(pred, num_classes=NUM_CLASSES):
    """Class ids from a label map or from logits shaped ``(..., num_classes, H, W)``."""
    if torch.is_tensor(pred):
        if pred.ndim >= 3 and pred.shape[-3] == num_classes and pred.dtype.is_floating_point:
            pred = pred.argmax(dim=-3)
        pred = pred.detach().cpu().numpy()
    else:
        pred = np.asarray(pred)
        if (
            pred.ndim >= 3
            and pred.shape[-3] == num_classes
            and np.issubdtype(pred.dtype, np.floating)
        ):
            pred = pred.argmax(axis=-3)
    return pred


def confusion_matrix(pred, target, num_classes=NUM_CLASSES):
    """``cm[t, p]`` counts pixels whose true class is t and predicted class is p."""
    pred = as_class_map(pred, num_classes).reshape(-1).astype(np.int64, copy=False)
    target = _to_numpy(target).reshape(-1).astype(np.int64, copy=False)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {target.shape}")
    if pred.size and (
        target.min() < 0
        or target.max() >= num_classes
        or pred.min() < 0
        or pred.max() >= num_classes
    ):
        raise ValueError(f"label out of range [0, {num_classes})")
    flat = target * num_classes + pred
    return np.bincount(flat, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def per_class_iou(cm):
    """IoU per class. A class with union 0 (absent on both sides) is NaN."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, tp / union, np.nan)


def fg_miou_from_cm(cm, background=BACKGROUND):
    """Return ``(foreground_miou, per_class_iou)`` from a confusion matrix."""
    iou = per_class_iou(cm)
    foreground = np.delete(iou, background)
    if not np.any(~np.isnan(foreground)):
        score = float("nan")
    else:
        score = float(np.nanmean(foreground))
    return score, iou


def fg_miou(pred, target, num_classes=NUM_CLASSES, background=BACKGROUND):
    """Foreground mIoU of one batch or one image.

    Returns ``(score, per_class_iou)`` with ``per_class_iou`` length ``num_classes``.
    For a whole split, prefer ``ConfusionMeter`` so every image shares one matrix.
    """
    return fg_miou_from_cm(confusion_matrix(pred, target, num_classes), background)


class ConfusionMeter:
    """Running confusion matrix. ``update`` with batches, then read ``fg_miou``."""

    def __init__(self, num_classes=NUM_CLASSES, background=BACKGROUND):
        self.num_classes = num_classes
        self.background = background
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        self.cm += confusion_matrix(pred, target, self.num_classes)

    def per_class_iou(self):
        return per_class_iou(self.cm)

    def fg_miou(self):
        return fg_miou_from_cm(self.cm, self.background)[0]

    def reset(self):
        self.cm[:] = 0
