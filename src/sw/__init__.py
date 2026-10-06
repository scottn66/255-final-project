"""Small helpers for the SpectralWaste segmentation project.

Import the dataset, the foreground mIoU meter, or the attention blocks:

    from sw import SpectralWasteDataset, class_weights, ConfusionMeter
"""

from sw.attention import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
    ResidualCrossAttentionBlock,
    ResidualSelfAttentionBlock,
)
from sw.data import (
    CLASS_NAMES,
    HYPER_LT_TRAIN_PIXEL_COUNTS,
    NUM_CLASSES,
    PALETTE,
    SPLIT_COUNTS,
    SpectralWasteDataset,
    class_weights,
)
from sw.metrics import ConfusionMeter, fg_miou, per_class_iou
from sw.pe import IndexBandPE, sinusoidal_wavelength_pe

__all__ = [
    "CLASS_NAMES",
    "HYPER_LT_TRAIN_PIXEL_COUNTS",
    "NUM_CLASSES",
    "PALETTE",
    "SPLIT_COUNTS",
    "SpectralWasteDataset",
    "class_weights",
    "ConfusionMeter",
    "fg_miou",
    "per_class_iou",
    "MultiHeadSelfAttention",
    "MultiHeadCrossAttention",
    "ResidualSelfAttentionBlock",
    "ResidualCrossAttentionBlock",
    "sinusoidal_wavelength_pe",
    "IndexBandPE",
]
