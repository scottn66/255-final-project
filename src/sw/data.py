"""Read SpectralWaste hyperspectral cubes and their segmentation masks.

Layout under the dataset root (the ``SW_ROOT`` folder):

    hyper/{train,val,test}/<stem>.tiff
    labels_hyper_lt/{train,val,test}/<stem>.png

A sample is ``(cube, mask)``:

* cube: float32, shape ``(C, H, W)``, values in ``[0, 1]``
  (on disk the cube is uint16 and band-first; real files are 224 x 256 x 256)
* mask: int64, shape ``(H, W)``, class ids 0..6

Cubes and masks are paired by filename stem. There is no ``*semantic.png``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset

# Class ids match meta.json. Id 0 is background and is left out of mIoU.
CLASS_NAMES = (
    "background",
    "film",
    "basket",
    "cardboard",
    "video_tape",
    "filament",
    "bag",
)
NUM_CLASSES = len(CLASS_NAMES)

# RGB palette from meta.json, one colour per class id.
PALETTE = (
    (0, 0, 0),
    (218, 247, 6),
    (51, 221, 255),
    (52, 50, 221),
    (202, 152, 195),
    (0, 128, 0),
    (255, 165, 0),
)

# Train-split pixel counts for labels_hyper_lt (the hyperspectral masks).
# meta.json "categories_pixel_counts" are the RGB-label counts. Do not use
# those numbers to weight a hyperspectral loss.
HYPER_LT_TRAIN_PIXEL_COUNTS = (
    27612233,
    1819248,
    1060150,
    220623,
    397471,
    33101,
    2542678,
)

SPLIT_COUNTS = {"train": 514, "val": 167, "test": 171}

UINT16_MAX = 65535.0

SW_ROOT_HELP = (
    "SW_ROOT is not set. From the repo root run `bash scripts/get_data.sh`, "
    "then `export SW_ROOT` to the folder the script prints "
    "(it contains hyper/ and labels_hyper_lt/). Restart the kernel after exporting."
)


def class_weights(counts=HYPER_LT_TRAIN_PIXEL_COUNTS) -> torch.Tensor:
    """Inverse-frequency class weights with mean 1, for ``nn.CrossEntropyLoss``.

    Pass the returned tensor as ``weight=``. Rare classes (filament) get a
    larger weight. ``counts`` defaults to the hyperspectral train counts above.
    """
    freq = torch.tensor(list(counts), dtype=torch.float32)
    weights = 1.0 / freq
    return weights / weights.mean()


def resolve_root(root=None) -> Path:
    """Dataset folder. ``root`` wins; otherwise the ``SW_ROOT`` environment variable."""
    if root is None:
        root = os.environ.get("SW_ROOT")
    if not root:
        raise RuntimeError(SW_ROOT_HELP)
    path = Path(root).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {path}")
    return path


class SpectralWasteDataset(Dataset):
    """Hyperspectral cubes and ``labels_hyper_lt`` masks for one split.

    Args:
        root: folder that contains ``hyper/`` and ``labels_hyper_lt/``.
            Defaults to the ``SW_ROOT`` environment variable.
        split: ``"train"``, ``"val"``, or ``"test"``.
        bands: optional band indexes to keep, in that order. ``None`` keeps
            every band.
        random_flip: if True, flip each sample left-right with probability 1/2.
            The cube and the mask flip together.
        check_split_size: if True, raise when this split does not have the
            published number of pairs (train 514, val 167, test 171).
    """

    def __init__(
        self,
        root=None,
        split="train",
        bands=None,
        random_flip=False,
        check_split_size=False,
    ):
        if split not in SPLIT_COUNTS:
            raise ValueError(f"split must be one of {tuple(SPLIT_COUNTS)}, got {split!r}")
        self.root = resolve_root(root)
        self.split = split
        self.bands = None if bands is None else [int(b) for b in bands]
        self.random_flip = bool(random_flip)
        self.pairs = _pair_split(self.root, split)
        if check_split_size and len(self.pairs) != SPLIT_COUNTS[split]:
            raise ValueError(
                f"{split} has {len(self.pairs)} cube/mask pairs, "
                f"expected {SPLIT_COUNTS[split]}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        cube_path, mask_path = self.pairs[index]
        raw = tifffile.imread(cube_path)
        cube = np.asarray(raw, dtype=np.float32) / np.float32(UINT16_MAX)
        if self.bands is not None:
            cube = cube[self.bands]
        with Image.open(mask_path) as img:
            mask = np.array(img)
        if mask.ndim != 2:
            raise ValueError(f"{mask_path.name} has shape {mask.shape}; expected (H, W) class ids")
        cube_t = torch.from_numpy(np.ascontiguousarray(cube))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64)))
        if self.random_flip and torch.rand(()) < 0.5:
            cube_t = cube_t.flip(-1)
            mask_t = mask_t.flip(-1)
        return cube_t, mask_t


def _pair_split(root: Path, split: str):
    """Return ``(cube_path, mask_path)`` pairs sorted by stem."""
    hyper = root / "hyper" / split
    labels = root / "labels_hyper_lt" / split
    if not hyper.is_dir() or not labels.is_dir():
        raise FileNotFoundError(
            f"missing {hyper} or {labels}. "
            "SW_ROOT must be the folder that contains hyper/ and labels_hyper_lt/."
        )
    cubes = {path.stem: path for path in hyper.glob("*.tiff")}
    masks = {path.stem: path for path in labels.glob("*.png")}
    only_cubes = sorted(set(cubes) - set(masks))
    only_masks = sorted(set(masks) - set(cubes))
    if only_cubes or only_masks:
        raise ValueError(
            f"{split}: cube and mask stems do not match "
            f"({len(only_cubes)} cubes without a mask, "
            f"{len(only_masks)} masks without a cube). "
            "Pair files by stem: hyper/<split>/<stem>.tiff <-> "
            "labels_hyper_lt/<split>/<stem>.png."
        )
    return [(cubes[stem], masks[stem]) for stem in sorted(cubes)]
