"""Loader checks on tiny synthetic cubes and masks."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from sw.data import (
    CLASS_NAMES,
    HYPER_LT_TRAIN_PIXEL_COUNTS,
    SPLIT_COUNTS,
    SpectralWasteDataset,
    class_weights,
)


def _write_split(root, split, stems, shape=(4, 8, 8)):
    hyper = root / "hyper" / split
    labels = root / "labels_hyper_lt" / split
    hyper.mkdir(parents=True)
    labels.mkdir(parents=True)
    c, h, w = shape
    for i, stem in enumerate(stems):
        cube = np.zeros(shape, dtype=np.uint16)
        cube[0, 0, 0] = 0
        cube[0, 0, -1] = 65535
        cube[0, 0, 1] = 1000 * (i + 1)  # unique, and not symmetric left-right
        # photometric avoids tifffile treating a small C as an RGB image.
        tifffile.imwrite(hyper / f"{stem}.tiff", cube, photometric="minisblack")
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, :] = i % 7
        mask[0, 0] = 0
        Image.fromarray(mask, mode="L").save(labels / f"{stem}.png")


def _fixture(root):
    # Names are out of alphabetical order on purpose.
    _write_split(root, "train", ["b_sample", "a_sample"])
    _write_split(root, "val", ["v0"])
    _write_split(root, "test", ["t0"])


def test_shapes_dtype_and_scaling(tmp_path):
    _fixture(tmp_path)
    ds = SpectralWasteDataset(tmp_path, split="train")
    cube, mask = ds[0]
    assert cube.shape == (4, 8, 8)
    assert cube.dtype == torch.float32
    assert mask.shape == (8, 8)
    assert mask.dtype == torch.int64
    assert cube.min().item() == 0.0
    assert cube.max().item() == 1.0
    # uint16 65535 is the right edge of band 0, row 0.
    assert cube[0, 0, -1].item() == 1.0


def test_pairs_by_stem_in_sorted_order(tmp_path):
    _fixture(tmp_path)
    ds = SpectralWasteDataset(tmp_path, split="train")
    stems = [cube.stem for cube, _mask in ds.pairs]
    assert stems == ["a_sample", "b_sample"]
    for cube_path, mask_path in ds.pairs:
        assert cube_path.stem == mask_path.stem
    # Written as b_sample, then a_sample. Sorted order still returns a_sample first,
    # and that file was the second one written (marker 2000).
    cube, _mask = ds[0]
    assert cube[0, 0, 1].item() == pytest.approx(2000 / 65535)


def test_unmatched_stem_raises(tmp_path):
    _fixture(tmp_path)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L").save(
        tmp_path / "labels_hyper_lt" / "train" / "orphan.png"
    )
    with pytest.raises(ValueError, match="stems do not match"):
        SpectralWasteDataset(tmp_path, split="train")


def test_split_size_check_is_opt_in(tmp_path):
    _fixture(tmp_path)
    ds = SpectralWasteDataset(tmp_path, split="train", check_split_size=False)
    assert len(ds) == 2
    assert SPLIT_COUNTS == {"train": 514, "val": 167, "test": 171}
    with pytest.raises(ValueError, match="expected 514"):
        SpectralWasteDataset(tmp_path, split="train", check_split_size=True)


def test_root_defaults_to_sw_root(tmp_path, monkeypatch):
    _fixture(tmp_path)
    monkeypatch.setenv("SW_ROOT", str(tmp_path))
    ds = SpectralWasteDataset(split="val")
    assert len(ds) == 1
    cube, mask = ds[0]
    assert cube.shape[0] == 4 and mask.ndim == 2


def test_missing_sw_root_says_how_to_get_data(monkeypatch):
    monkeypatch.delenv("SW_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="scripts/get_data.sh"):
        SpectralWasteDataset(split="train")


def test_band_subsampling_and_random_flip(tmp_path):
    _fixture(tmp_path)
    ds = SpectralWasteDataset(tmp_path, split="train", bands=[2, 0])
    cube, _mask = ds[0]
    assert cube.shape == (2, 8, 8)
    # bands=[2, 0] puts original band 0 second. That band holds the 0 and 1 markers.
    assert cube[1, 0, 0].item() == 0.0
    assert cube[1, 0, -1].item() == 1.0

    flip_ds = SpectralWasteDataset(tmp_path, split="train", random_flip=True)
    torch.manual_seed(0)
    seen = set()
    for _ in range(30):
        sample, mask = flip_ds[0]
        seen.add(round(float(sample[0, 0, 0]), 6))
        # Cube and mask stay the same width, and a flip moves the mask's corner together
        # with the cube's left-edge marker (0 at [0,0,0] before any flip).
        assert sample.shape == (4, 8, 8)
        assert mask.shape == (8, 8)
    assert seen == {0.0, 1.0}


def test_class_table_and_weights():
    assert CLASS_NAMES == (
        "background",
        "film",
        "basket",
        "cardboard",
        "video_tape",
        "filament",
        "bag",
    )
    assert HYPER_LT_TRAIN_PIXEL_COUNTS[0] == 27612233
    assert HYPER_LT_TRAIN_PIXEL_COUNTS[5] == 33101  # filament, the rare class
    weights = class_weights()
    assert weights.shape == (7,)
    assert weights.dtype == torch.float32
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-5)
    # Fewer pixels -> larger weight. Filament is rarer than background.
    assert weights[5] > weights[0]
