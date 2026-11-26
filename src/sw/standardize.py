"""Per-band mean/std standardisation for SpectralWaste cubes.

Stats are computed from the TRAIN split only and applied the same way to
train, val, and test. Caching the tensors outside git (default: under the
run ``out_dir``) keeps recomputation off the hot path without committing
``.npy`` / ``.pt`` artefacts.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class BandStandardizer:
    """Affine per-band map ``(x - mean) / std`` with shape ``(C,)`` stats."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean.ndim != 1 or std.ndim != 1:
            raise ValueError("mean and std must be 1-D (C,)")
        if mean.numel() != std.numel():
            raise ValueError(f"mean has {mean.numel()} bands, std has {std.numel()}")
        if torch.any(std <= 0):
            raise ValueError("std must be positive for every band")
        self.mean = mean
        self.std = std.clamp_min(eps)
        self.eps = float(eps)

    @property
    def n_bands(self) -> int:
        return int(self.mean.numel())

    def __call__(self, cube: torch.Tensor) -> torch.Tensor:
        """``cube`` is ``(C, H, W)`` or ``(B, C, H, W)``."""
        if cube.ndim == 3:
            if cube.shape[0] != self.n_bands:
                raise ValueError(
                    f"cube has {cube.shape[0]} bands, standardizer expects {self.n_bands}"
                )
            mean = self.mean.to(device=cube.device, dtype=cube.dtype).view(-1, 1, 1)
            std = self.std.to(device=cube.device, dtype=cube.dtype).view(-1, 1, 1)
            return (cube - mean) / std
        if cube.ndim == 4:
            if cube.shape[1] != self.n_bands:
                raise ValueError(
                    f"cube has {cube.shape[1]} bands, standardizer expects {self.n_bands}"
                )
            mean = self.mean.to(device=cube.device, dtype=cube.dtype).view(1, -1, 1, 1)
            std = self.std.to(device=cube.device, dtype=cube.dtype).view(1, -1, 1, 1)
            return (cube - mean) / std
        raise ValueError(f"expected (C,H,W) or (B,C,H,W), got {tuple(cube.shape)}")


def compute_band_stats(dataset: Dataset, *, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
    """Per-band mean and std over every pixel in ``dataset``.

    ``split`` must be ``"train"``. Passing ``"val"`` or ``"test"`` raises so
    cached stats cannot silently come from a held-out split.
    """
    if split != "train":
        raise ValueError(
            f"band stats must be computed from the train split only, got {split!r}"
        )
    if len(dataset) == 0:
        raise RuntimeError("cannot compute band stats on an empty dataset")

    total = None
    total_sq = None
    n_pixels = 0
    n_bands = None
    for cube, _mask in dataset:
        if not torch.is_tensor(cube):
            cube = torch.as_tensor(cube)
        cube = cube.detach().to(dtype=torch.float32)
        if cube.ndim != 3:
            raise ValueError(f"expected cube (C,H,W), got {tuple(cube.shape)}")
        c, h, w = cube.shape
        if n_bands is None:
            n_bands = c
            total = torch.zeros(c, dtype=torch.float64)
            total_sq = torch.zeros(c, dtype=torch.float64)
        elif c != n_bands:
            raise ValueError(f"inconsistent band count: saw {n_bands} then {c}")
        flat = cube.reshape(c, -1).to(dtype=torch.float64)
        total += flat.sum(dim=1)
        total_sq += (flat * flat).sum(dim=1)
        n_pixels += h * w

    mean = (total / n_pixels).to(dtype=torch.float32)
    var = (total_sq / n_pixels) - mean.to(dtype=torch.float64).pow(2)
    std = var.clamp_min(0.0).sqrt().to(dtype=torch.float32)
    # Degenerate bands (constant) get a unit std so the affine stays defined.
    std = torch.where(std > 0, std, torch.ones_like(std))
    return mean, std


def save_band_stats(path: Path | str, mean: torch.Tensor, std: torch.Tensor, *, split: str = "train") -> None:
    """Write mean/std tensors. Refuses to label the file as val/test."""
    if split != "train":
        raise ValueError(f"refusing to save band stats labeled split={split!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": torch.as_tensor(mean, dtype=torch.float32).cpu(),
        "std": torch.as_tensor(std, dtype=torch.float32).cpu(),
        "split": "train",
        "n_bands": int(torch.as_tensor(mean).numel()),
    }
    torch.save(payload, path)


def load_band_stats(path: Path | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a stats file. Raises if the file claims a non-train split."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    split = payload.get("split", "train")
    if split != "train":
        raise ValueError(
            f"{path} has split={split!r}; band stats must come from the train split only"
        )
    mean = torch.as_tensor(payload["mean"], dtype=torch.float32).reshape(-1)
    std = torch.as_tensor(payload["std"], dtype=torch.float32).reshape(-1)
    return mean, std


def load_or_compute_band_stats(
    cache_path: Path | str,
    dataset: Dataset,
    *,
    split: str = "train",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached train stats, or compute and cache them from ``dataset``."""
    if split != "train":
        raise ValueError(
            f"band stats must be computed from the train split only, got {split!r}"
        )
    cache_path = Path(cache_path)
    if cache_path.is_file():
        return load_band_stats(cache_path)
    mean, std = compute_band_stats(dataset, split=split)
    save_band_stats(cache_path, mean, std, split="train")
    return mean, std
