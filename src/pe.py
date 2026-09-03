"""Wavelength-aware positional encoding (LESSViT idea, sinusoidal v1).

SSFT §3.1 uses a learned table P ∈ R^{C×D} indexed by band *index*.
We replace that with a parameter-free sinusoid of the physical wavelength.
SSRoPE / 1D RoPE is a later stretch (LESSViT Appendix B).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sinusoidal_wavelength_pe(
    wavelengths_nm: torch.Tensor,
    d_model: int,
    lambda_scale: float = 1000.0,
    base: float = 10000.0,
) -> torch.Tensor:
    """Sinusoidal 1D PE on a wavelength vector.

    Args:
        wavelengths_nm: (C,) band-center wavelengths in nanometers.
        d_model: embedding width D.
        lambda_scale: divide nm by this before the sinusoid. Default 1000
            converts nm → microns so 400–2500 nm becomes 0.4–2.5 (PLAN.md
            risk 3). Set to 1.0 to use raw nm.
        base: Transformer geometric base (10000).

    Returns:
        pe: (C, D) float tensor on the same device/dtype as wavelengths_nm.
    """
    if wavelengths_nm.ndim != 1:
        raise ValueError(f"wavelengths_nm must be 1D (C,), got {tuple(wavelengths_nm.shape)}")
    if d_model < 1:
        raise ValueError("d_model must be >= 1")

    lam = wavelengths_nm.to(dtype=torch.float32) / lambda_scale  # (C,)
    half = (d_model + 1) // 2
    # div_term_i = base^{2i / d_model} for i = 0..half-1, paired sin/cos.
    i = torch.arange(half, device=wavelengths_nm.device, dtype=torch.float32)
    div = base ** (2 * i / d_model)
    angles = lam.unsqueeze(1) / div.unsqueeze(0)  # (C, half)
    pe = torch.zeros(wavelengths_nm.shape[0], d_model, device=wavelengths_nm.device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles[:, : pe[:, 0::2].shape[1]])
    n_cos = pe[:, 1::2].shape[1]
    pe[:, 1::2] = torch.cos(angles[:, :n_cos])
    return pe.to(dtype=wavelengths_nm.dtype if wavelengths_nm.dtype.is_floating_point else torch.float32)


class IndexBandPE(nn.Module):
    """Learned integer band-index embedding P ∈ R^{C_max×D} (SSFT §3.1 baseline).

    Packed 0..C-1 of the *current* tensor (PLAN.md A.3). Complementary subsets
    of the same C therefore reuse the same rows for different wavelengths.
    """

    def __init__(self, n_bands: int, d_model: int):
        super().__init__()
        self.table = nn.Embedding(n_bands, d_model)

    def forward(self, n: int | None = None) -> torch.Tensor:
        c = self.table.num_embeddings if n is None else n
        if c > self.table.num_embeddings:
            raise ValueError(f"requested C={c} > table size {self.table.num_embeddings}")
        idx = torch.arange(c, device=self.table.weight.device)
        return self.table(idx)
