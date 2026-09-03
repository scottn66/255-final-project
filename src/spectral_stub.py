"""Spike dual-path stub: tied embed + λ-PE + spectral MHSA, C-agnostic.

Not full SSFT (that is session 2). This only proves:
  - a 1-pixel spectrum of length C attends
  - C can change between two forwards without rebuilding the module
  - band_mask / wavelength vector travel with the cube

API: forward(cube, wavelengths, band_mask)  — never assumes a fixed C.
cube: (B, C, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention import ResidualSelfAttentionBlock
from pe import sinusoidal_wavelength_pe


class SpectralStub(nn.Module):
    """Tied scalar projection φ: R → D, add λ-PE, MHSA over the C axis.

    Matches SSFT §3.1's per-location spectral tokens, minus spatial pooling
    (the spike can run on any H×W, including 1×1).
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, lambda_scale: float = 1000.0):
        super().__init__()
        self.d_model = d_model
        self.lambda_scale = lambda_scale
        # φ maps each scalar band value to D. Shared across bands = tied.
        self.phi = nn.Linear(1, d_model)
        self.block = ResidualSelfAttentionBlock(d_model, n_heads)

    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: torch.Tensor,
        band_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            cube: (B, C, H, W)
            wavelengths: (C,) nm
            band_mask: optional (C,) bool, True = keep. Sliced, so C shrinks.

        Returns:
            tokens: (B, H, W, C_keep, D) after residual MHSA over C_keep.
        """
        if cube.ndim != 4:
            raise ValueError(f"cube must be BCHW, got {tuple(cube.shape)}")
        b, c, h, w = cube.shape
        if wavelengths.shape != (c,):
            raise ValueError(f"wavelengths must be ({c},), got {tuple(wavelengths.shape)}")

        if band_mask is not None:
            if band_mask.shape != (c,):
                raise ValueError(f"band_mask must be ({c},), got {tuple(band_mask.shape)}")
            if not bool(band_mask.any()):
                raise ValueError("band_mask dropped every band")
            cube = cube[:, band_mask]
            wavelengths = wavelengths[band_mask]
            c = cube.shape[1]

        # (B, C, H, W) -> (B*H*W, C, 1) scalars at each spatial location.
        scalars = cube.permute(0, 2, 3, 1).reshape(b * h * w, c, 1)
        tokens = self.phi(scalars)  # (BHW, C, D)
        pe = sinusoidal_wavelength_pe(
            wavelengths, self.d_model, lambda_scale=self.lambda_scale
        )  # (C, D)
        tokens = tokens + pe.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(0)
        tokens, _ = self.block(tokens)  # MHSA over C, independent per location
        return tokens.view(b, h, w, c, self.d_model)


def masked_mean_over_bands(tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool the spectral axis: (B, H, W, C, D) -> (B, H, W, D)."""
    return tokens.mean(dim=3)
