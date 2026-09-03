"""Spike: wavelength-conditioned spectral forward on a fake cube; C can change."""

from __future__ import annotations

import torch

from spectral_stub import SpectralStub, masked_mean_over_bands


def test_dummy_cube_then_variable_c():
    """Same module, C=32 then C=20, no rebuild. PLAN.md Phase 0 done-when."""
    stub = SpectralStub(d_model=64, n_heads=4)
    stub.eval()

    cube32 = torch.randn(2, 32, 16, 16)
    waves32 = torch.linspace(400.0, 2500.0, 32)
    mask = torch.ones(32, dtype=torch.bool)
    mask[torch.randperm(32)[:10]] = False  # drop 10 bands, keep 22
    assert mask.sum() >= 8

    out_masked = stub(cube32, waves32, band_mask=mask)
    c_keep = int(mask.sum())
    assert out_masked.shape == (2, 16, 16, c_keep, 64)
    assert torch.isfinite(out_masked).all()

    pooled = masked_mean_over_bands(out_masked)
    assert pooled.shape == (2, 16, 16, 64)

    cube20 = torch.randn(2, 20, 16, 16)
    waves20 = torch.linspace(430.0, 860.0, 20)
    out20 = stub(cube20, waves20, band_mask=None)
    assert out20.shape == (2, 16, 16, 20, 64)
    assert torch.isfinite(out20).all()


def test_one_pixel_spectrum_length_c():
    stub = SpectralStub(d_model=64, n_heads=4)
    cube = torch.randn(1, 13, 1, 1)
    waves = torch.linspace(500.0, 900.0, 13)
    out = stub(cube, waves)
    assert out.shape == (1, 1, 1, 13, 64)
    assert torch.isfinite(out).all()


def test_tied_phi_has_no_c_in_weight():
    """φ is Linear(1, D); C is not a module dimension."""
    stub = SpectralStub(d_model=64, n_heads=4)
    assert stub.phi.weight.shape == (64, 1)
    n_params = sum(p.numel() for p in stub.parameters())
    # Sanity: nothing in the module scales with a baked-in C.
    for name, p in stub.named_parameters():
        assert p.ndim == 0 or p.shape[0] != 32, name
    assert n_params > 0
