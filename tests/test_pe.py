"""Unit tests for sinusoidal wavelength PE."""

from __future__ import annotations

import torch

from pe import IndexBandPE, sinusoidal_wavelength_pe


def test_pe_shape_and_finite():
    wavelengths = torch.linspace(400.0, 2500.0, 32)
    pe = sinusoidal_wavelength_pe(wavelengths, d_model=64)
    assert pe.shape == (32, 64)
    assert torch.isfinite(pe).all()


def test_pe_variable_c_without_module_rebuild():
    pe32 = sinusoidal_wavelength_pe(torch.linspace(400.0, 2500.0, 32), 64)
    pe20 = sinusoidal_wavelength_pe(torch.linspace(450.0, 900.0, 20), 64)
    assert pe32.shape == (32, 64)
    assert pe20.shape == (20, 64)


def test_nearby_wavelengths_are_closer_than_far():
    """Risk 3: 800 nm closer to 810 nm than to 2100 nm in L2."""
    pe = sinusoidal_wavelength_pe(torch.tensor([800.0, 810.0, 2100.0]), d_model=64)
    d_near = torch.linalg.vector_norm(pe[0] - pe[1])
    d_far = torch.linalg.vector_norm(pe[0] - pe[2])
    assert d_near < d_far


def test_odd_d_model():
    pe = sinusoidal_wavelength_pe(torch.tensor([500.0, 700.0]), d_model=5)
    assert pe.shape == (2, 5)
    assert torch.isfinite(pe).all()


def test_index_pe_packed_rows():
    table = IndexBandPE(n_bands=32, d_model=64)
    pe = table(n=32)
    assert pe.shape == (32, 64)
    pe_short = table(n=20)
    assert pe_short.shape == (20, 64)
    assert torch.equal(pe_short, pe[:20])
