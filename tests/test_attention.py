"""Unit tests for from-scratch MHSA / cross-attention."""

from __future__ import annotations

import pytest
import torch

from attention import (
    FeedForward,
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
    ResidualCrossAttentionBlock,
    ResidualSelfAttentionBlock,
    scaled_dot_product_attention,
)


def test_scaled_dot_product_shapes_and_softmax_rows():
    b, h, s, d = 2, 4, 7, 16
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (b, h, s, d)
    assert weights.shape == (b, h, s, s)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_key_mask_zeros_masked_mass():
    s_q, s_k, d = 3, 5, 8
    q = torch.randn(1, s_q, d)
    k = torch.randn(1, s_k, d)
    v = torch.randn(1, s_k, d)
    key_mask = torch.tensor([[True, True, False, False, True]])
    _, weights = scaled_dot_product_attention(q, k, v, key_mask=key_mask)
    masked_mass = weights[..., ~key_mask[0]].abs().sum()
    kept_rows = weights[..., key_mask[0]].sum(dim=-1)
    assert masked_mass.item() < 1e-6
    assert torch.allclose(kept_rows, torch.ones_like(kept_rows), atol=1e-5)


def test_mhsa_output_shape():
    attn = MultiHeadSelfAttention(d_model=64, n_heads=4)
    x = torch.randn(3, 11, 64)
    out, weights = attn(x)
    assert out.shape == x.shape
    assert weights.shape == (3, 4, 11, 11)
    assert torch.allclose(weights.sum(-1), torch.ones(3, 4, 11), atol=1e-5)


def test_mhsa_rejects_non_divisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(d_model=64, n_heads=3)


def test_cross_attention_spatial_queries_spectral_keys():
    """SSFT §3.3: spatial tokens query, spectral tokens K/V."""
    d, heads = 64, 4
    n_spatial, n_spectral = 4, 32
    ca = MultiHeadCrossAttention(d, heads)
    spatial = torch.randn(2, n_spatial, d)
    spectral = torch.randn(2, n_spectral, d)
    out, weights = ca(spatial, spectral)
    assert out.shape == (2, n_spatial, d)
    assert weights.shape == (2, heads, n_spatial, n_spectral)
    assert torch.allclose(weights.sum(-1), torch.ones(2, heads, n_spatial), atol=1e-5)


def test_one_pixel_spectrum_attends():
    """A 1-pixel spectrum of length C attends without crash (PLAN.md spike)."""
    c, d = 17, 64
    attn = MultiHeadSelfAttention(d, n_heads=4)
    spectrum = torch.randn(1, c, d)  # N=1 spatial location
    out, weights = attn(spectrum)
    assert out.shape == (1, c, d)
    assert weights.shape == (1, 4, c, c)
    assert torch.isfinite(out).all()


def test_residual_self_and_cross_blocks():
    x = torch.randn(2, 8, 64)
    ctx = torch.randn(2, 12, 64)
    y, _ = ResidualSelfAttentionBlock(64, 4)(x)
    z, _ = ResidualCrossAttentionBlock(64, 4)(x, ctx)
    assert y.shape == x.shape
    assert z.shape == x.shape
    assert torch.isfinite(y).all() and torch.isfinite(z).all()


def test_ffn_shape():
    ffn = FeedForward(64)
    x = torch.randn(2, 5, 64)
    assert ffn(x).shape == x.shape


def test_no_banned_attention_modules():
    """The implementation must not wrap nn.MultiheadAttention."""
    import attention as m

    src = open(m.__file__, encoding="utf-8").read()
    assert "nn.MultiheadAttention(" not in src
    assert "F.multi_head_attention_forward(" not in src
    assert "import timm" not in src
    assert "transformers" not in src
