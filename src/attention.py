"""From-scratch attention primitives for SSFT-Wave.

Implements Q/K/V, scaled dot-product, multi-head split/concat, residual+FFN,
and cross-attention with nn.Linear + softmax only.

Banned: nn.MultiheadAttention, F.multi_head_attention_forward, timm, HuggingFace
ViT blocks. See PLAN.md (SSFT §3.1 / §3.3).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention.

    Args:
        q: (..., S_q, d)
        k: (..., S_k, d)
        v: (..., S_k, d)
        key_mask: optional bool tensor broadcastable to (..., S_k). True = keep
            the key, False = mask it out (score set to -inf). Typical shape
            (B, S_k) or (B, 1, 1, S_k).

    Returns:
        out: (..., S_q, d)
        weights: (..., S_q, S_k) softmax over the last dim.
    """
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    if key_mask is not None:
        mask = key_mask
        # Unsqueeze so a (..., S_k) mask broadcasts over the query axis.
        while mask.ndim < scores.ndim:
            mask = mask.unsqueeze(-2)
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    out = torch.matmul(weights, v)
    return out, weights


def _split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, S, D) -> (B, H, S, d_h)."""
    b, s, d = x.shape
    d_h = d // n_heads
    return x.view(b, s, n_heads, d_h).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, S, d_h) -> (B, S, D)."""
    b, n_heads, s, d_h = x.shape
    return x.transpose(1, 2).contiguous().view(b, s, n_heads * d_h)


class MultiHeadSelfAttention(nn.Module):
    """MHSA: Q, K, V from the same sequence. From-scratch, no nn.MultiheadAttention."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, S, D)
            key_mask: (B, S) bool, True = keep.

        Returns:
            out: (B, S, D)
            weights: (B, H, S, S)
        """
        q = _split_heads(self.w_q(x), self.n_heads)
        k = _split_heads(self.w_k(x), self.n_heads)
        v = _split_heads(self.w_v(x), self.n_heads)
        attn_out, weights = scaled_dot_product_attention(q, k, v, key_mask=key_mask)
        return self.w_o(_merge_heads(attn_out)), weights


class MultiHeadCrossAttention(nn.Module):
    """Cross-attention: Q from query, K/V from context. SSFT §3.3 fusion uses this
    with spatial tokens as query and spectral tokens as key/value.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, S_q, D)
            context: (B, S_kv, D)
            key_mask: (B, S_kv) bool, True = keep.

        Returns:
            out: (B, S_q, D)
            weights: (B, H, S_q, S_kv)
        """
        q = _split_heads(self.w_q(query), self.n_heads)
        k = _split_heads(self.w_k(context), self.n_heads)
        v = _split_heads(self.w_v(context), self.n_heads)
        attn_out, weights = scaled_dot_product_attention(q, k, v, key_mask=key_mask)
        return self.w_o(_merge_heads(attn_out)), weights


class FeedForward(nn.Module):
    """Position-wise two-layer FFN (SSFT §3.1 / §3.3). Hidden = 4D by lock."""

    def __init__(self, d_model: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden if hidden is not None else 4 * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class ResidualSelfAttentionBlock(nn.Module):
    """Pre-LN residual MHSA + FFN. Paper omits LN; PLAN.md locks Pre-LN."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(
        self, x: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, weights = self.attn(self.ln1(x), key_mask=key_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, weights


class ResidualCrossAttentionBlock(nn.Module):
    """Pre-LN residual cross-attention + FFN (fusion block)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = MultiHeadCrossAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, weights = self.attn(
            self.ln_q(query), self.ln_kv(context), key_mask=key_mask
        )
        x = query + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, weights
