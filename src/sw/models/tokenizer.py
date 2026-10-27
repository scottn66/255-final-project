"""
Band Tokenizer

Main task is to convert groups of bands into tokens. The tokens will then be used in the transformer model.

"""

from __future__ import annotations

import torch
import torch.nn as nn

from sw.pe import IndexBandPE


class BandTokenizer(nn.Module):
    """(B, C, H, W) -> tokens (B*h*w, G, D) and the (h, w) grid size."""

    def __init__(self, n_bands=224, d_model=64, group=16, patch=8):
        super().__init__()
        if n_bands % group != 0:
            raise ValueError(f"n_bands {n_bands} must be divisible by group {group}")
        self.n_bands = n_bands
        self.group = group
        self.n_groups = n_bands // group
        self.proj = nn.Conv2d(group, d_model, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(d_model)
        self.group_emb = IndexBandPE(self.n_groups, d_model)  

    def forward(self, x):
        b, c, h_in, w_in = x.shape
        if c != self.n_bands:
            raise ValueError(f"Input has {c} bands but tokenizer expects {self.n_bands}")
        t = self.proj(x.reshape(b * self.n_groups, self.group, h_in, w_in))
        d, h, w = t.shape[1:]
        t = t.reshape(b, self.n_groups, d, h, w).permute(0, 3, 4, 1, 2)
        t = self.norm(t.reshape(b, h * w, self.n_groups, d))   # (B, h*w, G, D)
        return t + self.group_emb(), (h, w)