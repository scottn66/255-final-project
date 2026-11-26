"""
Band Tokenizer

Main task is to convert groups of bands into tokens. The tokens will then be used in the transformer model.

"""

from __future__ import annotations

import torch
import torch.nn as nn

from sw.pe import IndexBandPE


class BandTokenizer(nn.Module):
    """(B, C, H, W) -> tokens (B, h*w, G, D) and the (h, w) grid size.

    Tokens are row-major over the patch grid: token n is patch
    (n // w, n % w). Pixels past h * patch or w * patch are dropped.
    """

    def __init__(self, n_bands=224, d_model=64, group=16, patch=8, pe_init_std=1.0):
        super().__init__()
        if n_bands % group != 0:
            raise ValueError(f"n_bands {n_bands} must be divisible by group {group}")
        self.n_bands = n_bands
        self.group = group
        self.n_groups = n_bands // group
        self.pe_init_std = float(pe_init_std)
        self.proj = nn.Conv2d(group, d_model, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(d_model)
        self.group_emb = IndexBandPE(self.n_groups, d_model, init_std=pe_init_std)  

    def forward(self, x):
        b, c, h_in, w_in = x.shape
        if c != self.n_bands:
            raise ValueError(f"Input has {c} bands but tokenizer expects {self.n_bands}")
        t = self.proj(x.reshape(b * self.n_groups, self.group, h_in, w_in))
        d, h, w = t.shape[1:]
        t = t.reshape(b, self.n_groups, d, h, w).permute(0, 3, 4, 1, 2)
        t = self.norm(t.reshape(b, h * w, self.n_groups, d))   # (B, h*w, G, D)
        return t + self.group_emb(), (h, w)


class CollapsedTokenizer(nn.Module):
    """(B, C, H, W) -> tokens (B, h*w, 1, D) and the (h, w) grid size.

    The band-collapsed control. A 1x1 conv mixes all ``n_bands`` bands down
    to ``reduce`` channels at every pixel. Then a patch conv
    (``reduce -> d_model``, kernel and stride ``patch``) and a LayerNorm
    make one token per patch. The band axis is gone before any attention,
    so there is one group (``n_groups == 1``) and no group embedding: one
    group has no order to encode.

    Why the control needs this, and not ``BandTokenizer`` with no spectral
    blocks: a shared conv, an added group embedding and a mean over groups
    give the same pooled token whatever order the 14 groups come in. That
    model cannot tell which of the 14 band groups a feature sits in (it
    still sees a band's position inside its group). Here every band has its
    own weights in the 1x1 conv, so band position is kept.

    ``reduce=16`` is deliberate. The patch conv then has the same shape as
    the band tokenizer's (16 -> 64), and the control model is matched on
    parameters: ``BandTransformer(0, 8, tokenizer="collapsed")`` has 491,191
    against 488,487 for ``BandTransformer(2, 6)``. With 224 -> 32 the
    control would have 560,327, about 15% over the fused model. Only the
    parameter total is matched. FLOPs and step time are not.
    Token scale is not matched either: at init the tokens that go into
    ``SpatialPE`` have RMS 1.0 here, and they vary about 3 to 4 times as
    much from patch to patch as the band path's pooled tokens (the mean
    over 14 groups shrinks those).

    Tokens are row-major over the patch grid, exactly like
    ``BandTokenizer``: token n is patch (n // w, n % w). Pixels past
    h * patch or w * patch are dropped. The output keeps a group axis of
    length 1, so ``BandTransformer.forward`` runs unchanged (a mean over
    one group is the identity).
    """

    def __init__(self, n_bands=224, d_model=64, reduce=16, patch=8):
        super().__init__()
        if reduce < 1:
            raise ValueError(f"reduce must be >= 1, got {reduce}")
        self.n_bands = n_bands
        self.n_groups = 1
        self.reduce = nn.Conv2d(n_bands, reduce, kernel_size=1)
        self.proj = nn.Conv2d(reduce, d_model, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        b, c, h_in, w_in = x.shape
        if c != self.n_bands:
            raise ValueError(f"Input has {c} bands but tokenizer expects {self.n_bands}")
        t = self.proj(self.reduce(x))   # (B, D, h, w)
        d, h, w = t.shape[1:]
        t = self.norm(t.reshape(b, d, h * w).transpose(1, 2))   # (B, h*w, D), row-major
        return t.unsqueeze(2), (h, w)   # (B, h*w, 1, D)
