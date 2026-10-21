# -*- coding: utf-8 -*-
"""Channel aggregation (§3.5) — the NEW block between the CNN backbone and
the TSO concatenation. Reduces per-channel embeddings E (B, T, C, D) to
(B, T, D), respecting a boolean channel mask (False = absent/dropped).

Modes: ``mean``, ``max``, ``attention`` (selectable for ablation).

Attention (single head — prior art found performance flat from 1 to 16 heads
even with 12 channels):

    u     = tanh(E @ W_ctx + b_ctx)     (B, T, C, K), K=32
    s     = u @ W_v                      (B, T, C)
    alpha = softmax(s over C)            (B, T, C)
    e_agg = sum_c alpha_c * E_c          (B, T, D)

Scoring weights are SHARED across channels, so the block is permutation- and
count-invariant; C=1 needs no special casing (softmax over one element = 1).
``alpha`` is returned as an optional output — interpretable as a learned
per-epoch signal-quality index (wanted for a figure).

Masking is implemented with ``torch.where`` (no data-dependent shapes) to
stay ONNX-exportable.
"""

from typing import Optional, Tuple

import torch
from torch import nn

AGGREGATION_MODES = ("mean", "max", "attention")
_NEG_INF = -1e9


class ChannelAggregation(nn.Module):
    def __init__(self, mode: str = "attention", embed_dim: int = 64,
                 context_dim: int = 32):
        super().__init__()
        if mode not in AGGREGATION_MODES:
            raise ValueError(f"mode must be one of {AGGREGATION_MODES}")
        self.mode = mode
        if mode == "attention":
            self.ctx = nn.Linear(embed_dim, context_dim)          # W_ctx, b_ctx
            self.score = nn.Linear(context_dim, 1, bias=False)    # W_v

    def forward(self, e: torch.Tensor, mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """e: (B, T, C, D); mask: (B, T, C) bool or None (=all present).

        Returns (e_agg (B, T, D), alpha (B, T, C) or None). ``alpha`` is the
        attention distribution; for mean/max modes it is None.
        """
        if mask is None:
            mask = torch.ones(e.shape[:3], dtype=torch.bool, device=e.device)
        maskf = mask.to(e.dtype)

        if self.mode == "mean":
            denom = maskf.sum(dim=-1, keepdim=True).clamp(min=1.0)
            agg = (e * maskf.unsqueeze(-1)).sum(dim=-2) / denom
            return agg, None

        if self.mode == "max":
            neg = torch.full_like(e, _NEG_INF)
            masked = torch.where(mask.unsqueeze(-1), e, neg)
            agg = masked.amax(dim=-2)
            # all-masked (shouldn't happen) -> fall back to zeros
            any_valid = mask.any(dim=-1, keepdim=True)
            return torch.where(any_valid, agg, torch.zeros_like(agg)), None

        # attention
        u = torch.tanh(self.ctx(e))                    # (B, T, C, K)
        s = self.score(u).squeeze(-1)                  # (B, T, C)
        s = torch.where(mask, s, torch.full_like(s, _NEG_INF))
        alpha = torch.softmax(s, dim=-1)               # (B, T, C)
        agg = (alpha.unsqueeze(-1) * e).sum(dim=-2)    # (B, T, D)
        return agg, alpha
