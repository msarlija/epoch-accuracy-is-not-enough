# -*- coding: utf-8 -*-
"""TSO (time-since-onset) feature branch — math ported exactly from the
Keras ``time_features`` + time_1/time_2 dense uplift:

    t -> [t, sin 2πt, cos 2πt, sin 4πt, cos 4πt, sin 6πt, cos 6πt]  (7-D)
      -> Dense(16, relu) -> Dense(32, relu)                          (32-D)

Input jitter U(−0.025, 0.025) is applied in the DATASET (train only), not
here, so the module stays deterministic and export-friendly.

Ablation arms (§3.7), replacing the old random-U[0,1] "time-naïve" baseline:

* ``full``  — normalized TSO through the branch (proposed model)
* ``none``  — branch absent from the graph entirely (parameter count drops)
* ``zero``  — branch present, INPUT zero-filled; matches ``full``'s
              parameter count exactly, isolating temporal information from
              capacity
"""

import math

import torch
from torch import nn

TSO_FEATURE_DIM = 7
TSO_MODES = ("full", "none", "zero")


def tso_features(t: torch.Tensor) -> torch.Tensor:
    """(...,) -> (..., 7) harmonic expansion."""
    two_pi = 2.0 * math.pi
    return torch.stack([
        t,
        torch.sin(two_pi * t), torch.cos(two_pi * t),
        torch.sin(2 * two_pi * t), torch.cos(2 * two_pi * t),
        torch.sin(3 * two_pi * t), torch.cos(3 * two_pi * t),
    ], dim=-1)


class TSOEmbedding(nn.Module):
    """(B, T) normalized time -> (B, T, out_dim) embedding, or None when the
    mode is 'none' (callers treat a None branch as absent)."""

    def __init__(self, mode: str = "full", hidden_dim: int = 16,
                 out_dim: int = 32):
        super().__init__()
        if mode not in TSO_MODES:
            raise ValueError(f"tso mode must be one of {TSO_MODES}")
        self.mode = mode
        self.out_dim = out_dim if mode != "none" else 0
        #: width of the raw harmonic skip fed to the sequence head (0 when the
        #: branch is absent, so the time-naive arm carries no time at all)
        self.raw_dim = TSO_FEATURE_DIM if mode != "none" else 0
        if mode != "none":
            self.time_1 = nn.Linear(TSO_FEATURE_DIM, hidden_dim)
            self.time_2 = nn.Linear(hidden_dim, out_dim)
            self.act = nn.ReLU()

    def raw_features(self, t: torch.Tensor):
        """The un-uplifted 7-D harmonic expansion, for the classifier skip.

        Returns None when the branch is absent, so the time-naive arm carries no
        skip either — otherwise ``tso_none`` would still receive time and stop
        being a time-naive control.

        ``mode`` is honoured here for the same reason: ``zero`` feeds a constant
        through the skip exactly as it feeds one through the branch, which is
        what keeps it parameter- AND information-matched to ``full``.

        Raw, not uplifted, matching Keras: the two Dense uplift layers are
        commented out in ``get_sequential_model``, so its skip carries the bare
        harmonics.
        """
        if self.mode == "none":
            return None
        if self.mode == "zero":
            t = torch.zeros_like(t)
        return tso_features(t)

    def forward(self, t: torch.Tensor):
        if self.mode == "none":
            return None
        if self.mode == "zero":
            t = torch.zeros_like(t)
        feats = tso_features(t)
        return self.act(self.time_2(self.act(self.time_1(feats))))
