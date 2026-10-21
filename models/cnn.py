# -*- coding: utf-8 -*-
"""CNN backbone — PyTorch port of ``get_cnn_with_time`` (models_ngcs.py,
"Version 3 - new") in the Keras repo.

Ported faithfully: two parallel entry branches (narrow kernels 5/11 for fast
morphology, wide kernels 25/51 for slow morphology), then the conv/pool/
dropout ladder 32-32-64-128-256 with kernel 5, SpatialDropout after the first
pool, plain Dropout deeper. All kernels are odd, so PyTorch ``padding=k//2``
reproduces Keras ``padding='same'`` exactly.

Changes vs. Keras (intentional, per the task spec):

* input length is 3000 samples (30 s @ 100 Hz) instead of 1250. The pooling
  schedule (5 pools of 2 -> /32) leaves a 93-position final feature map at
  L=3000 (vs 39 at L=1250), receptive field 3.23 s.

  A SIXTH pool was tried and REVERTED — recorded here so it is not re-attempted.
  The argument for it was that global AVERAGE pooling now averages 93 positions
  instead of 39, so a spindle spanning one or two receptive fields contributes
  ~1/93 of the average rather than ~1/39, diluted 2.4x purely because the window
  got wider (global MAX pooling is scale-free and unaffected, so half the 64-d
  embedding took the hit and half did not). Six pools gave 46 positions, RF
  4.51 s, and halved the cost of ``conv5_1`` — the most expensive layer at 26.8%
  of the backbone — at no parameter cost.

  Measured on fold 0 it was WORSE: -0.017 external kappa, worse in 7 of 7
  external cohorts, internal within noise. Halving temporal resolution (323 ->
  652 ms between positions) evidently costs more than the averaging dilution it
  fixes. The reasoning was sound and the data disagreed; keep five pools.
* the time branch lives in models/tso.py, and the dual pooled projections
  produce the per-channel embedding E directly: proj convs output
  ``embed_dim//2`` each so that GAP‖GMP concat is ``embed_dim`` (=64), the
  per-channel embedding that channel aggregation (§3.5) consumes. (The Keras
  model projected to 64+64=128 and mixed time in before a Dense(64); that
  Dense moved into the fusion head, models/seq.py.)

The module is applied per channel with SHARED weights: callers fold the
channel axis into the batch (see models/model.py).

NORMALISATION. ``norm`` inserts a normalisation layer in the standard position,
conv -> norm -> ReLU, in every conv block; the 1x1 projections feeding the
pooling heads are left alone, being output head rather than feature extraction.
The default is ``none``, which is a deliberate modelling choice rather than an
oversight: this task's classes are partly DEFINED by amplitude (AASM scores N3
from slow waves >75 uV peak-to-peak), and the levels below differ in how much of
that they destroy. The `norm` ablation was run and
dropped: instance norm significantly HURT on the de-confounded run (kappa
-0.0095, p=0.0016). The option remains; no shipped config uses it.
"""

import torch
from torch import nn


NORM_MODES = ("none", "batch", "group", "instance")


def make_norm(mode: str, channels: int) -> nn.Module:
    """Normalisation layer for ``channels`` feature maps over a time axis.

    * ``none``     — identity; amplitude reaches the classifier intact.
    * ``batch``    — BatchNorm1d. Normalises each feature across the batch, so
      between-epoch amplitude ORDER survives, but inference uses running
      statistics estimated on the training pool. With a pool that is 56%
      mastoid-proxy and external cohorts that are mostly prefrontal and differ
      several-fold in amplitude, that is the concrete domain-shift liability.
    * ``group``    — GroupNorm, batch-independent, so it separates "normalisation
      helps optimisation" from "batch statistics transfer badly". Group size is
      held at 8 channels where possible, and every channel count in this backbone
      divides evenly.
    * ``instance`` — InstanceNorm1d, per epoch per feature over TIME. This is the
      variant that deletes amplitude: each epoch is divided by its own spread, so
      a 75 uV slow wave and a 15 uV one become indistinguishable. ``affine=True``
      keeps its parameter count comparable to the other levels, and does not
      restore the information — the affine scale is learned per feature and
      constant across epochs.
    """
    if mode == "none":
        return nn.Identity()
    if mode == "batch":
        return nn.BatchNorm1d(channels)
    if mode == "group":
        return nn.GroupNorm(max(1, min(32, channels // 8)), channels)
    if mode == "instance":
        return nn.InstanceNorm1d(channels, affine=True)
    raise ValueError(f"norm must be one of {NORM_MODES}, got {mode!r}")


class ConvBlock(nn.Module):
    """conv -> norm -> ReLU (norm is Identity unless requested)."""

    def __init__(self, c_in: int, c_out: int, kernel: int, norm: str = "none"):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2)
        self.norm = make_norm(norm, c_out)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class CNNBackbone(nn.Module):
    """(B*, 1, n_samples) -> (B*, embed_dim) per-channel epoch embedding."""

    def __init__(self, embed_dim: int = 64, droprate: float = 0.3,
                 droprate_spatial: float = 0.05, norm: str = "none"):
        super().__init__()
        if embed_dim % 2:
            raise ValueError("embed_dim must be even (GAP/GMP halves)")
        if norm not in NORM_MODES:
            raise ValueError(f"norm must be one of {NORM_MODES}, got {norm!r}")
        half = embed_dim // 2
        self.embed_dim = embed_dim
        self.norm_mode = norm

        # --- Branch A: narrow kernels (fast morphology) ---
        self.conv1_1 = ConvBlock(1, 16, 5, norm)
        self.conv1_2 = ConvBlock(16, 16, 11, norm)
        # --- Branch B: wide kernels (smoother / slower morphology) ---
        self.conv1_wide_1 = ConvBlock(1, 8, 25, norm)
        self.conv1_wide_2 = ConvBlock(8, 8, 51, norm)

        self.pool = nn.MaxPool1d(2)
        self.spatial_drop = nn.Dropout1d(droprate_spatial)
        self.drop = nn.Dropout(droprate)

        self.conv2_1 = ConvBlock(24, 32, 5, norm)
        self.conv2_2 = ConvBlock(32, 32, 5, norm)
        self.conv3_1 = ConvBlock(32, 64, 5, norm)
        self.conv4_1 = ConvBlock(64, 128, 5, norm)
        self.conv5_1 = ConvBlock(128, 256, 5, norm)

        # 1x1 projections feeding the two pooling paths
        self.proj_max = nn.Conv1d(256, half, 1)
        self.proj_avg = nn.Conv1d(256, half, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*, 1, L)
        xa = self.conv1_2(self.conv1_1(x))
        xb = self.conv1_wide_2(self.conv1_wide_1(x))
        h = torch.cat([xa, xb], dim=1)          # (B*, 24, L)

        h = self.spatial_drop(self.pool(h))     # L/2

        h = self.conv2_1(h)
        h = self.pool(h)                        # L/4
        h = self.conv2_2(h)
        h = self.drop(self.pool(h))             # L/8

        h = self.conv3_1(h)
        h = self.drop(self.pool(h))             # L/16

        h = self.conv4_1(h)
        h = self.drop(self.pool(h))             # L/32

        h = self.drop(self.conv5_1(h))          # (B*, 256, L/32)

        h_max = self.proj_max(h).amax(dim=-1)   # global max pool
        h_avg = self.proj_avg(h).mean(dim=-1)   # global average pool
        return torch.cat([h_max, h_avg], dim=1)  # (B*, embed_dim)
