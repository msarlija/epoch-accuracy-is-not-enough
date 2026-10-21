# -*- coding: utf-8 -*-
"""Fusion head + Bi-GRU sequence encoder + classifier (§3.5 tail):

    concat(e_agg 64, tso 32) -> Dense 128 -> Bi-GRU 64 -> Bi-GRU 64
    -> per-epoch linear(5)

The fusion Dense is shared between Stage A (CNN-only, no recurrence) and
Stage B (sequence) so Stage A weights warm-start Stage B directly. GRUs are
kept plain (batch_first, no packed sequences, no data-dependent control
flow) so ONNX export of a future unidirectional deployment variant stays
possible.
"""

from typing import Optional

import torch
from torch import nn

N_CLASSES = 5


FUSION_NORM_MODES = ("post", "eeg_only", "none")


class FusionHead(nn.Module):
    """(B, T, D_in) -> (B, T, fusion_dim), LayerNorm placement selectable.

    Input is ``concat([eeg_aggregate, tso_embedding])``. Where the LayerNorm sits
    decides whether the TIME features get normalised too:

    * ``post`` (default, the ported behaviour) — ``ReLU(LN(dense(x)))``. The
      dense has already mixed EEG and time, so the LayerNorm normalises both.
    * ``eeg_only`` — ``ReLU(dense([LN(eeg), tso]))``. The EEG representation is
      still conditioned before the recurrent stack; the time features reach it
      raw.
    * ``none`` — no normalisation at all.

    WHY ``eeg_only`` EXISTS. Measured on a trained time-aware checkpoint, the
    ``post`` LayerNorm does NOT disadvantage time relative to EEG — the TSO share
    of the fused response is 84.9% either way — but it does strip the per-epoch
    mean, and that mean correlates +0.700 with time-of-night. So it deletes one
    channel the time branch could otherwise use to encode absolute position.

    Note the recurrent benefit LayerNorm is famous for (Ba et al. 2016) comes
    from normalising INSIDE the recurrent cell at every step, stabilising
    hidden-to-hidden dynamics. ``SequenceEncoder`` uses plain ``nn.GRU`` with no
    internal LayerNorm, so none of these modes deliver that; they only condition
    the GRU's input.
    """

    def __init__(self, in_dim: int, fusion_dim: int = 128,
                 mode: str = "post", eeg_dim: Optional[int] = None):
        super().__init__()
        if mode not in FUSION_NORM_MODES:
            raise ValueError(f"fusion_norm must be one of {FUSION_NORM_MODES}, "
                             f"got {mode!r}")
        if mode == "eeg_only" and eeg_dim is None:
            raise ValueError("fusion_norm='eeg_only' needs eeg_dim to know "
                             "where the EEG block ends and TSO begins")
        self.mode = mode
        self.eeg_dim = int(eeg_dim) if eeg_dim is not None else None
        self.dense = nn.Linear(in_dim, fusion_dim)
        #: Domain-adaptation point on the EEG representation ALONE, after the
        #: LayerNorm and before the time features are concatenated.
        #:
        #: Placement is forced. An adapter on the raw aggregated output would be
        #: partially cancelled by the LayerNorm that follows it -- LN removes a
        #: per-epoch mean and scale, which is exactly the kind of correction an
        #: affine map applies. Sitting after the LN, the correction survives
        #: into the fusion dense and therefore into the GRUs.
        #:
        #: Adapting here rather than at the classifier means the correction is
        #: made to the SENSORY features and the recurrent model then integrates
        #: corrected input, instead of the correction being applied after
        #: temporal integration has already happened. The time branch is left
        #: untouched, so the time signal is not distorted by it.
        self.adapt_eeg = ActivationAdapter(eeg_dim) if mode == "eeg_only" else None
        if mode == "post":
            self.norm = nn.LayerNorm(fusion_dim)
        elif mode == "eeg_only":
            self.norm = nn.LayerNorm(self.eeg_dim)
        else:
            self.norm = nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "eeg_only":
            eeg, rest = x[..., :self.eeg_dim], x[..., self.eeg_dim:]
            normed = self.norm(eeg)
            if self.adapt_eeg is not None:
                normed = self.adapt_eeg(normed)
            x = torch.cat([normed, rest], dim=-1)
            return self.act(self.dense(x))
        return self.act(self.norm(self.dense(x)))


class SequenceEncoder(nn.Module):
    """Two stacked Bi-GRUs with inter-layer dropout; TimeDistributed by
    construction (batch_first GRUs already operate per-step).

    Returns BOTH layers' outputs. The original Keras model feeds
    ``Concatenate([X_gru_1, X_gru_2])`` to its head rather than the last layer
    alone, so the classifier sees the shallower representation as well as the
    deeper one.

    No LayerNorm here, deliberately: the Keras model has ``LayerNormalization``
    after each Bi-GRU and BOTH lines are commented out, i.e. they were tried and
    rejected. That is stronger evidence than the LN-RNN literature, which in any
    case describes normalising INSIDE the cell rather than on the output.
    """

    def __init__(self, in_dim: int = 128, hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.gru1 = nn.GRU(in_dim, hidden, batch_first=True,
                           bidirectional=True)
        self.gru2 = nn.GRU(2 * hidden, hidden, batch_first=True,
                           bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor):
        h1, _ = self.gru1(x)
        h1 = self.drop(h1)
        h2, _ = self.gru2(h1)
        return h1, self.drop(h2)


class SequenceHead(nn.Module):
    """Keras' ``Dense(64) -> ReLU -> Dense(nclass)`` over the merged skip.

    Its input is ``concat([gru1, gru2, fused, time_harmonics])``. The point of
    the concatenation is the RAW time path: in the Keras model the LayerNorm
    normalises a representation in which EEG and time are already mixed, and
    absolute time-of-night survives only because ``time_feats`` is concatenated
    here, bypassing both the LayerNorm and the recurrent stack. The port
    previously dropped this skip, which left the LayerNorm as the only route for
    time — and it measurably strips a per-epoch mean correlating +0.700 with
    time-of-night.
    """

    def __init__(self, in_dim: int, hidden: int = 64,
                 n_classes: int = N_CLASSES):
        super().__init__()
        self.dense = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        #: TWO injection points, because they are not interchangeable.
        #:
        #: ``adapt_pre`` acts on the PRE-activation, so it changes which units
        #: fire: a shift can move a unit across the ReLU threshold and alter the
        #: representation's sparsity pattern. This is where AdaBN-style methods
        #: belong, since BatchNorm also sits pre-activation.
        #:
        #: ``adapt`` acts on the POST-activation, the representation ``fc``
        #: actually reads. Convenient for feature-space translation, but weaker
        #: than it looks: for any unit that is zero across the target domain, a
        #: post-ReLU shift is EXACTLY equivalent to changing ``fc.bias``, so it
        #: buys nothing beyond a per-class logit offset. Rectification has
        #: already discarded the information such a unit could have carried.
        self.adapt_pre = ActivationAdapter(hidden)
        self.adapt = ActivationAdapter(hidden)
        #: THIRD point, on the head's INPUT -- the concatenation itself, before
        #: any of this head's weights touch it. This is the representation the
        #: published domain-adaptation study aligns (its D=327 = 64 CNN + 256
        #: Bi-GRU + 7 time; here 391 = 128 + 128 gru + 128 fused + 7 time), so
        #: it is the point to use when reproducing that method. It is a
        #: genuinely different location from the two above, which sit AFTER
        #: ``dense`` has already mixed the concatenated blocks together.
        self.adapt_in = ActivationAdapter(in_dim)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        head_in = self.adapt_in(x)
        feats = self.adapt(self.act(self.adapt_pre(self.dense(head_in))))
        logits = self.fc(feats)
        return (logits, feats, head_in) if return_features else logits


class ActivationAdapter(nn.Module):
    """Diagonal affine transform of the classifier's penultimate activations.

    ``y = x * scale + shift``, an exact identity until something sets it. This
    is the single, well-defined injection point for post-hoc domain adaptation
    that works by shifting activations (AdaBN-style statistic matching,
    CORAL-style diagonal alignment, per-cohort bias correction). Without it the
    only ways in are monkey-patching or forward hooks, both of which break
    silently when the module structure changes.

    Placement is AFTER the head's ReLU and BEFORE the final linear map, i.e. on
    the representation the classifier actually reads. Two consequences worth
    knowing:

    * Nothing normalises downstream of this point, so an additive shift
      survives to the logits. A LayerNorm after it would cancel the mean
      component of any shift and silently halve what the method does.
    * The space is post-ReLU, hence non-negative for the un-adapted model. A
      shift may push activations negative; that is allowed, and is the point --
      the final layer is linear and has no trouble with it.

    The parameters are NON-PERSISTENT buffers: they follow ``.to(device)`` and
    ``.eval()`` but never enter ``state_dict``. That is deliberate. Checkpoints
    written before this class existed still load, and adaptation state stays a
    property of a deployment rather than of the trained weights.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("shift", torch.zeros(dim), persistent=False)
        self.register_buffer("scale", torch.ones(dim), persistent=False)
        self.enabled = False
        self.preserve_zeros = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        y = x * self.scale + self.shift
        if self.preserve_zeros:
            # Keep the rectified floor. A post-ReLU SHIFT otherwise lifts every
            # exact zero onto the same constant (0 + shift), so units that were
            # "off" all collapse to one value and the representation stops
            # meaning "this feature is absent". Scale alone never does this
            # (0 * scale == 0); this flag only matters when shift != 0.
            y = torch.where(x > 0, y, x)
        return y

    def set(self, shift=None, scale=None, preserve_zeros=None) -> None:
        """Set either term; passing both None disables and resets to identity.

        ``preserve_zeros`` is meaningful only at the POST-activation point,
        where exact zeros carry the meaning above. At the pre-activation point
        zero is an ordinary value and the flag should stay False.
        """
        if preserve_zeros is not None:
            self.preserve_zeros = bool(preserve_zeros)
        if shift is None and scale is None:
            self.enabled = False
            self.preserve_zeros = False
            self.shift.zero_()
            self.scale.fill_(1.0)
            return
        if shift is not None:
            t = torch.as_tensor(shift, dtype=self.shift.dtype,
                                device=self.shift.device).reshape(-1)
            if t.numel() not in (1, self.dim):
                raise ValueError(f"shift must be scalar or length {self.dim}, "
                                 f"got {tuple(t.shape)}")
            self.shift.copy_(t.expand(self.dim))
        if scale is not None:
            t = torch.as_tensor(scale, dtype=self.scale.dtype,
                                device=self.scale.device).reshape(-1)
            if t.numel() not in (1, self.dim):
                raise ValueError(f"scale must be scalar or length {self.dim}, "
                                 f"got {tuple(t.shape)}")
            self.scale.copy_(t.expand(self.dim))
        self.enabled = True


class Classifier(nn.Module):
    """Per-epoch logits. Softmax is left to the loss / caller."""

    def __init__(self, in_dim: int, n_classes: int = N_CLASSES):
        super().__init__()
        #: same injection point as SequenceHead, so Stage A adapts identically.
        #: Here the representation is the fused embedding itself -- there is no
        #: hidden layer in this head -- which is still the input the classifier
        #: reads, and still has nothing normalising it downstream.
        self.adapt = ActivationAdapter(in_dim)
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # This head has no hidden layer, so its input IS the representation the
        # classifier reads: "head_in" and "post" name the same tensor here.
        feats = self.adapt(x)
        logits = self.fc(feats)
        return (logits, feats, feats) if return_features else logits
