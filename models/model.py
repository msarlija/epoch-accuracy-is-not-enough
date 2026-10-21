# -*- coding: utf-8 -*-
"""Model assembly (§3.5):

    Input           (B, T, C, 3000)
      ↓ fold C into batch → (B·T·C, 1, 3000)
    CNN backbone    shared weights across channels
      ↓ GAP + GMP concat → (B·T·C, 64)
      ↓ unfold → (B, T, C, 64)
    Channel aggregation → (B, T, 64)
      ↓ concat TSO embedding (32)
    Dense 128 → Bi-GRU 64 → Bi-GRU 64
      ↓ concat(gru1, gru2, fused, raw time harmonics)
    Dense 64 → per-epoch logits(5)

Stage A (CNN-only) uses the SAME module with ``with_sequence=False``: the
recurrent encoder is skipped and a separate epoch head classifies the fused
representation directly. Stage A is a separately evaluated model (its metrics
are scientifically load-bearing), and its cnn/aggregation/tso/fusion weights
warm-start Stage B via ``load_state_dict(..., strict=False)``.

All experiment arms are constructor arguments driven by config:
``tso_mode`` ∈ {full, none, zero}, ``aggregation`` ∈ {mean, max, attention},
``norm`` ∈ {none, batch, group, instance} (backbone normalisation, §models/cnn).
"""

import logging
from typing import Dict, Optional

import torch
from torch import nn

from .aggregation import ChannelAggregation
from .cnn import CNNBackbone
from .seq import (Classifier, FusionHead, SequenceEncoder,
                  SequenceHead, N_CLASSES)
from .tso import TSOEmbedding


log = logging.getLogger(__name__)


class SleepStager(nn.Module):
    def __init__(self,
                 with_sequence: bool = True,
                 tso_mode: str = "full",
                 aggregation: str = "attention",
                 embed_dim: int = 64,
                 tso_dim: int = 32,
                 tso_hidden: int = 16,
                 fusion_dim: int = 128,
                 gru_hidden: int = 64,
                 droprate_cnn: float = 0.3,
                 droprate_cnn_spatial: float = 0.05,
                 norm: str = "none",
                 fusion_norm: str = "post",
                 head_time_skip: bool = False,
                 droprate_gru: float = 0.1,
                 attention_context_dim: int = 32,
                 n_classes: int = N_CLASSES):
        super().__init__()
        self.with_sequence = with_sequence
        self.tso_mode = tso_mode
        self.aggregation_mode = aggregation

        self.cnn = CNNBackbone(embed_dim=embed_dim, droprate=droprate_cnn,
                               droprate_spatial=droprate_cnn_spatial,
                               norm=norm)
        self.agg = ChannelAggregation(mode=aggregation, embed_dim=embed_dim,
                                      context_dim=attention_context_dim)
        self.tso = TSOEmbedding(mode=tso_mode, hidden_dim=tso_hidden,
                                out_dim=tso_dim)
        self.fuse = FusionHead(embed_dim + self.tso.out_dim, fusion_dim,
                               mode=fusion_norm, eeg_dim=embed_dim)

        if with_sequence:
            self.seq = SequenceEncoder(fusion_dim, gru_hidden, droprate_gru)
            # Keras' merge: Concatenate([gru_1, gru_2, X_features_out,
            # time_feats]) -> Dense(64) -> ReLU -> Dense(nclass). The raw time
            # harmonics are what let absolute time-of-night reach the classifier
            # without passing through the fusion LayerNorm or the GRUs.
            # Raw time no longer reaches the head directly. Measured on the E6
            # checkpoints, that route carried essentially NOTHING: disabling it
            # at inference cost 0.0001 kappa, against 0.0084 for the branch. It
            # was a two-layer path from time to the logits (Dense(64) ->
            # Linear(5)) bypassing both the GRU and the EEG, so removing it
            # leaves the only time route as the one that must interact with the
            # recurrent representation.
            self.head_time_skip = bool(head_time_skip)
            # Runtime gate for the Stage B warm-up, toggled by
            # training.stages. NOT a constructor argument: it is training-phase
            # state, not architecture, and must leave the parameter count
            # untouched so the arms stay matched.
            self.head_time_skip_active = True
            head_in = 2 * self.seq.out_dim + fusion_dim + (
                self.tso.raw_dim if self.head_time_skip else 0)
            self.seq_head = SequenceHead(head_in, gru_hidden, n_classes)
        else:
            self.epoch_head = Classifier(fusion_dim, n_classes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, tso: torch.Tensor,
                channel_mask: Optional[torch.Tensor] = None,
                return_alpha: bool = False,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        x:            (B, T, C, S) normalized EEG epochs
        tso:          (B, T) normalized time-since-onset
        channel_mask: (B, T, C) bool, False = absent/dropped channel
        Returns {'logits': (B, T, n_classes)[, 'alpha': (B, T, C)]}.
        """
        B, T, C, S = x.shape
        flat = x.reshape(B * T * C, 1, S)          # fold channels into batch
        emb = self.cnn(flat)                        # (B·T·C, D) shared weights
        emb = emb.reshape(B, T, C, -1)              # unfold

        agg, alpha = self.agg(emb, channel_mask)    # (B, T, D)

        t_emb = self.tso(tso)                       # (B, T, 32) or None
        fused_in = agg if t_emb is None else torch.cat([agg, t_emb], dim=-1)
        h = self.fuse(fused_in)                     # (B, T, fusion_dim)

        if self.with_sequence:
            g1, g2 = self.seq(h)
            parts = [g1, g2, h]
            if self.head_time_skip:
                t_raw = self.tso.raw_features(tso)
                if t_raw is not None:
                    # Zeroed, not omitted: the head's input width is fixed at
                    # construction. Zero input also means zero gradient to the
                    # weights on these dims, so they stay at init until the
                    # skip is switched on.
                    parts.append(t_raw if self.head_time_skip_active
                                 else torch.zeros_like(t_raw))
            head_out = self.seq_head(torch.cat(parts, dim=-1),
                                     return_features=return_features)
        else:
            head_out = self.epoch_head(h, return_features=return_features)
        logits, feats, head_in = (head_out if return_features
                                  else (head_out, None, None))

        out = {"logits": logits}
        if return_alpha and alpha is not None:
            out["alpha"] = alpha
        if return_features:
            out["features"] = feats        # what fc reads (post-ReLU, 64-d)
            out["head_input"] = head_in    # the concatenation (391-d)
        return out

    # ------------------------------------------------------------------
    def set_dropout_rates(self, cnn=None, gru=None, cnn_spatial=None) -> dict:
        """Retune dropout in place, for the stage-dependent schedule.

        Dropout is a LAYER, not a parameter, so freezing the backbone does not
        disable it: without this, Stage B's frozen warm-up injects full CNN
        dropout noise into features that are not learning, corrupting the very
        representation the fresh GRUs are trying to fit. The Keras pipeline sets
        ``droprate_cnn = 0`` for exactly that phase and raises it to 0.1 only
        once everything unfreezes.

        ``cnn_spatial`` is only changed when passed. The shipped schedule zeroes
        it during the frozen warm-up and restores it on unfreeze — a deliberate
        divergence from Keras, which hardcodes ``droprate_1D = 0.05``
        independently of ``droprate``. A frozen backbone should present a
        DETERMINISTIC representation: its weights cannot overfit while they are
        not being updated, so the noise buys no regularisation and only makes the
        GRUs fit a moving target.

        Returns the rates actually applied, for logging.
        """
        applied = {}
        if cnn is not None:
            self.cnn.drop.p = float(cnn)
            applied["cnn"] = float(cnn)
        if cnn_spatial is not None:
            self.cnn.spatial_drop.p = float(cnn_spatial)
            applied["cnn_spatial"] = float(cnn_spatial)
        if gru is not None and hasattr(self, "seq"):
            self.seq.drop.p = float(gru)
            applied["gru"] = float(gru)
        return applied

    def set_time_skip_active(self, active: bool) -> bool:
        """Switch the raw time skip into or out of the classifier.

        The skip is a PARAMETER-FREE route (``tso.raw_features`` is a fixed
        function of t), so ``requires_grad_(False)`` cannot hold it back — the
        thing that learns is the ``seq_head`` weights reading those 7 dims, and
        those are freshly initialised in Stage B and always trainable.

        That is the problem this exists to solve. During the Stage B warm-up the
        backbone, aggregation, TSO branch and fusion are all frozen
        (``cnn_parameters``), so the raw skip would be the ONLY unfrozen time
        route into the logits, available precisely while the GRUs are at their
        weakest and least useful. A two-layer path from time to the logits that
        bypasses both the GRU and the EEG is the easiest available loss
        reduction, and letting the head fit it first is textbook shortcut
        learning. Gating it means the GRUs are trained before the classifier is
        allowed to use raw time at all.

        No-op on models built without the skip. Returns the resulting state.
        """
        if getattr(self, "head_time_skip", False):
            self.head_time_skip_active = bool(active)
            return self.head_time_skip_active
        return False

    def forward_from_agg(self, agg: torch.Tensor, tso: torch.Tensor,
                         return_features: bool = False) -> Dict[str, torch.Tensor]:
        """Everything after the CNN, from a cached channel-aggregated tensor.

        ``agg`` is (B, T, embed_dim) as produced by the backbone plus channel
        aggregation. Lets a correction applied at the EEG representation be
        evaluated without re-running the backbone, which is ~57% of the
        parameters and the overwhelming majority of the compute.
        """
        t_emb = self.tso(tso)
        fused_in = agg if t_emb is None else torch.cat([agg, t_emb], dim=-1)
        h = self.fuse(fused_in)
        if not self.with_sequence:
            out = self.epoch_head(h, return_features=return_features)
            logits, feats, head_in = (out if return_features else (out, None, None))
        else:
            g1, g2 = self.seq(h)
            parts = [g1, g2, h]
            if self.head_time_skip:
                t_raw = self.tso.raw_features(tso)
                if t_raw is not None:
                    parts.append(t_raw if self.head_time_skip_active
                                 else torch.zeros_like(t_raw))
            out = self.seq_head(torch.cat(parts, dim=-1),
                                return_features=return_features)
            logits, feats, head_in = (out if return_features else (out, None, None))
        res = {"logits": logits}
        if return_features:
            res["features"], res["head_input"] = feats, head_in
        return res

    # ------------------------------------------------------------------
    @property
    def adapter(self):
        """The post-activation ActivationAdapter in whichever head this has."""
        return (self.seq_head if self.with_sequence else self.epoch_head).adapt

    def adapter_at(self, where: str = "post"):
        """``head_in`` (the concatenation entering the head), ``pre`` (after
        the head's dense, before its ReLU) or ``post`` (what fc reads).

        Stage A has no ``pre``: its Classifier is a bare linear map with no
        hidden layer, and its input is already the post-ReLU fusion output.
        """
        head = self.seq_head if self.with_sequence else self.epoch_head
        if where == "eeg":
            a = getattr(self.fuse, "adapt_eeg", None)
            if a is None:
                raise ValueError("where='eeg' needs fusion_norm='eeg_only'; "
                                 "there is no separable EEG block otherwise")
            return a
        if where == "post":
            return self.adapter
        if where == "head_in":
            # Stage A's Classifier has no hidden layer, so its input already IS
            # what the classifier reads and `adapt` serves both roles.
            return getattr(head, "adapt_in", head.adapt)
        if where != "pre":
            raise ValueError(
                f"where must be 'eeg', 'pre', 'post' or 'head_in', "
                f"got {where!r}")
        pre = getattr(head, "adapt_pre", None)
        if pre is None:
            raise ValueError(
                "this head has no pre-activation point (Stage A's Classifier "
                "has no hidden layer); use where='post' or 'head_in'")
        return pre

    def set_domain_shift(self, shift=None, scale=None, where: str = "post",
                         preserve_zeros=None) -> None:
        """Configure post-hoc domain adaptation on the classifier's input.

        ``y = x * scale + shift`` applied to the representation the final linear
        layer reads. Call with no arguments to reset to an exact identity.

        Typical use: collect ``forward(..., return_features=True)["features"]``
        over a source and a target cohort, then set ``scale`` and ``shift`` to
        map the target's per-dimension statistics onto the source's. Nothing
        downstream normalises, so the transform reaches the logits intact.
        """
        self.adapter_at(where).set(shift=shift, scale=scale,
                                   preserve_zeros=preserve_zeros)

    def feature_dim(self) -> int:
        """Width of the adapted representation, for sizing shift/scale."""
        return self.adapter.dim

    def eeg_parameters(self):
        """The EEG pathway: backbone plus channel aggregation.

        Frozen during Stage A's optional time warm-up, so the time branch and
        the readout above it train first. Aggregation is included because
        attention over a randomly initialised backbone is meaningless — leaving
        it trainable would fit weights to features about to change completely.
        """
        for module in (self.cnn, self.agg):
            yield from module.parameters()

    def parameter_groups(self) -> Dict[str, nn.Module]:
        """The blocks Stage B's warm-up freezes, addressable by name.

        Exists so the unfreeze can be PARTIAL. The default schedule releases all
        four at the boundary, but the CNN is an epoch-level feature extractor
        over 30 s windows -- largely device- and montage-independent -- and
        fine-tuning it on the internal cohorts is a plausible route to
        specialising it to those recording setups. Releasing only ``tso`` and
        ``fuse`` keeps the visual features fixed while letting the time branch
        and the layer that consumes it adapt to the sequence objective.
        """
        return {"cnn": self.cnn, "agg": self.agg,
                "tso": self.tso, "fuse": self.fuse}

    def cnn_parameters(self):
        """Parameters frozen during the Stage B warm-up: everything Stage A
        trained (backbone + aggregation + TSO + fusion)."""
        for module in (self.cnn, self.agg, self.tso, self.fuse):
            yield from module.parameters()

    def load_stage_a(self, state_dict: dict) -> None:
        """Warm-start from a Stage A checkpoint: shared submodules load by
        name; Stage A's epoch_head has no counterpart here and is skipped."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        allowed_missing = ("seq.", "seq_head.")
        allowed_unexpected = ("epoch_head.",)
        bad_missing = [k for k in missing if not k.startswith(allowed_missing)]
        bad_unexpected = [k for k in unexpected
                          if not k.startswith(allowed_unexpected)]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"Stage A -> B weight transfer mismatch. "
                f"missing={bad_missing}, unexpected={bad_unexpected}")


def reconcile_head_time_skip(model_cfg: dict, state_dict: dict,
                            label: str = "") -> dict:
    """Take ``head_time_skip`` from the CHECKPOINT, not the config.

    E5 and E6 ran before ``head_time_skip`` existed as a config key, when the
    skip was unconditional; E7 introduced the key with a default of False. Their
    ``resolved_config.yaml`` snapshots therefore do not mention it, and
    rebuilding from them silently produces a 384-wide head against a 391-wide
    checkpoint -- a shape error at best, and at worst a different model than the
    one whose results the snapshot describes. Pinning the repo configs cannot
    fix a snapshot already written, so the width stored in the weights is the
    authority.

    Returns a copy of ``model_cfg`` with the key corrected, and logs when it
    disagreed. No-op for Stage A checkpoints, which have no sequence head.
    """
    out = dict(model_cfg)
    w = state_dict.get("seq_head.dense.weight")
    if w is None:
        return out
    base = 2 * 128 + 128                # bi-GRU x2 + fusion, without the skip
    on = int(w.shape[1]) > base
    if bool(out.get("head_time_skip", False)) != on:
        log.warning("%s: config says head_time_skip=%s but the checkpoint head "
                    "is %d wide -> using %s", label or "checkpoint",
                    out.get("head_time_skip", False), int(w.shape[1]), on)
        out["head_time_skip"] = on
    return out


def build_model(cfg: dict) -> SleepStager:
    """Construct from the `model:` section of an experiment config."""
    return SleepStager(
        with_sequence=cfg.get("with_sequence", True),
        tso_mode=cfg.get("tso_mode", "full"),
        aggregation=cfg.get("aggregation", "attention"),
        embed_dim=cfg.get("embed_dim", 64),
        tso_dim=cfg.get("tso_dim", 32),
        tso_hidden=cfg.get("tso_hidden", 16),
        fusion_dim=cfg.get("fusion_dim", 128),
        gru_hidden=cfg.get("gru_hidden", 64),
        droprate_cnn=cfg.get("droprate_cnn", 0.3),
        droprate_cnn_spatial=cfg.get("droprate_cnn_spatial", 0.05),
        norm=cfg.get("norm", "none"),
        fusion_norm=cfg.get("fusion_norm", "post"),
        head_time_skip=cfg.get("head_time_skip", False),
        droprate_gru=cfg.get("droprate_gru", 0.1),
        attention_context_dim=cfg.get("attention_context_dim", 32),
    )
