# -*- coding: utf-8 -*-
"""Two-stage curriculum (§3.8), replacing the old three-stage one.

Stage A — CNN-only. A separately EVALUATED model, not just a warm start:
the TSO effect is significant here and vanishes once recurrent layers are
added, so Stage A must produce reportable metrics. seq_len=1.

Stage B — sequence. Merges the old Stages 1 and 2: full-length training
sequences throughout, beginning with a short FROZEN warm-up (CNN/agg/TSO/
fusion frozen for ``warmup_epochs`` so the fresh Bi-GRU and heads don't
backprop garbage into a trained CNN), then everything unfreezes and training
continues at ``lr_after_warmup``. The warm-up lives INSIDE the stage: one
checkpoint, one early-stopping criterion, one LR schedule.

Sequence length is picked from context DURATION, not epoch count
(old 256×10 s ≈ 42.7 min ≈ 85 epochs at 30 s).
"""

import logging
import os
from typing import Optional

import torch

from models.model import build_model
from .loop import LoopConfig, train_model

log = logging.getLogger(__name__)

STAGE_A_CKPT = "stage_a.pt"
STAGE_B_CKPT = "stage_b.pt"
STAGE_B_WARMUP_CKPT = "stage_b_warmup.pt"


def run_stage_a(model_cfg: dict, train_loader, val_loader, stage_cfg: dict,
                run_dir: str):
    """Train the CNN-only model. Returns (model, history)."""
    cfg = dict(model_cfg)
    cfg["with_sequence"] = False
    model = build_model(cfg)

    # Optional TIME WARM-UP: train the time branch and the readout above it
    # first, with the EEG pathway frozen, then unfreeze.
    #
    # Applied to every arm regardless of tso_mode so the arms stay comparable —
    # all of them spend the same number of epochs with the backbone frozen, so
    # none gets more effective EEG training than another. For tso_mode 'none'
    # there is no time branch, and the warm-up simply fits the fusion and head
    # to a frozen random backbone, which is the analogous no-op.
    #
    # WORTH WATCHING: this fits P(stage | time) before the EEG can contribute,
    # which is the co-adaptation pattern that would let time crowd out EEG
    # features. Measured on E6 that risk looks small — disabling time entirely
    # cost only 0.008 kappa — but if the warm-up increases the model's reliance
    # on time, this schedule is where it would come from.
    warm = int(stage_cfg.get("time_warmup_epochs", 0) or 0)
    epoch_hook = None
    if warm > 0:
        for p_ in model.eeg_parameters():
            p_.requires_grad_(False)
        log.info("Stage A: time warm-up — EEG pathway frozen for %d epoch(s)",
                 warm)

        def epoch_hook(epoch: int, m):          # noqa: F811
            if epoch == warm:
                for q_ in m.eeg_parameters():
                    q_.requires_grad_(True)
                log.info("Stage A: time warm-up over — unfroze the EEG pathway")
                return {"lr": stage_cfg.get("lr", 1e-3)}
            return None

    loop_cfg = LoopConfig(
        epochs=stage_cfg.get("epochs", 30),
        lr=stage_cfg.get("lr", 1e-3),
        early_stop_patience=stage_cfg.get("early_stop_patience", 6),
        plateau_patience=stage_cfg.get("plateau_patience", 3),
        plateau_factor=stage_cfg.get("plateau_factor", 0.1),
        device=stage_cfg.get("device",
                             "cuda" if torch.cuda.is_available() else "cpu"),
        epoch_hook=epoch_hook,
    )
    ckpt = os.path.join(run_dir, STAGE_A_CKPT)
    history = train_model(model, train_loader, val_loader, loop_cfg, ckpt)
    return model, history


def run_stage_b(model_cfg: dict, train_loader, val_loader, stage_cfg: dict,  # noqa: C901
                run_dir: str, stage_a_ckpt: Optional[str] = None,
                short_loaders=None):
    """Train the sequence model, warm-starting from Stage A when available.

    TWO SCHEDULES are available.

    * default — one pass at full sequence length, with the backbone frozen for
      ``warmup_epochs`` (3) before unfreezing.
    * ``stage_b.gru_warmup.enabled`` — the schedule from the original
      manuscript, in two phases: the Bi-GRUs train ALONE on SHORT sequences
      (``gru_warmup.context_epochs``, e.g. 16) for ``gru_warmup.epochs`` while
      everything Stage A trained stays frozen, and only then is everything
      unfrozen for ``epochs`` at full sequence length. ``short_loaders`` must
      supply (train, val) loaders built at the short length; ``warmup_epochs``
      is ignored because the freezing happens in phase 1 instead.

      The returned history carries the phase-2 curve, with the phase-1 curve
      under the ``warmup`` key.

    Returns (model, history).
    """
    cfg = dict(model_cfg)
    cfg["with_sequence"] = True
    model = build_model(cfg)

    stage_a_ckpt = stage_a_ckpt or os.path.join(run_dir, STAGE_A_CKPT)
    if os.path.exists(stage_a_ckpt):
        state = torch.load(stage_a_ckpt, map_location="cpu")["state_dict"]
        model.load_stage_a(state)
        log.info("Stage B warm-started from %s", stage_a_ckpt)
    else:
        log.warning("No Stage A checkpoint at %s — Stage B trains from "
                    "scratch (frozen warm-up will hold random CNN weights!)",
                    stage_a_ckpt)

    warmup_epochs = int(stage_cfg.get("warmup_epochs", 3))
    lr_warmup = float(stage_cfg.get("lr_warmup", 1e-3))
    lr_after = float(stage_cfg.get("lr_after_warmup", 1e-4))

    # Stage-dependent dropout, as in the Keras pipeline. Dropout is a LAYER, not
    # a parameter, so freezing the backbone does NOT disable it -- without this
    # the frozen warm-up feeds 30% dropout noise into features that are not
    # learning. Keras runs droprate_cnn=0 while frozen and 0.1 once unfrozen,
    # with the GRU dropout falling 0.1 -> 0.01 at the same boundary.
    drop_cfg = stage_cfg.get("dropout") or {}
    warm_drop = drop_cfg.get("warmup") or {}
    fine_drop = drop_cfg.get("finetune") or {}
    if warm_drop:
        applied = model.set_dropout_rates(cnn=warm_drop.get("cnn"),
                                          gru=warm_drop.get("gru"),
                                          cnn_spatial=warm_drop.get("cnn_spatial"))
        log.info("Stage B warm-up dropout: %s", applied)

    gru_cfg = stage_cfg.get("gru_warmup") or {}
    two_phase = bool(gru_cfg.get("enabled")) and short_loaders is not None
    if gru_cfg.get("enabled") and short_loaders is None:
        raise ValueError(
            "stage_b.gru_warmup is enabled but no short-sequence loaders were "
            "passed; run_experiment must build them (see gru_warmup.context_epochs)")

    def _device():
        return stage_cfg.get("device",
                             "cuda" if torch.cuda.is_available() else "cpu")

    if two_phase:
        # ---- Phase B1: Bi-GRU only, SHORT sequences ------------------------
        # Everything Stage A trained stays frozen while the fresh recurrent
        # stack learns stage TRANSITIONS on short windows. Short sequences are
        # the point: at seq_len 16 an epoch of training contains many more
        # independent transitions than at 85, and the GRU is the only thing
        # being fitted, so it sees them at a high rate without the backbone
        # drifting underneath it.
        for p_ in model.cnn_parameters():
            p_.requires_grad_(False)
        n_warm = int(gru_cfg.get("epochs", 20))
        log.info("Stage B phase 1/2: Bi-GRU only, frozen backbone, short "
                 "sequences, %d epochs at lr=%.2e", n_warm,
                 float(gru_cfg.get("lr", lr_warmup)))
        warm_cfg = LoopConfig(
            epochs=n_warm,
            lr=float(gru_cfg.get("lr", lr_warmup)),
            early_stop_patience=int(gru_cfg.get("early_stop_patience",
                                                stage_cfg.get("early_stop_patience", 6))),
            plateau_patience=stage_cfg.get("plateau_patience", 3),
            plateau_factor=stage_cfg.get("plateau_factor", 0.1),
            device=_device(),
        )
        warm_ckpt = os.path.join(run_dir, STAGE_B_WARMUP_CKPT)
        # train_model restores the BEST weights before returning, so phase 2
        # starts from the best short-sequence GRU rather than the last one
        warm_hist = train_model(model, short_loaders[0], short_loaders[1],
                                warm_cfg, warm_ckpt)

        # ---- Phase B2: everything unfrozen, FULL-LENGTH sequences ----------
        for p_ in model.cnn_parameters():
            p_.requires_grad_(True)
        log.info("Stage B phase 2/2: all parameters unfrozen, full-length "
                 "sequences, %d epochs at lr=%.2e",
                 int(stage_cfg.get("epochs", 50)), lr_after)
        main_cfg = LoopConfig(
            epochs=stage_cfg.get("epochs", 50),
            lr=lr_after,
            early_stop_patience=stage_cfg.get("early_stop_patience", 6),
            plateau_patience=stage_cfg.get("plateau_patience", 3),
            plateau_factor=stage_cfg.get("plateau_factor", 0.1),
            device=_device(),
        )
        ckpt = os.path.join(run_dir, STAGE_B_CKPT)
        history = train_model(model, train_loader, val_loader, main_cfg, ckpt)
        history["warmup"] = warm_hist
        return model, history

    # freeze everything Stage A trained for the warm-up
    for p in model.cnn_parameters():
        p.requires_grad_(False)

    # Which blocks come back at the boundary. Default: all of them, the
    # inherited behaviour. A subset keeps the rest frozen for the whole stage --
    # e.g. ["tso", "fuse"] holds the CNN feature extractor fixed and adapts only
    # the time branch and its fusion, so the epoch-level features stay whatever
    # Stage A made them.
    groups = model.parameter_groups()
    # `is None`, not `or`: an explicit empty list means "release NOTHING, keep
    # every Stage A block frozen for the whole stage". Under `or` that falsy
    # list fell through to the default and released ALL FOUR -- the exact
    # opposite of what the config says, silently and with no error.
    _uf = stage_cfg.get("unfreeze")
    unfreeze_names = list(groups) if _uf is None else list(_uf)
    unknown = [n for n in unfreeze_names if n not in groups]
    if unknown:
        raise ValueError(f"stage_b.unfreeze names {unknown} not in "
                         f"{sorted(groups)}")
    if set(unfreeze_names) != set(groups):
        log.info("Stage B: PARTIAL unfreeze -- %s will be released at epoch %d; "
                 "%s stay frozen for the whole stage",
                 sorted(unfreeze_names),
                 warmup_epochs,
                 sorted(set(groups) - set(unfreeze_names)))

    # ...and, if asked, close the raw time skip for the same period. The frozen
    # set (cnn/agg/TSO/fusion) already covers every time route EXCEPT this one,
    # which is parameter-free and so cannot be frozen -- only switched off. Left
    # open it would be the sole trainable path from time to the logits during
    # the warm-up, bypassing the GRUs we are trying to train first.
    gate_skip = bool(stage_cfg.get("freeze_time_skip", False)) and getattr(
        model, "head_time_skip", False)
    if gate_skip:
        model.set_time_skip_active(False)
        log.info("Stage B warm-up: raw time skip CLOSED until epoch %d",
                 warmup_epochs)

    def epoch_hook(epoch: int, m):
        """Unfreeze at the warm-up boundary; returning a dict makes the loop
        rebuild its optimizer/scheduler with the reduced LR."""
        if epoch == warmup_epochs:
            g = m.parameter_groups()
            for name in unfreeze_names:
                for p in g[name].parameters():
                    p.requires_grad_(True)
            if fine_drop:
                applied = m.set_dropout_rates(
                    cnn=fine_drop.get("cnn"), gru=fine_drop.get("gru"),
                    cnn_spatial=fine_drop.get("cnn_spatial"))
                log.info("Stage B fine-tune dropout: %s", applied)
            if gate_skip:
                m.set_time_skip_active(True)
                log.info("Stage B: raw time skip OPENED")
            log.info("Stage B: warm-up over — unfroze %s, continuing at "
                     "lr=%.2e", "/".join(sorted(unfreeze_names)), lr_after)
            return {"lr": lr_after}
        return None

    loop_cfg = LoopConfig(
        epochs=stage_cfg.get("epochs", 50),
        lr=lr_warmup,
        early_stop_patience=stage_cfg.get("early_stop_patience", 6),
        plateau_patience=stage_cfg.get("plateau_patience", 3),
        plateau_factor=stage_cfg.get("plateau_factor", 0.1),
        device=stage_cfg.get("device",
                             "cuda" if torch.cuda.is_available() else "cpu"),
        epoch_hook=epoch_hook,
    )
    ckpt = os.path.join(run_dir, STAGE_B_CKPT)
    history = train_model(model, train_loader, val_loader, loop_cfg, ckpt)

    # train_model restores the best-val checkpoint, which can predate the
    # unfreeze. head_time_skip_active is a plain attribute, not a buffer, so it
    # does NOT travel in the state_dict -- the restored model would come back
    # with the skip open even if those weights were trained with it closed, and
    # the head's raw-time columns would still be at init while receiving real
    # harmonics. Match the gate to the phase the weights actually came from.
    if gate_skip:
        best = int(history.get("best_epoch", [-1])[0])
        if 0 <= best < warmup_epochs:
            model.set_time_skip_active(False)
            log.warning(
                "Stage B best epoch %d is inside the warm-up (<%d): the raw "
                "time skip stays CLOSED for evaluation, so inference matches "
                "how these weights were trained", best, warmup_epochs)
    return model, history
