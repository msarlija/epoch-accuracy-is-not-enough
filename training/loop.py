# -*- coding: utf-8 -*-
"""Train/validation loop with checkpointing, early stopping, LR plateau
schedule.

Loss is PLAIN UNWEIGHTED cross-entropy (§3.8) with ``ignore_index=-1`` for
unscored / purity-excluded epochs. No class weighting, no focal loss — the
loss must not be a moving part when the epoch/macrostructure relationship is
what's being measured.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)

IGNORE_INDEX = -1


@dataclass
class LoopConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    #: early stopping / checkpoint criterion: validation accuracy (as in the
    #: old pipeline). One criterion per stage — Stage B's warm-up is INSIDE
    #: the stage (§3.8), not a separate criterion.
    early_stop_patience: int = 6
    plateau_patience: int = 3
    plateau_factor: float = 0.1
    grad_clip_norm: Optional[float] = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    #: epoch hook: called as hook(epoch, model) before each epoch; used by
    #: Stage B to unfreeze after the warm-up
    epoch_hook: Optional[Callable] = None


def _run_batches(model, loader, criterion, optimizer, device,
                 grad_clip: Optional[float], train: bool) -> Dict[str, float]:
    model.train() if train else model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    #: stateful Stage B: one forward-GRU state per stream, carried across the
    #: consecutive windows of a recording and zeroed where a new one starts.
    #: Detached after every step (truncated backpropagation through time).
    states: Dict[str, torch.Tensor] = {}
    with ctx:
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            tso = batch["tso"].to(device, non_blocking=True)
            mask = batch["channel_mask"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            if "stream" in batch:
                sid = batch["stream"].to(device).long()
                keep = (~batch["reset"].to(device).bool()).float()[:, None]
                need = int(sid.max()) + 1
                prev = {}
                for k in ("g1", "g2"):
                    if k in states and states[k].shape[0] >= need:
                        prev[k] = states[k][sid] * keep
                out = model(x, tso, channel_mask=mask, seq_state=prev or None,
                            return_state=True)
                for k, v in out["seq_state"].items():
                    if k not in states or states[k].shape[0] < need:
                        grown = v.new_zeros(max(need, 64), v.shape[1])
                        if k in states:
                            grown[:states[k].shape[0]] = states[k]
                        states[k] = grown
                    states[k] = states[k].clone()
                    states[k][sid] = v.detach()
            else:
                out = model(x, tso, channel_mask=mask)
            logits = out["logits"]                       # (B, T, K)
            loss = criterion(logits.reshape(-1, logits.shape[-1]),
                             y.reshape(-1))

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad),
                        grad_clip)
                optimizer.step()

            with torch.no_grad():
                valid = y.reshape(-1) != IGNORE_INDEX
                n_valid = int(valid.sum())
                if n_valid:
                    pred = logits.reshape(-1, logits.shape[-1]).argmax(-1)
                    total_correct += int(
                        (pred[valid] == y.reshape(-1)[valid]).sum())
                    total_loss += float(loss) * n_valid
                    total_count += n_valid

    if total_count == 0:
        return {"loss": float("nan"), "accuracy": float("nan")}
    return {"loss": total_loss / total_count,
            "accuracy": total_correct / total_count}


def train_model(model: nn.Module, train_loader, val_loader, cfg: LoopConfig,
                checkpoint_path: str) -> Dict[str, list]:
    """Train with a single checkpoint / early-stopping criterion / LR
    schedule; restores the best weights before returning. Returns history."""
    device = torch.device(cfg.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)  # unweighted
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.plateau_factor,
        patience=cfg.plateau_patience)

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "lr": []}
    best_metric, best_epoch = -np.inf, -1

    for epoch in range(cfg.epochs):
        if cfg.epoch_hook is not None:
            # may unfreeze parameters; refresh the optimizer's view
            changed = cfg.epoch_hook(epoch, model)
            if changed:
                optimizer = torch.optim.Adam(
                    (p for p in model.parameters() if p.requires_grad),
                    lr=changed.get("lr", cfg.lr),
                    weight_decay=cfg.weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=cfg.plateau_factor,
                    patience=cfg.plateau_patience)

        tr = _run_batches(model, train_loader, criterion, optimizer, device,
                          cfg.grad_clip_norm, train=True)
        va = _run_batches(model, val_loader, criterion, optimizer, device,
                          None, train=False)
        scheduler.step(va["accuracy"])

        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["accuracy"])
        history["lr"].append(lr_now)
        log.info("epoch %03d  train loss %.4f acc %.4f | "
                 "val loss %.4f acc %.4f | lr %.2e",
                 epoch, tr["loss"], tr["accuracy"],
                 va["loss"], va["accuracy"], lr_now)

        if va["accuracy"] > best_metric:
            best_metric, best_epoch = va["accuracy"], epoch
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            torch.save({"state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_accuracy": best_metric}, checkpoint_path)
        elif epoch - best_epoch >= cfg.early_stop_patience:
            log.info("early stop at epoch %d (best %.4f @ %d)",
                     epoch, best_metric, best_epoch)
            break

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    history["best_epoch"] = [best_epoch]
    history["best_val_accuracy"] = [best_metric]
    return history
