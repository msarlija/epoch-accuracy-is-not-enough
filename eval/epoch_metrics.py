# -*- coding: utf-8 -*-
"""Epoch-level metrics (§3.13).

Accuracy, macro-F1, Cohen's κ, per-class F1. PER-COHORT is the primary
report; pooled is secondary (pooled numbers are a weighted average that
N2-heavy datasets inflate). Also stratified by montage family and by channel
count.
"""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)

from data.readers.base import STAGE_NAMES, VALID_STAGES

LABELS = list(VALID_STAGES)


def epoch_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Metrics over valid (label >= 0) epochs of one or many recordings."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    keep = np.isin(y_true, LABELS)
    y_true, y_pred = y_true[keep], y_pred[keep]
    if y_true.size == 0:
        return {"n_epochs": 0}

    out = {
        "n_epochs": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS,
                                   average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=LABELS)),
    }
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None,
                         zero_division=0)
    for lab, f1 in zip(LABELS, per_class):
        out[f"f1_{STAGE_NAMES[lab]}"] = float(f1)
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray,
              normalize: str = "true") -> np.ndarray:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    keep = np.isin(y_true, LABELS)
    return confusion_matrix(y_true[keep], y_pred[keep], labels=LABELS,
                            normalize=normalize)


def stratified_epoch_metrics(records: List[dict]) -> pd.DataFrame:
    """``records``: dicts with keys y_true, y_pred, cohort, montage_family,
    n_channels, rec_id.

    Returns a long DataFrame with one row per stratum: per-cohort strata
    first (primary), then montage-family and channel-count strata, then the
    pooled row (secondary).
    """
    df_rows = []

    def add(stratum_type: str, stratum: str, recs: Sequence[dict]):
        yt = np.concatenate([np.asarray(r["y_true"]).ravel() for r in recs])
        yp = np.concatenate([np.asarray(r["y_pred"]).ravel() for r in recs])
        row = {"stratum_type": stratum_type, "stratum": stratum,
               "n_recordings": len(recs)}
        row.update(epoch_metrics(yt, yp))
        df_rows.append(row)

    for key, stratum_type in (("cohort", "cohort"),
                              ("montage_family", "montage_family"),
                              ("n_channels", "n_channels")):
        values = sorted({str(r[key]) for r in records})
        for v in values:
            add(stratum_type, v, [r for r in records if str(r[key]) == v])

    add("pooled", "all", records)
    return pd.DataFrame(df_rows)


def per_recording_epoch_metrics(records: List[dict]) -> pd.DataFrame:
    """One row per recording — the unit of statistical analysis (§3.15)."""
    rows = []
    for r in records:
        row = {"rec_id": r["rec_id"], "cohort": r["cohort"],
               # falls back to rec_id, which makes every recording its own
               # cluster -- i.e. exactly the old recording-level behaviour
               "subject_id": r.get("subject_id", r["rec_id"]),
               "montage_family": r["montage_family"],
               "n_channels": r["n_channels"]}
        row.update(epoch_metrics(r["y_true"], r["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def stage_ranking_metrics(records: List[dict]) -> pd.DataFrame:
    """One-vs-rest AUROC and AUPRC per stage, from the logged probabilities.

    Threshold-free: measures how well the model RANKS epochs of each stage,
    independently of where argmax happens to put the decision boundary. That
    matters most for the minority stages — N1 is 3-15% of epochs depending on
    cohort and N3 and REM are 10-15%, and on a 10% class a mediocre model can
    still post a high AUROC.

    AUPRC is reported alongside because it is the one sensitive to prevalence:
    its baseline is the class rate itself, so AUPRC 0.30 on a 10% class is a
    real signal while AUROC 0.80 on the same class may not be.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = []
    for r in records:
        proba = r.get("proba")
        if proba is None:
            continue
        y = np.asarray(r["y_true"])
        keep = y >= 0
        if keep.sum() < 50:
            continue
        y, p = y[keep], np.asarray(proba)[keep]
        row = {"rec_id": r["rec_id"], "cohort": r.get("cohort", ""),
               "subject_id": r.get("subject_id", r["rec_id"])}
        for s, name in enumerate(("W", "N1", "N2", "N3", "REM")):
            pos = (y == s).astype(int)
            row[f"prevalence_{name}"] = float(pos.mean())
            if pos.sum() == 0 or pos.sum() == len(pos):
                row[f"auroc_{name}"] = np.nan
                row[f"auprc_{name}"] = np.nan
                continue
            row[f"auroc_{name}"] = float(roc_auc_score(pos, p[:, s]))
            row[f"auprc_{name}"] = float(average_precision_score(pos, p[:, s]))
        rows.append(row)
    return pd.DataFrame(rows)
