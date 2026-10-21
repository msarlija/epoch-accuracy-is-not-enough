# -*- coding: utf-8 -*-
"""Learning curves and per-run summary tables.

``training.loop.train_model`` already returns a per-epoch history; this module is
what makes it inspectable after the fact — one CSV per stage plus a figure with
train and validation traces, the selected (best) epoch marked, and the final
held-out numbers printed on the panel so a curve cannot be read in isolation
from the test result it produced.
"""

import json
import logging
import os
from typing import Dict, Optional

import pandas as pd

log = logging.getLogger(__name__)

#: Per-epoch series train_model records; the scalars it appends alongside them
#: (best_epoch, best_val_accuracy) are single-element lists and are pulled out
#: separately rather than being padded into the table.
CURVE_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr")


def history_table(history: Dict[str, list]) -> pd.DataFrame:
    """Per-epoch history dict -> tidy DataFrame with an ``epoch`` column."""
    series = {k: history[k] for k in CURVE_KEYS if k in history}
    if not series:
        return pd.DataFrame()
    n = max(len(v) for v in series.values())
    df = pd.DataFrame({k: list(v) + [None] * (n - len(v))
                       for k, v in series.items()})
    df.insert(0, "epoch", range(n))
    best = history.get("best_epoch") or []
    if best:
        df["is_best_epoch"] = df["epoch"] == best[0]
    return df


def write_histories(histories: Dict[str, Dict[str, list]], run_dir: str) -> None:
    for stage, hist in histories.items():
        df = history_table(hist)
        if len(df):
            df.to_csv(os.path.join(run_dir, f"history_{stage}.csv"), index=False)


def plot_learning_curves(histories: Dict[str, Dict[str, list]], run_dir: str,
                         summary: Optional[dict] = None,
                         title: str = "") -> Optional[str]:
    """Loss + accuracy per stage, best epoch marked, final metrics annotated.

    Returns the figure path, or None if there was nothing to plot. Never raises:
    a failed plot must not lose a finished training run.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # headless; must precede pyplot
        import matplotlib.pyplot as plt
    except Exception:                   # noqa: BLE001
        log.exception("matplotlib unavailable; skipping learning curves")
        return None

    stages = [s for s in ("stage_a", "stage_b_warmup", "stage_b")
              if s in histories and len(history_table(histories[s]))]
    if not stages:
        return None

    try:
        fig, axes = plt.subplots(2, len(stages), figsize=(6.4 * len(stages), 7.2),
                                 squeeze=False)
        for col, stage in enumerate(stages):
            df = history_table(histories[stage])
            best = (histories[stage].get("best_epoch") or [None])[0]
            for row, (lo_key, hi_key, label) in enumerate(
                    [("train_loss", "val_loss", "cross-entropy loss"),
                     ("train_acc", "val_acc", "epoch accuracy")]):
                ax = axes[row][col]
                if lo_key in df:
                    ax.plot(df["epoch"], df[lo_key], "-o", ms=3, label="train")
                if hi_key in df:
                    ax.plot(df["epoch"], df[hi_key], "-s", ms=3, label="val")
                if best is not None and best >= 0:
                    ax.axvline(best, color="k", ls=":", lw=1,
                               label=f"selected (epoch {best})")
                ax.set_xlabel("epoch")
                ax.set_ylabel(label)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
                if row == 0:
                    ax.set_title(stage.replace("_", " ").upper())

            # final held-out numbers on the accuracy panel, so the curve is
            # never read without the result it led to
            if summary and stage in summary:
                lines = []
                for split in ("val", "test", "external"):
                    m = summary[stage].get(split)
                    if m:
                        lines.append(
                            f"{split}: kappa {m.get('kappa', float('nan')):.3f}  "
                            f"acc {m.get('accuracy', float('nan')):.3f}  "
                            f"(n={m.get('n_recordings', 0)})")
                if lines:
                    axes[1][col].text(
                        0.02, 0.02, "\n".join(lines), transform=axes[1][col].transAxes,
                        fontsize=8, va="bottom", ha="left", family="monospace",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.85))

        if title:
            fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        path = os.path.join(run_dir, "learning_curves.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception:                   # noqa: BLE001 — never lose a run to a plot
        log.exception("learning-curve plotting failed")
        return None


#: Metrics summarised per split. Order is the reporting order.
SUMMARY_METRICS = ("accuracy", "f1_macro", "kappa", "rol_ae_min",
                   "swsol_ae_min")


def summarize_per_recording(joined: pd.DataFrame) -> dict:
    """Per-SUBJECT means (the headline numbers), plus per-recording for reference.

    The reported value for each metric is the mean over SUBJECTS of that
    subject's mean — not the mean over recordings. The two differ wherever a
    person contributed more than one night (BOAS 124 recordings / 97 subjects,
    WSC 102 / 95), and averaging over recordings silently weights those people
    2-3x, which is the same non-independence the folds and the bootstrap already
    control for. Keeping the analysis unit consistent across the split, the
    statistics and the reported number is the point.

    Never a pooled metric over concatenated epochs: that additionally weights
    long recordings more, and is not the quantity the paired tests operate on.

    ``per_recording`` carries the unweighted recording-level means so the
    difference is visible rather than hidden.
    """
    out = {"n_recordings": int(len(joined))}
    have_subj = "subject_id" in joined.columns
    out["n_subjects"] = int(joined["subject_id"].nunique()) if have_subj \
        else int(len(joined))

    per_rec = {}
    for col in SUMMARY_METRICS:
        if col not in joined.columns:
            continue
        val = joined[col].mean()
        per_rec[col] = None if pd.isna(val) else float(val)
        if have_subj:
            # mean within subject first, then across subjects
            val = joined.groupby("subject_id")[col].mean().mean()
        out[col] = None if pd.isna(val) else float(val)
    out["per_recording"] = per_rec
    out["unit"] = "subject" if have_subj else "recording"
    return out


def write_summary(summary: dict, run_dir: str) -> str:
    path = os.path.join(run_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path
