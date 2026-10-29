# -*- coding: utf-8 -*-
"""Supplementary Table S12: confusion matrices of the time-aware ten-fold ensemble and of GSSC on
the external recordings of Table 5 (WSC excluded, since GSSC was trained on it), on the same epochs:
expert-scored epochs that GSSC scores (its last partial epoch is unscored).

Rows are the expert stage, columns the predicted stage, in percent of the expert stage (the
diagonal is the recall). Precision and F1 per stage are pooled over all epochs. Needs the GSSC
stages saved by gssc_score.py (analysis/gssc_hyp/) and the ensemble run of make_ensemble_run.py.

    python scripts/gssc_confusion.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging"))
RUNS = f"{ROOT}/runs"
HYP = f"{ROOT}/analysis/gssc_hyp"
NAMES = ["W", "N1", "N2", "N3", "REM"]


def main() -> int:
    G = pd.read_csv(f"{ROOT}/analysis/gssc_external_perrec.csv")
    G = G[G.cohort != "WSC"]
    ys, ens, gssc = [], [], []
    for rid in G.rec_id:
        z = np.load(f"{RUNS}/ens_full_seed0_fold0of10/stage_b/predictions_external_v2/{rid}.npz")
        g = np.load(f"{HYP}/{rid}.npz")["hyp"].astype(int)
        y, e = z["y_true"].astype(int), z["proba"].argmax(1)
        n = min(len(y), len(e), len(g))
        k = y[:n] >= 0
        ys.append(y[:n][k]); ens.append(e[:n][k]); gssc.append(g[:n][k])
    y = np.concatenate(ys)
    out = {}
    for name, p in (("ensemble", np.concatenate(ens)), ("GSSC", np.concatenate(gssc))):
        cm = confusion_matrix(y, p, labels=range(5)).astype(float)
        rec, prec = np.diag(cm) / cm.sum(1), np.diag(cm) / cm.sum(0)
        out[name] = dict(rows=100 * cm / cm.sum(1, keepdims=True), precision=100 * prec, recall=100 * rec,
                         f1=100 * 2 * prec * rec / (prec + rec))
    n_stage = np.bincount(y, minlength=5)
    print(f"{len(G)} recordings, {len(y)} epochs; expert epochs per stage:", dict(zip(NAMES, n_stage)))
    for name, o in out.items():
        print(f"\n== {name}: row % (rows expert, columns predicted)")
        print(pd.DataFrame(o["rows"], index=NAMES, columns=NAMES).round(1).to_string())
        print(pd.DataFrame({k: o[k] for k in ("recall", "precision", "f1")}, index=NAMES).T.round(1).to_string())
    print("\n== LaTeX rows: expert stage & n & ensemble W..REM & GSSC W..REM")
    for i, s in enumerate(NAMES):
        cells = [f"\\textbf{{{v:.1f}}}" if j == i else f"{v:.1f}" for j, v in enumerate(out["ensemble"]["rows"][i])] + \
                [f"\\textbf{{{v:.1f}}}" if j == i else f"{v:.1f}" for j, v in enumerate(out["GSSC"]["rows"][i])]
        print(f"    {s} & {n_stage[i]:,} & " + " & ".join(cells) + " \\\\")
    for key, lab in (("precision", "Precision"), ("f1", "F1")):
        print(f"    {lab} & & " + " & ".join(f"{v:.1f}" for v in out["ensemble"][key]) + " & " +
              " & ".join(f"{v:.1f}" for v in out["GSSC"][key]) + " \\\\")
    return 0


if __name__ == "__main__":
    sys.exit(main())
