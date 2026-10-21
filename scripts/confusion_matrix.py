# -*- coding: utf-8 -*-
"""Confusion matrices for the time ablation, and their difference.

Two arms scored on the same epochs, so the interesting object is not either
matrix but the DIFFERENCE between them: the ablation moves a small number of
epochs, and a pair of near-identical matrices printed side by side hides that.
The difference matrix is given in percentage points of the row (true-stage)
total, so a cell reads "this fraction of all true-N1 epochs moved into N2".

Counting convention. Each internal recording is scored once (it appears in
exactly one test fold). Each EXTERNAL recording is scored by all ten fold
models, so external matrices pool 10 evaluations per recording and describe
the average behaviour of the ensemble, not of one model. Stated rather than
hidden because the external epoch counts are ~10x the epoch count of the
corpus.

    python scripts/confusion_matrix.py --arm-b none
    python scripts/confusion_matrix.py --arm-b zero --split test --latex
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import external_cohorts                              # noqa: E402
from data.readers.base import STAGE_NAMES                       # noqa: E402

RUNS = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/runs")
NAMES = [STAGE_NAMES[i] for i in range(5)]


def load_pairs(prefix, arm, split, folds):
    """Concatenated (y_true, y_pred) over every recording of every fold."""
    yt, yp, keys = [], [], []
    for f in range(folds):
        run_dir = f"{RUNS}/{prefix}_{arm}_seed0_fold{f}of10"
        d = f"{run_dir}/stage_b/predictions_{split}"
        if not os.path.isdir(d):
            raise SystemExit(f"missing {d}")
        m = pd.read_csv(f"{run_dir}/stage_b/{split}/per_recording_metrics.csv")
        if "external" in split:
            m = m[m.cohort.isin(external_cohorts())]
        keep = set(m.rec_id.astype(str))
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".npz") or fn[:-4] not in keep:
                continue
            z = np.load(os.path.join(d, fn))
            t, p = z["y_true"], z["y_pred"]
            m = (t >= 0) & (p >= 0)
            yt.append(t[m]); yp.append(p[m])
            keys.append((f, fn[:-4], int(m.sum())))
    return np.concatenate(yt), np.concatenate(yp), keys


def confusion(t, p):
    c = np.zeros((5, 5), dtype=np.int64)
    np.add.at(c, (t.astype(int), p.astype(int)), 1)
    return c


def show(c, title, latex=False):
    rows = c.sum(1, keepdims=True).clip(min=1)
    pct = 100 * c / rows
    print(f"\n{title}   (n = {c.sum():,} epochs)")
    print(f"  {'true\\pred':<10}" + "".join(f"{n:>9}" for n in NAMES) +
          f"{'recall':>10}")
    for i, n in enumerate(NAMES):
        print(f"  {n:<10}" + "".join(f"{pct[i, j]:>8.1f}%" for j in range(5)) +
              f"{pct[i, i]:>9.1f}%")
    prec = 100 * np.diag(c) / c.sum(0).clip(min=1)
    print(f"  {'precision':<10}" + "".join(f"{prec[j]:>8.1f}%" for j in range(5)))
    if latex:
        print("\n  % LaTeX body")
        for i, n in enumerate(NAMES):
            print(f"  {n} & " + " & ".join(f"{pct[i, j]:.1f}" for j in range(5)) + r" \\")


def show_diff(ca, cb, label_a, label_b):
    """(arm A - arm B) in percentage points of the true-stage total."""
    ra = ca.sum(1, keepdims=True).clip(min=1)
    rb = cb.sum(1, keepdims=True).clip(min=1)
    d = 100 * ca / ra - 100 * cb / rb
    print(f"\nDIFFERENCE  {label_a} - {label_b}  (pp of each true-stage total; "
          f"+ = {label_a} sends MORE there)")
    print(f"  {'true\\pred':<10}" + "".join(f"{n:>9}" for n in NAMES) +
          f"{'d recall':>10}")
    for i, n in enumerate(NAMES):
        print(f"  {n:<10}" + "".join(f"{d[i, j]:>+8.2f} " for j in range(5)) +
              f"{d[i, i]:>+9.2f}")
    # the cell that matters for the paper's claim
    i, j = 4, 4
    print(f"\n  REM recall {label_a} {100*ca[i,i]/ra[i,0]:.1f}% vs "
          f"{label_b} {100*cb[i,i]/rb[i,0]:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default="final")
    ap.add_argument("--arm-a", default="full")
    ap.add_argument("--arm-b", default="zero")
    ap.add_argument("--splits", nargs="+", default=["test", "external_v2"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    for split in args.splits:
        lab = "INTERNAL" if split == "test" else "EXTERNAL"
        ta, pa, ka = load_pairs(args.prefix, args.arm_a, split, args.folds)
        tb, pb, kb = load_pairs(args.prefix, args.arm_b, split, args.folds)
        if len(ta) != len(tb) or not np.array_equal(ta, tb):
            print(f"WARNING {lab}: arms do not cover identical epochs "
                  f"({len(ta):,} vs {len(tb):,}); matrices are not paired")
        print(f"\n{'='*78}\n{lab}  ({len(ka)} recording-evaluations)\n{'='*78}")
        ca, cb = confusion(ta, pa), confusion(tb, pb)
        show(ca, f"{args.arm_a}  (time-aware)", args.latex)
        show(cb, f"{args.arm_b}  (time-naive)", args.latex)
        show_diff(ca, cb, args.arm_a, args.arm_b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
