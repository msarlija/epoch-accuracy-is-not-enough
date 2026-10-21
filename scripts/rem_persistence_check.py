# -*- coding: utf-8 -*-
"""Does the false-early-REM effect survive a REM-persistence rule?

The obvious objection to the early-REM result is that the spurious onsets are
isolated single epochs, so a trivial post-hoc rule -- require the REM bout to
last K epochs before it counts as onset -- would remove them without any
time-of-night input, making the time branch redundant for this purpose.

This tests it. REM onset is redefined as the first epoch of the first run of at
least K consecutive predicted-REM epochs, applied identically to both arms and
to the ground truth, and the two endpoints of sec 6.5 are recomputed for each K.
If the contrast collapses as K grows, the effect was a smoothing artifact; if it
survives, the time branch is suppressing sustained REM misdetections that
persistence cannot reach.

    python scripts/rem_persistence_check.py --arm-b none
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import external_cohorts  # noqa: E402
from eval.macrostructure import sleep_onset_index  # noqa: E402


def allowed_recs(run_dir, split):
    """rec_ids of ``split`` that belong to the config's external cohorts
    (all of them for internal splits)."""
    m = pd.read_csv(f"{run_dir}/stage_b/{split}/per_recording_metrics.csv")
    if "external" in split:
        m = m[m.cohort.isin(external_cohorts())]
    return set(m.rec_id.astype(str))

RUNS = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/runs")
REM = 4


def first_run_start(y, k, start=0):
    """Index of the first epoch of the first run of >=k consecutive REM at or
    after ``start``."""
    m = (np.asarray(y) == REM).astype(np.int8)
    m[:start] = 0
    if k <= 1:
        idx = np.flatnonzero(m)
        return int(idx[0]) if len(idx) else -1
    run = 0
    for i, v in enumerate(m):
        run = run + 1 if v else 0
        if run >= k:
            return i - k + 1
    return -1


def collect(prefix, arm, split, folds, ks, epoch_s=30.0):
    rows = []
    for f in range(folds):
        run_dir = f"{RUNS}/{prefix}_{arm}_seed0_fold{f}of10"
        d = f"{run_dir}/stage_b/predictions_{split}"
        keep = allowed_recs(run_dir, split)
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".npz") or fn[:-4] not in keep:
                continue
            z = np.load(os.path.join(d, fn))
            yt, yp = z["y_true"], z["y_pred"]
            r = {"rec_id": fn[:-4], "fold": f}
            # truth uses the SAME rule, so the comparison stays like-for-like
            # latency is measured from each hypnogram's OWN sleep onset, as in
            # eval.macrostructure; measuring from recording start (as an
            # earlier version did) is a different quantity, so K=1 would not
            # reproduce the headline count
            t_on = sleep_onset_index(yt); p_on = sleep_onset_index(yp)
            for k in ks:
                ti = first_run_start(yt, k, t_on) if t_on is not None else -1
                pi = first_run_start(yp, k, p_on) if p_on is not None else -1
                r[f"true_k{k}"] = (ti - t_on) * epoch_s / 60.0 if ti >= 0 else np.nan
                r[f"pred_k{k}"] = (pi - p_on) * epoch_s / 60.0 if pi >= 0 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default="final")
    ap.add_argument("--arm-a", default="full")
    ap.add_argument("--arm-b", default="zero")
    ap.add_argument("--split", default="external_v2")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--early", type=float, default=15.0)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 6, 10],
                    help="minimum consecutive REM epochs to count as onset "
                         "(30 s each, so 6 = 3 min)")
    args = ap.parse_args()

    A = collect(args.prefix, args.arm_a, args.split, args.folds, args.ks)
    B = collect(args.prefix, args.arm_b, args.split, args.folds, args.ks)
    m = A.merge(B, on=["rec_id", "fold"], suffixes=("_a", "_b"))
    print(f"{m.rec_id.nunique()} recordings x {args.folds} folds "
          f"= {len(m)} evaluations\n")
    print(f"  {'K':>3}{'min':>7}  | {'false early REM (pred<=15, true>15)':<44}"
          f"| {'outside 15 min':<28}")
    print(f"  {'':>3}{'':>7}  | {args.arm_b:>8}{args.arm_a:>8}{'fix':>6}"
          f"{'brk':>5}{'p':>10}   | {args.arm_b:>8}{args.arm_a:>8}{'p':>10}")
    for k in args.ks:
        t = m[f"true_k{k}_a"]
        # recordings with a defined latency under the rule in truth and both arms
        ok = t.notna() & m[f"pred_k{k}_a"].notna() & m[f"pred_k{k}_b"].notna()
        te = ok & (t <= args.early)
        fa = ok & (m[f"pred_k{k}_a"] <= args.early) & ~te
        fb = ok & (m[f"pred_k{k}_b"] <= args.early) & ~te
        g = (pd.DataFrame({"rec": m.rec_id, "a": fa, "b": fb, "ok": ok})
             .groupby("rec").agg(a=("a", "any"), b=("b", "any"), ok=("ok", "any")))
        g = g[g.ok]   # per RECORDING: fails if it fails under any fold model
        fix = int((g.b & ~g.a).sum()); brk = int((g.a & ~g.b).sum())
        p = stats.binomtest(brk, fix + brk, 0.5).pvalue if fix + brk else float("nan")
        # agreement endpoint, same rule; undefined predictions are failures
        ea = (m[f"pred_k{k}_a"] - t).abs()
        eb = (m[f"pred_k{k}_b"] - t).abs()
        oa = ok & (~(ea <= 15))
        ob = ok & (~(eb <= 15))
        h = (pd.DataFrame({"rec": m.rec_id, "a": oa, "b": ob, "ok": ok})
             .groupby("rec").agg(a=("a", "any"), b=("b", "any"), ok=("ok", "any")))
        h = h[h.ok]
        f2 = int((h.b & ~h.a).sum()); b2 = int((h.a & ~h.b).sum())
        p2 = stats.binomtest(b2, f2 + b2, 0.5).pvalue if f2 + b2 else float("nan")
        print(f"  {k:>3}{k*0.5:>7.1f}  | n={len(g):<4d}{100*g.b.mean():>6.2f}%"
              f"{100*g.a.mean():>7.2f}%{fix:>6}{brk:>5}{p:>10.2g}   | "
              f"{100*h.b.mean():>7.2f}%{100*h.a.mean():>7.2f}%{p2:>10.2g}")
    print("\n  K=1 reproduces the headline definition. If the contrast holds as K "
          "grows,\n  the effect is not removable by smoothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
