# -*- coding: utf-8 -*-
"""Tail-failure analysis of the time ablation: does the time branch stop the
model inventing physiologically implausible REM onsets?

WHY THIS EXISTS RATHER THAN A MEAN. The pooled ROL effect (~5.5 min external) is
carried by a minority of recordings: the median per-recording improvement is
+0.15 min, about half of recordings are unchanged within a minute, and the ten
worst recordings account for ~69% of the total. A mean is a poor summary of that
and invites the objection that outliers drive it.

The hypothesis, though, was never about means -- it was that time-of-night
information stops the model predicting REM onsets that are physiologically
implausible. So the outcome is defined as a BINARY failure per recording, the
design is paired (same recordings, two arms), and the test is McNemar's exact
test on discordant pairs. That is robust to outliers, well powered, and states
exactly the claim.

Early REM is what makes this work: it is genuinely rare (0.5-0.8% of
recordings), so a predicted ROL <= 15 min is almost always wrong and a prior
over time of night can veto it. There is no equivalent for slow-wave sleep --
late SWS is common, so no time prior can rule it out -- which is why the effect
is REM-specific and no analogous SWS analysis is reported.

    python scripts/failure_rate_analysis.py --prefix final --folds 10
    python scripts/failure_rate_analysis.py --prefix final --threshold 20
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import external_cohorts  # noqa: E402

from eval.stats import paired_bootstrap  # noqa: E402


def load(runs_root, prefix, arm, split, folds):
    frames = []
    for f in range(folds):
        p = (f"{runs_root}/{prefix}_{arm}_seed0_fold{f}of10/"
             f"stage_b/{split}/per_recording_metrics.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = pd.read_csv(p)
        if "external" in split:
            d = d[d.cohort.isin(external_cohorts())]
        # rec_id repeats across folds for the external set (every fold's model
        # scores all external recordings), so a fold-qualified key is needed for
        # pairing. rec_id itself stays, for collapsing to distinct recordings.
        d["key"] = d["rec_id"].astype(str) + f"|f{f}"
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def failure_table(m, thr):
    """One row per DISTINCT recording; it fails if the model calls a false early
    REM onset in ANY of its evaluations."""
    m = m.copy()
    m["true_early"] = m.rol_true_min <= thr
    m["f_fail"] = (m.rol_pred_min_f <= thr) & ~m.true_early
    m["z_fail"] = (m.rol_pred_min_z <= thr) & ~m.true_early
    return m.groupby("rec_id").agg(f_fail=("f_fail", "any"),
                                   z_fail=("z_fail", "any"),
                                   true_early=("true_early", "any")).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/runs"))
    ap.add_argument("--prefix", default="final")
    ap.add_argument("--arm-a", default="full", help="time-aware arm")
    ap.add_argument("--arm-b", default="zero", help="parameter-matched control")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=15.0,
                    help="predicted ROL at or below this is 'implausibly early'")
    ap.add_argument("--external-split", default="external_v2",
                    help="name of the external evaluation directory. "
                         "external_v2 is the current external set ("
                         "recordings); 'external' is the superseded 8-cohort "
                         "set written before DOD-O was added, DREAMS was split "
                         "and WSC was capped.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    thr = args.threshold
    out_rows = []

    for split, label in (("test", "INTERNAL"),
                         (args.external_split, "EXTERNAL")):
        fu = load(args.runs_root, args.prefix, args.arm_a, split, args.folds)
        ze = load(args.runs_root, args.prefix, args.arm_b, split, args.folds)
        m = fu[["key", "rec_id", "cohort", "subject_id", "rol_ae_min",
                "rol_pred_min", "rol_true_min"]].merge(
            ze[["key", "rol_ae_min", "rol_pred_min"]], on="key",
            suffixes=("_f", "_z")).dropna(
            subset=["rol_pred_min_f", "rol_pred_min_z", "rol_true_min"])

        print(f"\n{'='*72}\n{label}  ({m.rec_id.nunique()} distinct recordings, "
              f"{len(m)} evaluations)\n{'='*72}")

        # ---- 1. the claim: failure rate, tested as a paired proportion
        g = failure_table(m, thr)
        b = int((g.z_fail & ~g.f_fail).sum())   # control fails, time-aware does not
        c = int((g.f_fail & ~g.z_fail).sum())   # the reverse
        p = stats.binomtest(c, b + c, 0.5).pvalue if (b + c) else float("nan")
        print(f"\nFALSE EARLY-REM (predicted ROL <= {thr:g} min, true ROL > {thr:g})")
        print(f"  {args.arm_b:5s} fails on {int(g.z_fail.sum()):4d} / {len(g)} "
              f"recordings ({100*g.z_fail.mean():5.2f}%)")
        print(f"  {args.arm_a:5s} fails on {int(g.f_fail.sum()):4d} / {len(g)} "
              f"recordings ({100*g.f_fail.mean():5.2f}%)")
        print(f"  discordant pairs: {args.arm_b}-only {b}, {args.arm_a}-only {c}")
        print(f"  McNemar exact p = {p:.3g}")
        print(f"  recordings with GENUINE early REM: {int(g.true_early.sum())} "
              f"-- sensitivity is NOT assessable at this count")
        out_rows.append(dict(split=label, metric="false_early_rem", n=len(g),
                             rate_b=g.z_fail.mean(), rate_a=g.f_fail.mean(),
                             discordant_b=b, discordant_a=c, p=p,
                             n_true_early=int(g.true_early.sum())))

        # ---- 1b. the paired agreement endpoint, same threshold
        # Same 15-minute threshold as above but on the ABSOLUTE ERROR rather
        # than the predicted value: "was this recording's latency recovered to
        # within 15 min", not "did the model invent an early onset". The two
        # answer different questions and are reported together so the tail
        # claim and the agreement claim use one threshold between them.
        w = m.copy()
        w["a_out"] = w.rol_ae_min_f > thr
        w["b_out"] = w.rol_ae_min_z > thr
        gw = w.groupby("rec_id").agg(a_out=("a_out", "any"),
                                     b_out=("b_out", "any")).reset_index()
        bw = int((gw.b_out & ~gw.a_out).sum())
        cw = int((gw.a_out & ~gw.b_out).sum())
        pw = stats.binomtest(cw, bw + cw, 0.5).pvalue if (bw + cw) else float("nan")
        print(f"\nOUTSIDE {thr:g} MIN (|predicted - true| > {thr:g})")
        print(f"  {args.arm_b:5s} {int(gw.b_out.sum()):4d} / {len(gw)} "
              f"({100*gw.b_out.mean():5.2f}%)")
        print(f"  {args.arm_a:5s} {int(gw.a_out.sum()):4d} / {len(gw)} "
              f"({100*gw.a_out.mean():5.2f}%)")
        print(f"  discordant: {args.arm_b}-only {bw}, {args.arm_a}-only {cw}")
        print(f"  McNemar exact p = {pw:.3g}")
        out_rows.append(dict(split=label, metric=f"outside_{thr:g}min", n=len(gw),
                             rate_b=gw.b_out.mean(), rate_a=gw.a_out.mean(),
                             discordant_b=bw, discordant_a=cw, p=pw))

        # ---- 2. calibration: predicted rate against the true rate
        print(f"\nCALIBRATION: how often each model predicts ROL <= {thr:g} min")
        for lab, col in ((args.arm_b, "rol_pred_min_z"), (args.arm_a, "rol_pred_min_f"),
                         ("TRUTH", "rol_true_min")):
            rate = m.groupby("rec_id")[col].apply(lambda s, c=thr: bool((s <= c).any())).mean()
            print(f"  {lab:6s} {100*rate:5.2f}%")

        # ---- 3. why a mean is the wrong summary: the distribution
        d = m.groupby("rec_id").agg(a=("rol_ae_min_f", "mean"),
                                    b=("rol_ae_min_z", "mean")).reset_index()
        d["impr"] = d.b - d.a
        n = len(d)
        better = int((d.impr > 1).sum()); worse = int((d.impr < -1).sum())
        share10 = d.impr.sort_values(ascending=False).head(10).sum() / d.impr.sum() * 100
        print(f"\nDISTRIBUTION of per-recording improvement ({args.arm_b} - {args.arm_a})")
        print(f"  mean {d.impr.mean():+7.2f}   median {d.impr.median():+7.2f}   "
              f"IQR [{d.impr.quantile(.25):+.2f}, {d.impr.quantile(.75):+.2f}]")
        print(f"  {args.arm_a} better by >1 min: {better:4d} ({100*better/n:5.1f}%)   "
              f"worse by >1 min: {worse:4d} ({100*worse/n:5.1f}%)   "
              f"within 1 min: {n-better-worse:4d} ({100*(n-better-worse)/n:5.1f}%)")
        print(f"  top 10 recordings account for {share10:.1f}% of the total improvement")
        r = paired_bootstrap(fu.rename(columns={"key": "rec_id", "rec_id": "_r"}),
                             ze.rename(columns={"key": "rec_id", "rec_id": "_r"}),
                             "rol_ae_min", n_boot=10000, seed=0)
        print(f"  mean difference (subject-clustered bootstrap): {r['mean_diff']:+.2f} min "
              f"CI[{r['ci_low']:+.2f},{r['ci_high']:+.2f}] p={r['p_value']:.4f}")

        # ---- 4. where the effect lives, binned on a MODEL-INDEPENDENT variable
        print("\nBY TRUE ROL (model-independent bins: no regression-to-the-mean artifact)")
        print(f"  {'true ROL':14s} {'n':>6s} {args.arm_b:>9s} {args.arm_a:>9s} {'improvement':>12s}")
        for lo, hi in [(0, 60), (60, 90), (90, 120), (120, 180), (180, np.inf)]:
            s = m[(m.rol_true_min >= lo) & (m.rol_true_min < hi)]
            if len(s) < 20:
                continue
            lbl = f"{lo:g}-{'inf' if np.isinf(hi) else f'{hi:g}'} min"
            print(f"  {lbl:14s} {len(s):6d} {s.rol_ae_min_z.mean():9.2f} "
                  f"{s.rol_ae_min_f.mean():9.2f} "
                  f"{s.rol_ae_min_z.mean()-s.rol_ae_min_f.mean():+12.2f}")

    if args.out:
        pd.DataFrame(out_rows).to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
