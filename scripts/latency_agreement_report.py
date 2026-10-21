# -*- coding: utf-8 -*-
"""REM-onset agreement of an arm pair: tolerance sweep, per-cohort tables and
the per-recording collapse that the manuscript reports.

Complements ablation_report.py (epoch metrics, mean latency errors) and
failure_rate_analysis.py (false early REM). Everything here is paired on the
fold-qualified recording key and clustered on the subject, as in those scripts.

    python scripts/latency_agreement_report.py --arm-b zero
    python scripts/latency_agreement_report.py --arm-b none --stages stage_b
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import external_cohorts            # noqa: E402
from eval.stats import paired_bootstrap        # noqa: E402

RUNS = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/runs")
NICE = {"DODH": "DOD-H", "DODO": "DOD-O"}


def recs(prefix, arm, stage, split, folds):
    fr = []
    for f in range(folds):
        p = f"{RUNS}/{prefix}_{arm}_seed0_fold{f}of10/{stage}/{split}/per_recording_metrics.csv"
        d = pd.read_csv(p)
        if "external" in split:
            d = d[d.cohort.isin(external_cohorts())]
        d = d.assign(fold=f, key=[f"{f}:{r}" for r in d.rec_id])
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


def boot(m, col_a, col_b, fn):
    a = m[["key", "cohort", "subject_id"]].copy(); a["rec_id"] = m["key"]; a["v"] = fn(m[col_a])
    b = pd.DataFrame({"rec_id": m["key"], "v": fn(m[col_b])})
    return paired_bootstrap(a, b, "v", n_boot=10000, seed=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default="final")
    ap.add_argument("--arm-a", default="full")
    ap.add_argument("--arm-b", default="zero")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--stages", nargs="+", default=["stage_b", "stage_a"])
    ap.add_argument("--splits", nargs="+", default=["external_v2", "test"])
    ap.add_argument("--ks", nargs="+", type=float, default=[5, 10, 15, 20, 30, 60])
    ap.add_argument("--early", type=float, default=15.0)
    args = ap.parse_args()
    A_, B_ = args.arm_a, args.arm_b

    for stage in args.stages:
        for split in args.splits:
            A = recs(args.prefix, A_, stage, split, args.folds)
            B = recs(args.prefix, B_, stage, split, args.folds)
            cols = ["key", "rec_id", "cohort", "subject_id", "fold", "kappa", "rol_ae_min",
                    "rol_true_min", "rol_pred_min"]
            m = A[cols].merge(B[["key", "kappa", "rol_ae_min", "rol_pred_min"]], on="key",
                              suffixes=("_a", "_b"))
            lat = m.dropna(subset=["rol_ae_min_a", "rol_ae_min_b"])
            print("=" * 78)
            print(f"{stage.upper()}  {split}  {A_} vs {B_}: {m.rec_id.nunique()} recordings, "
                  f"{lat.rec_id.nunique()} with a defined latency under both "
                  f"({lat.subject_id.nunique()} subjects, {len(lat)} evaluations)")
            print("=" * 78)
            print(f"\nTOLERANCE SWEEP  (per-evaluation rate, diff {A_} - {B_} in pp, subject-clustered)")
            print(f"  {'k':>4} {A_:>8} {B_:>8} {'diff':>7} {'CI':>18} {'p':>8}")
            for k in args.ks:
                r = boot(lat, "rol_ae_min_a", "rol_ae_min_b", lambda s: (s <= k).astype(float))
                print(f"  {k:4.0f} {100*(lat.rol_ae_min_a<=k).mean():8.1f} {100*(lat.rol_ae_min_b<=k).mean():8.1f} "
                      f"{100*r['mean_diff']:+7.2f} [{100*r['ci_low']:+6.2f},{100*r['ci_high']:+6.2f}] {r['p_value']:8.4f}")
            r = boot(lat, "rol_ae_min_a", "rol_ae_min_b", lambda s: s)
            print(f"  mean ROL AE {A_} {lat.rol_ae_min_a.mean():.2f}  {B_} {lat.rol_ae_min_b.mean():.2f}  "
                  f"diff {r['mean_diff']:+.2f} [{r['ci_low']:+.2f},{r['ci_high']:+.2f}] p={r['p_value']:.4f}")

            print(f"\nPER COHORT  (kappa over all recordings; ROL over defined-latency recordings; within-10 diff {A_} - {B_})")
            print(f"  {'cohort':12} {'n':>4} {'k_'+B_:>8} {'k_'+A_:>8} {'AE_'+B_:>7} {'AE_'+A_:>7} {'w10_'+B_:>8} {'w10_'+A_:>8} {'diff':>7} {'CI':>16}")
            for coh, g in lat.groupby("cohort"):
                gk = m[m.cohort == coh]
                r = boot(g, "rol_ae_min_a", "rol_ae_min_b", lambda s: (s <= 10).astype(float))
                print(f"  {NICE.get(coh, coh):12} {g.rec_id.nunique():4d} {gk.kappa_b.mean():8.3f} {gk.kappa_a.mean():8.3f} "
                      f"{g.rol_ae_min_b.mean():7.1f} {g.rol_ae_min_a.mean():7.1f} "
                      f"{100*(g.rol_ae_min_b<=10).mean():8.1f} {100*(g.rol_ae_min_a<=10).mean():8.1f} "
                      f"{100*r['mean_diff']:+7.1f} [{100*r['ci_low']:+5.1f},{100*r['ci_high']:+5.1f}]")

            if stage != "stage_b":
                continue
            # one row per distinct recording (mean over its evaluations)
            pr = lat.groupby("rec_id").agg(a=("rol_ae_min_a", "mean"), b=("rol_ae_min_b", "mean"),
                                           true=("rol_true_min", "first"),
                                           in_a=("rol_ae_min_a", lambda s: bool((s <= 10).all())),
                                           in_b=("rol_ae_min_b", lambda s: bool((s <= 10).all())),
                                           early_b=("rol_pred_min_b", lambda s: bool((s <= args.early).any())),
                                           early_a=("rol_pred_min_a", lambda s: bool((s <= args.early).any())))
            imp = pr.b - pr.a
            print(f"\nPER RECORDING ({len(pr)} recordings, mean over folds)")
            for lab, col in ((A_, pr.a), (B_, pr.b)):
                print(f"  {lab:5s} mean {col.mean():6.2f}  median {col.median():6.2f}  p75 {col.quantile(.75):6.2f}")
            print(f"  improvement ({B_} - {A_}): mean {imp.mean():+.2f} median {imp.median():+.2f} "
                  f"IQR [{imp.quantile(.25):+.2f}, {imp.quantile(.75):+.2f}]")
            print(f"  {A_} better by >1 min: {(imp>1).sum()} ({100*(imp>1).mean():.1f}%)  worse by >1 min: {(imp<-1).sum()} "
                  f"({100*(imp<-1).mean():.1f}%)  within 1 min: {(imp.abs()<=1).sum()} ({100*(imp.abs()<=1).mean():.1f}%)")
            top = imp.sort_values(ascending=False).head(10).sum()
            print(f"  top 10 recordings: {100*top/imp.sum():.0f}% of the total improvement")
            gained = pr.in_a & ~pr.in_b; lost = pr.in_b & ~pr.in_a
            corrected = pr.early_b & ~pr.early_a & (pr.true > args.early)
            print(f"  within-10 in every fold: gained {gained.sum()}, lost {lost.sum()}")
            print(f"  corrected early-REM recordings: {corrected.sum()}; of these within-10 in every fold: "
                  f"{(corrected & pr.in_a).sum()} (true ROL {pr.true[corrected & pr.in_a].min():.0f} to "
                  f"{pr.true[corrected & pr.in_a].max():.0f} min); corrected among gained: {(corrected & gained).sum()}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
