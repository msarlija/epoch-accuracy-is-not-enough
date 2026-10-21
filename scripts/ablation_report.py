# -*- coding: utf-8 -*-
"""The headline ablation table: tso_full / tso_zero / tso_none across k folds.

Produces every number the manuscript's results section needs, for both stages
and both splits, plus the per-cohort breakdown and the paired statistics.

WHY NOT POOLING FOLDS NAIVELY. A natural approach pools the INTERNAL test
tables, where each pool recording is tested exactly once across the k folds, and
they assert rec_id uniqueness to guarantee it. The EXTERNAL set is different:
every fold's model scores all external recordings, so a recording appears k
times and that assert would (correctly) fire. This script keeps the two straight
by pairing on a fold-qualified key while still collapsing to distinct recordings
wherever a count is reported.

Statistics: subject-clustered, cohort-stratified paired bootstrap (subjects
resampled within stratum, pairs intact), Benjamini-Hochberg FDR within each
metric family. See eval/stats.py.

    python scripts/ablation_report.py --prefix final --folds 10
    python scripts/ablation_report.py --prefix final --out-dir $SLEEP_DATA_ROOT/analysis
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import external_cohorts                # noqa: E402
from eval.stats import fdr_bh, paired_bootstrap  # noqa: E402

MACRO = ["rol_ae_min", "swsol_ae_min"]
EPOCH = ["kappa", "accuracy", "f1_macro"]


def load(runs_root, prefix, arm, stage, split, folds):
    frames = []
    for f in range(folds):
        p = f"{runs_root}/{prefix}_{arm}_seed0_fold{f}of10/{stage}/{split}/per_recording_metrics.csv"
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        d = pd.read_csv(p)
        if "external" in split:
            # the config is the one place a cohort is in or out of the external set
            d = d[d.cohort.isin(external_cohorts())]
        d["fold"] = f
        d["distinct_rec"] = d["rec_id"].astype(str)
        # pair on a fold-qualified id: external recordings recur across folds
        d["rec_id"] = d["distinct_rec"] + f"|f{f}"
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def summary_row(d):
    return dict(kappa=d.kappa.mean(), accuracy=d.accuracy.mean(),
                f1_macro=d.f1_macro.mean(), rol_ae=d.rol_ae_min.mean(),
                rol_w15=100 * (d.rol_ae_min <= 15).mean(),
                swsol_ae=d.swsol_ae_min.mean(),
                n_rec=d.distinct_rec.nunique(), n_subj=d.subject_id.nunique())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/runs"))
    ap.add_argument("--prefix", default="final")
    ap.add_argument("--arms", nargs="+", default=["full", "zero", "none"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--stages", nargs="+", default=["stage_b", "stage_a"])
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--external-split", default="external_v2",
                    help="external evaluation directory; external_v2 is the "
                         "current external set, 'external' the superseded one")
    args = ap.parse_args()

    tables = []
    for stage in args.stages:
        for split, label in (("test", "INTERNAL"),
                             (args.external_split, "EXTERNAL")):
            A = {a: load(args.runs_root, args.prefix, a, stage, split, args.folds)
                 for a in args.arms}
            print(f"\n{'='*78}\n{stage.upper()}  {label}  ({args.folds} folds)\n{'='*78}")
            print(f"{'arm':6s} {'recs':>5s} {'subj':>5s} {'kappa':>7s} {'acc':>7s} "
                  f"{'macroF1':>8s} {'ROL AE':>7s} {'ROL w15':>8s} {'SWSOL':>7s}")
            for a in args.arms:
                r = summary_row(A[a])
                tables.append(dict(stage=stage, split=label, arm=a, **r))
                print(f"{a:6s} {r['n_rec']:5d} {r['n_subj']:5d} {r['kappa']:7.4f} "
                      f"{r['accuracy']:7.4f} {r['f1_macro']:8.4f} {r['rol_ae']:7.2f} "
                      f"{r['rol_w15']:7.1f}% {r['swsol_ae']:7.2f}")

            # paired tests, FDR within each metric family
            for fam_name, fam in (("macrostructure", MACRO), ("epoch-level", EPOCH)):
                res, lab = [], []
                for i, x in enumerate(args.arms):
                    for y in args.arms[i + 1:]:
                        for m in fam:
                            res.append(paired_bootstrap(A[x], A[y], m, n_boot=10000, seed=0))
                            lab.append((x, y, m))
                qs = fdr_bh([r["p_value"] for r in res])
                print(f"\n  paired bootstrap, {fam_name} family (negative = FIRST arm better)")
                for (x, y, m), r, q in zip(lab, res, qs):
                    star = "*" if q < 0.05 else " "
                    print(f"    {x:5s} vs {y:5s} {m:14s} d={r['mean_diff']:+8.4f} "
                          f"CI[{r['ci_low']:+8.4f},{r['ci_high']:+8.4f}] q={q:.4f}{star} "
                          f"n={r['n']}/{r['n_subjects']}subj")

            if split == "external" and len(args.arms) >= 2:
                x, y = args.arms[0], args.arms[1]
                print(f"\n  per-cohort ROL AE, {x} vs {y}")
                print(f"    {'cohort':14s} {'recs':>5s} {x:>8s} {y:>8s} {'gap':>8s}")
                helped = 0
                for c in sorted(A[x].cohort.unique()):
                    a1 = A[x][A[x].cohort == c]; a2 = A[y][A[y].cohort == c]
                    g = a1.rol_ae_min.mean() - a2.rol_ae_min.mean()
                    helped += g < 0
                    print(f"    {c:14s} {a1.distinct_rec.nunique():5d} "
                          f"{a1.rol_ae_min.mean():8.2f} {a2.rol_ae_min.mean():8.2f} {g:+8.2f}")
                print(f"    -> {x} better in {helped}/{A[x].cohort.nunique()} cohorts")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, f"{args.prefix}_ablation_{args.folds}fold.csv")
        pd.DataFrame(tables).to_csv(out, index=False)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
