# -*- coding: utf-8 -*-
"""Ensemble numbers of the paper that the fold-level scripts do not produce.

``single``   Table 2: single fold models against their ten-fold ensemble, per
             arm (metric averaged over the ten fold models vs the mean-softmax
             ensemble), paired per recording, subject-clustered bootstrap.
``contrast`` the paired statistics of two ensembles (default time-aware vs
             time-naive): kappa, accuracy, macro-F1, ROL and SWSOL error, with
             FDR within each family (the external block of Table 3 and
             Tables 4 and S2 for the branch-removed arm, --arm-b none).
``size``     Supplementary Table S6: performance against the number of fold
             models in the ensemble.
``onsets``   Section 3.4: share of recordings whose REM onset is more than
             10 min late / early, for two ensembles.

The ensemble runs are written by scripts/make_ensemble_run.py as
runs/ens_<arm>_seed0_fold0of10, next to the fold runs runs/final_<arm>_*.

    python scripts/ensemble_report.py single|contrast|size|onsets [--arm-b zero]
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.epoch_metrics import epoch_metrics             # noqa: E402
from eval.macrostructure import record_macrostructure    # noqa: E402
from eval.stats import paired_bootstrap                  # noqa: E402

RUNS = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/runs")
NB = 10000


def _w10(d):
    d = d.copy()
    d["w10"] = np.where(d.rol_ae_min.isna(), np.nan, (d.rol_ae_min <= 10).astype(float))
    d["subject_id"] = d.subject_id.astype(str)
    return d


def fold_table(arm, st, folds=10):
    fr = []
    for f in range(folds):
        d = pd.read_csv(f"{RUNS}/final_{arm}_seed0_fold{f}of10/{st}/external_v2/per_recording_metrics.csv")
        d["fold"] = f
        fr.append(d)
    return _w10(pd.concat(fr))


def ens_table(arm, st):
    return _w10(pd.read_csv(f"{RUNS}/ens_{arm}_seed0_fold0of10/{st}/external_v2/per_recording_metrics.csv"))


def boot(A, B, col, nb=NB):
    m = A[["rec_id", "cohort", "subject_id", col]].merge(B[["rec_id", col]], on="rec_id", suffixes=("", "_b")).dropna()
    X = m[["rec_id", "cohort", "subject_id"]].assign(v=m[col].values)
    Z = m[["rec_id"]].assign(v=m[col + "_b"].values)
    r = paired_bootstrap(X, Z, "v", n_boot=nb, seed=0)
    return len(m), m[col].mean(), m[col + "_b"].mean(), r["mean_diff"], r["ci_low"], r["ci_high"], r["p_value"]


def fdr(p):
    p = np.asarray(p, float)
    o = np.argsort(p)[::-1]
    q, prev = np.empty(len(p)), 1.0
    for rank, i in zip(range(len(p), 0, -1), o):
        prev = min(prev, p[i] * len(p) / rank)
        q[i] = prev
    return q


def single(args):
    for st in ("stage_b", "stage_a"):
        for arm in ("full", args.arm_b):
            S = fold_table(arm, st, args.folds).groupby(["rec_id", "cohort", "subject_id"])[
                ["kappa", "accuracy", "f1_macro", "w10"]].mean().reset_index()
            E = ens_table(arm, st)
            for c, sc in (("kappa", 1), ("accuracy", 100), ("f1_macro", 100), ("w10", 100)):
                n, ve, vs, d, lo, hi, p = boot(E, S, c)
                print(f"{st} {arm} {c:9s} n={n} single {sc * vs:7.3f} ensemble {sc * ve:7.3f} "
                      f"d {sc * d:+.4f} [{sc * lo:+.4f}, {sc * hi:+.4f}] p={p:.4f}")


def contrast(args):
    for st in ("stage_b", "stage_a"):
        A, N = ens_table("full", st), ens_table(args.arm_b, st)
        print(f"== {st}: full vs {args.arm_b}, {len(A)} recordings")
        for fam in (("kappa", "accuracy", "f1_macro"), ("rol_ae_min", "swsol_ae_min"), ("w10",)):
            res = [(c,) + boot(A, N, c) for c in fam]
            qs = fdr([r[-1] for r in res])
            for (c, n, va, vb, d, lo, hi, p), q in zip(res, qs):
                sc = 100 if c in ("accuracy", "f1_macro", "w10") else 1
                print(f"  {c:13s} n={n:3d} {args.arm_b} {sc * vb:8.4f} full {sc * va:8.4f} d {sc * d:+8.4f} "
                      f"[{sc * lo:+.4f}, {sc * hi:+.4f}] p={p:.4f} q={q:.4f}")


def size(args):
    ids = sorted(ens_table("full", "stage_b").rec_id)
    rng = np.random.default_rng(0)
    rows = []
    for arm in ("full", args.arm_b):
        P, Y = {}, {}
        for rid in ids:
            zs = [np.load(f"{RUNS}/final_{arm}_seed0_fold{f}of10/stage_b/predictions_external_v2/{rid}.npz")
                  for f in range(args.folds)]
            n = min(len(z["proba"]) for z in zs)
            P[rid] = np.stack([z["proba"][:n] for z in zs])
            Y[rid] = zs[0]["y_true"].astype(int)[:n]
        for k in range(1, args.folds + 1):
            subs = list(itertools.combinations(range(args.folds), k))
            if len(subs) > 60:
                subs = [tuple(rng.choice(args.folds, k, replace=False)) for _ in range(60)]
            kap, w10 = [], []
            for s in subs:
                kk, ww = [], []
                for rid in ids:
                    y = Y[rid]
                    h = P[rid][list(s)].mean(0).argmax(1)
                    kk.append(epoch_metrics(y[y >= 0], h[y >= 0])["kappa"])
                    mac = record_macrostructure(y, np.where(y >= 0, h, -1))
                    if not np.isnan(mac["rol_ae_min"]):
                        ww.append(mac["rol_ae_min"] <= 10)
                kap.append(np.mean(kk)); w10.append(np.mean(ww))
            rows.append(dict(arm=arm, k=k, subsets=len(subs), kappa=np.mean(kap), w10=100 * np.mean(w10)))
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(RUNS), "analysis", "ensemble_size.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    T.to_csv(out, index=False)
    print(T.round(4).to_string(index=False))


def onsets(args):
    A, N = ens_table("full", "stage_b"), ens_table(args.arm_b, "stage_b")
    m = A.merge(N, on="rec_id", suffixes=("_a", "_n")).dropna(subset=["rol_ae_min_a", "rol_ae_min_n"])
    m["subject_id"], m["cohort"] = m.subject_id_a, m.cohort_a
    for s in ("a", "n"):
        e = m["rol_pred_min_" + s] - m.rol_true_min_a
        m["late_" + s], m["prem_" + s] = (e > 10).astype(float), (e < -10).astype(float)
    for v in ("late", "prem"):
        n, va, vb, d, lo, hi, p = boot(m.rename(columns={v + "_a": v}), m.rename(columns={v + "_n": v}), v)
        print(f"{v}: {args.arm_b} {100 * vb:.1f}%  full {100 * va:.1f}%  d {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] p={p:.4f}")
    m["bin"] = pd.cut(m.rol_true_min_a, [0, 60, 90, 120, 180, 1e9], labels=["<60", "60-90", "90-120", "120-180", ">180"])
    g = m.groupby("bin", observed=True)
    print(pd.DataFrame({"n": g.size(), f"{args.arm_b}": g.rol_ae_min_n.mean(), "full": g.rol_ae_min_a.mean()}).round(1).to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=("single", "contrast", "size", "onsets"))
    ap.add_argument("--arm-b", default="zero")
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()
    {"single": single, "contrast": contrast, "size": size, "onsets": onsets}[args.which](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
