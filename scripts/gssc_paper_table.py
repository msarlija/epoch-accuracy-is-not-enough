# -*- coding: utf-8 -*-
"""Table 5 and Supplementary Table S11 of the paper: the time-naive and
time-aware models against the pretrained GSSC stager on the external cohorts.

GSSC (Hanna and Floel 2023, package version 0.0.9, networks sig_net_v1 and
gru_net_v1) scored every QC-passed EEG channel of each external recording,
without EOG (scripts/gssc_score.py). It was trained on SHHS2, CFS, WSC
and NCHSDB, so WSC is excluded. Each GSSC hypnogram is paired with the ten
fold evaluations of the same recording (subject-clustered paired bootstrap,
10,000 resamples). False early REM counts recordings with a predicted ROL of
15 min or less and a longer expert ROL, under any fold model for the two
models and once for GSSC.

    python scripts/gssc_paper_table.py                      # single fold models
    python scripts/gssc_paper_table.py --prefix ens --folds 1   # the ensembles
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.stats import paired_bootstrap  # noqa: E402

ROOT = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging"))
RUNS = f"{ROOT}/runs"
EXCLUDE = {"WSC"}


def folds(arm, ids, n=10, prefix="final"):
    fr = []
    for k in range(n):
        d = pd.read_csv(f"{RUNS}/{prefix}_{arm}_seed0_fold{k}of10/stage_b/external_v2/per_recording_metrics.csv")
        d = d[d.rec_id.isin(ids)].copy()
        d["fold"] = k
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


def pb(d, a, b):
    X = d[["key", "cohort", "subject_id"]].rename(columns={"key": "rec_id"}).assign(v=d[a].values)
    Z = d[["key"]].rename(columns={"key": "rec_id"}).assign(v=d[b].values)
    r = paired_bootstrap(X, Z, "v", n_boot=10000, seed=0)
    return r["mean_diff"], r["ci_low"], r["ci_high"], r["p_value"]


def early(d, s):
    e = (d["rol_pred_min_" + s] <= 15) & (d.rol_true_min_a > 15)
    return int(e.groupby(d.rec_id).any().sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gssc", default=f"{ROOT}/analysis/gssc_external_perrec.csv")
    ap.add_argument("--prefix", default="final", help="final for the fold models, ens for the ensemble (with --folds 1)")
    ap.add_argument("--folds", type=int, default=10, help="1 for an ensemble run directory")
    a = ap.parse_args()
    G = pd.read_csv(a.gssc).drop_duplicates("rec_id")
    G = G[~G.cohort.isin(EXCLUDE)].copy()
    G["subject_id"] = G.subject_id.astype(str)
    cols = ["rec_id", "fold", "kappa", "accuracy", "f1_macro", "rol_true_min", "rol_pred_min", "rol_ae_min"]
    A, N = folds("full", set(G.rec_id), a.folds, a.prefix), folds("zero", set(G.rec_id), a.folds, a.prefix)
    m = A[["cohort"] + cols].merge(N[cols], on=["rec_id", "fold"], suffixes=("_a", "_n"))
    m = m.merge(G[["rec_id", "subject_id", "kappa", "accuracy", "f1_macro", "rol_pred_min", "rol_ae_min"]]
                .rename(columns={c: c + "_y" for c in ("kappa", "accuracy", "f1_macro", "rol_pred_min", "rol_ae_min")}),
                on="rec_id")
    m["key"] = m.rec_id + "|f" + m.fold.astype(str)
    lat = m.dropna(subset=["rol_true_min_a", "rol_ae_min_a", "rol_ae_min_n", "rol_ae_min_y"]).copy()
    for s in ("a", "n", "y"):
        lat["w_" + s] = (lat["rol_ae_min_" + s] <= 10).astype(float)

    print(f"=== Table 5: {m.rec_id.nunique()} recordings ({m.subject_id.nunique()} subjects), {len(m)} recording-folds; "
          f"ROL over {len(lat)} evaluations of {lat.rec_id.nunique()} recordings")
    print(f"{'':24s} {'GSSC':>8s} {'naive':>8s} {'aware':>8s}   aware minus GSSC | naive minus GSSC")
    for met, sc, lab in (("kappa", 1, "Cohen's kappa"), ("accuracy", 100, "Accuracy (%)"),
                         ("f1_macro", 100, "Macro-F1 (%)")):
        d, lo, hi, p = pb(m, met + "_a", met + "_y")
        dn, lon, hin, pn = pb(m, met + "_n", met + "_y")
        print(f"{lab:24s} {sc * m[met + '_y'].mean():8.3f} {sc * m[met + '_n'].mean():8.3f} "
              f"{sc * m[met + '_a'].mean():8.3f}   {sc * d:+.4f} [{sc * lo:+.4f}, {sc * hi:+.4f}] p={p:.4f}"
              f" | {sc * dn:+.4f} [{sc * lon:+.4f}, {sc * hin:+.4f}] p={pn:.4f}")
    d, lo, hi, p = pb(lat, "w_a", "w_y")
    dn, lon, hin, pn = pb(lat, "w_n", "w_y")
    print(f"{'ROL within 10 min (%)':24s} {100 * lat.w_y.mean():8.1f} {100 * lat.w_n.mean():8.1f} "
          f"{100 * lat.w_a.mean():8.1f}   {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] p={p:.4f}"
          f" | {100 * dn:+.1f} [{100 * lon:+.1f}, {100 * hin:+.1f}] p={pn:.4f}")
    lat["ae_y"], lat["ae_a"], lat["ae_n"] = lat.rol_ae_min_y, lat.rol_ae_min_a, lat.rol_ae_min_n
    d, lo, hi, p = pb(lat, "ae_a", "ae_y")
    print(f"{'ROL abs error (min)':24s} {lat.ae_y.mean():8.1f} {lat.ae_n.mean():8.1f} {lat.ae_a.mean():8.1f}"
          f"   {d:+.1f} [{lo:+.1f}, {hi:+.1f}] p={p:.4f}")
    print(f"{'False early REM (rec.)':24s} {early(lat, 'y'):8d} {early(lat, 'n'):8d} {early(lat, 'a'):8d}"
          f"   of {lat.rec_id.nunique()}")

    print("\n=== Supplementary Table S11 (n | kappa | within-10 % | false early REM: GSSC / naive / aware)")
    for c, g in m.groupby("cohort"):
        gl = lat[lat.cohort == c]
        d, lo, hi, p = pb(g, "kappa_a", "kappa_y")
        print(f"{c:9s} n={g.rec_id.nunique():3d} | {g.kappa_y.mean():.3f} {g.kappa_n.mean():.3f} {g.kappa_a.mean():.3f}"
              f" (aware-GSSC {d:+.3f} p={p:.3f}) | {100 * gl.w_y.mean():5.1f} {100 * gl.w_n.mean():5.1f} "
              f"{100 * gl.w_a.mean():5.1f} | {early(gl, 'y')} {early(gl, 'n')} {early(gl, 'a')}")
    print(f"{'All':9s} n={m.rec_id.nunique():3d} | {m.kappa_y.mean():.3f} {m.kappa_n.mean():.3f} {m.kappa_a.mean():.3f}"
          f" | {100 * lat.w_y.mean():5.1f} {100 * lat.w_n.mean():5.1f} {100 * lat.w_a.mean():5.1f} | "
          f"{early(lat, 'y')} {early(lat, 'n')} {early(lat, 'a')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
