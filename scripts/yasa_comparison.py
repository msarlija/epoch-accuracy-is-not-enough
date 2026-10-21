# -*- coding: utf-8 -*-
"""The open-source YASA stager on the external cohorts (Table 5 and
Supplementary Table S11 of the paper).

YASA (version 0.8.0, pre-trained classifier) scores the same prepared signal
the models receive: the first channel of each external recording that passed
quality control, at 100 Hz, without EOG, EMG or demographic covariates. Its
hypnograms are evaluated with the paper's definitions and paired with the ten
fold evaluations of the time-naive and time-aware models on the same
recordings (subject-clustered paired bootstrap).

    uv sync --extra yasa
    python scripts/yasa_comparison.py

LightGBM, which YASA uses, needs the OpenMP runtime (libgomp1 on Debian/Ubuntu).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config                           # noqa: E402
from data import manifest as mf                          # noqa: E402
from data.sampling import apply_cohort_caps              # noqa: E402
from eval.epoch_metrics import epoch_metrics             # noqa: E402
from eval.macrostructure import record_macrostructure    # noqa: E402
from eval.stats import paired_bootstrap                  # noqa: E402

ROOT = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging"))
RUNS = f"{ROOT}/runs"
OUT = f"{ROOT}/analysis/yasa_external_perrec.csv"
LABEL = {"W": 0, "WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "REM": 4}


def first_good_channel(row):
    f = row["qc_channel_flags"]
    f = json.loads(f) if isinstance(f, str) else f
    return int(np.flatnonzero(np.asarray(f) == 1)[0]) if f is not None and len(f) else 0


def stage_yasa(x, ch):
    import mne
    import yasa
    mne.set_log_level("ERROR")
    raw = mne.io.RawArray(x[:, ch][None].astype(np.float64) * 1e-6, mne.create_info(["EEG"], 100.0, "eeg"))
    hyp = yasa.SleepStaging(raw, eeg_name="EEG").predict()
    labels = np.asarray(getattr(hyp, "hypno", hyp))
    out = np.array([LABEL.get(str(s), -1) for s in labels])
    assert (out >= 0).all(), set(labels) - set(LABEL)
    return out


def score_yasa(ext, data_dir):
    rows = []
    for _, r in ext.iterrows():
        a = mf.open_arrays(data_dir, r.rec_id, mmap=True)
        y = np.asarray(a["y"], dtype=int)
        pred = stage_yasa(np.asarray(a["x"]), first_good_channel(r))
        n = min(len(y), len(pred))
        yt, yp = y[:n], np.where(y[:n] < 0, -1, pred[:n])
        m = epoch_metrics(yt[yt >= 0], yp[yt >= 0])
        mac = record_macrostructure(yt, yp)
        rows.append(dict(rec_id=r.rec_id, cohort=r.cohort, subject_id=str(r.subject_id), kappa=m["kappa"],
                         accuracy=m["accuracy"], f1_macro=m["f1_macro"], rol_pred_min=mac["rol_pred_min"],
                         rol_ae_min=mac["rol_ae_min"]))
    return pd.DataFrame(rows)


def folds(arm, ids):
    fr = []
    for k in range(10):
        d = pd.read_csv(f"{RUNS}/final_{arm}_seed0_fold{k}of10/stage_b/external_v2/per_recording_metrics.csv")
        d = d[d.rec_id.isin(ids)].copy(); d["fold"] = k; fr.append(d)
    return pd.concat(fr, ignore_index=True)


def pb(d, a, b):
    X = d[["key", "cohort", "subject_id"]].rename(columns={"key": "rec_id"}).assign(v=d[a].values)
    Z = d[["key"]].rename(columns={"key": "rec_id"}).assign(v=d[b].values)
    r = paired_bootstrap(X, Z, "v", n_boot=10000, seed=0)
    return r["mean_diff"], r["ci_low"], r["ci_high"], r["p_value"]


def early(d, s):
    """recordings with a false early onset: under any fold for the models, once for YASA"""
    e = (d["rol_pred_min_" + s] <= 15) & (d.rol_true_min_a > 15)
    return int(e.groupby(d.rec_id).any().sum())


def main():
    cfg = load_config("configs/final_full.yaml")
    man = mf.load_manifest(cfg["data_dir"])
    man = man[man.qc_pass == 1]
    ext = apply_cohort_caps(man[man.cohort.isin(cfg["external_cohorts"])], cfg.get("cohort_caps") or {},
                            seed=cfg.get("seed", 0)).reset_index(drop=True)
    if os.path.exists(OUT):
        Y = pd.read_csv(OUT)
    else:
        Y = score_yasa(ext, cfg["data_dir"])
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        Y.to_csv(OUT, index=False)
    print(f"YASA scored {len(Y)} external recordings ({OUT})")

    cols = ["rec_id", "fold", "kappa", "accuracy", "f1_macro", "rol_true_min", "rol_pred_min", "rol_ae_min"]
    A, N = folds("full", set(Y.rec_id)), folds("zero", set(Y.rec_id))
    m = A[["cohort"] + cols].merge(N[cols], on=["rec_id", "fold"], suffixes=("_a", "_n"))
    m = m.merge(Y[["rec_id", "subject_id", "kappa", "accuracy", "f1_macro", "rol_pred_min", "rol_ae_min"]]
                .rename(columns={c: c + "_y" for c in ("kappa", "accuracy", "f1_macro", "rol_pred_min", "rol_ae_min")}),
                on="rec_id")
    m["key"] = m.rec_id + "|f" + m.fold.astype(str)
    lat = m.dropna(subset=["rol_true_min_a", "rol_ae_min_a", "rol_ae_min_n", "rol_ae_min_y"]).copy()
    for s in ("a", "n", "y"):
        lat["w_" + s] = (lat["rol_ae_min_" + s] <= 10).astype(float)

    print(f"\n=== Table 5: {m.rec_id.nunique()} recordings, {len(m)} recording-folds; "
          f"ROL over {len(lat)} evaluations of {lat.rec_id.nunique()} recordings")
    print(f"{'':24s} {'YASA':>8s} {'naive':>8s} {'aware':>8s}   aware minus YASA")
    for met, sc, lab in (("kappa", 1, "Cohen's kappa"), ("accuracy", 100, "Accuracy (%)"), ("f1_macro", 100, "Macro-F1 (%)")):
        d, lo, hi, p = pb(m, met + "_a", met + "_y")
        print(f"{lab:24s} {sc * m[met + '_y'].mean():8.3f} {sc * m[met + '_n'].mean():8.3f} {sc * m[met + '_a'].mean():8.3f}"
              f"   {sc * d:+.3f} [{sc * lo:+.3f}, {sc * hi:+.3f}] p={p:.4f}")
    d, lo, hi, p = pb(lat, "w_a", "w_y")
    print(f"{'ROL within 10 min (%)':24s} {100 * lat.w_y.mean():8.1f} {100 * lat.w_n.mean():8.1f} {100 * lat.w_a.mean():8.1f}"
          f"   {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] p={p:.4f}")
    for s, lab in (("n", "naive"), ("a", "aware")):
        d, lo, hi, p = pb(m, f"kappa_{s}", "kappa_y")
        print(f"   {lab} minus YASA kappa {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    print(f"{'False early REM (rec.)':24s} {early(lat, 'y'):8d} {early(lat, 'n'):8d} {early(lat, 'a'):8d}")

    print("\n=== Supplementary Table S11 (kappa | within-10 % | false early REM: YASA / naive / aware)")
    for c, g in m.groupby("cohort"):
        gl = lat[lat.cohort == c]
        print(f"{c:9s} n={g.rec_id.nunique():3d} | {g.kappa_y.mean():.3f} {g.kappa_n.mean():.3f} {g.kappa_a.mean():.3f} | "
              f"{100 * gl.w_y.mean():5.1f} {100 * gl.w_n.mean():5.1f} {100 * gl.w_a.mean():5.1f} | "
              f"{early(gl, 'y')} {early(gl, 'n')} {early(gl, 'a')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
