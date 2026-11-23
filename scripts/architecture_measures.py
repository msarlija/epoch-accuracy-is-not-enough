# -*- coding: utf-8 -*-
"""Sleep-architecture measures beyond REM-onset latency, time-aware against
time-naive, and the time contrast in recordings with altered sleep architecture.

Reviewer questions (round 3): (2) do REM duration, sleep efficiency, N3 duration
and N3 timing also move with the time input; (3) does the time input help or harm
in severe sleep apnoea or fragmented sleep.

Per recording, from the wall-clock hypnograms (30 s epochs, -1 excluded):
total sleep time, sleep efficiency (TST over scored epochs), WASO, REM and N3
duration, REM and N3 share of TST, N3 share of the first third of the sleep
period, N3 and REM centroid (mean elapsed time since sleep onset of the stage's
epochs), awakenings per hour of sleep and stage transitions per hour. Each is
computed on the expert and on the predicted hypnogram; the paired contrast is the
absolute error of the time-aware minus the time-naive arm, subject-clustered
bootstrap stratified by cohort (eval.stats.paired_bootstrap), BH over the family.

    SLEEP_DATA_ROOT=~/data/neurostage/paper_root_e15c uv run python scripts/architecture_measures.py
Outputs: $SLEEP_DATA_ROOT/analysis/paper/architecture_measures/*.csv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import external_cohorts                     # noqa: E402
from eval.stats import paired_bootstrap, fdr_bh         # noqa: E402

ROOT = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging"))
RUNS = f"{ROOT}/runs"
OUT = f"{ROOT}/analysis/paper/architecture_measures"
TABLES = os.path.expanduser("~/projects/neurostage/data_tables")
EP = 0.5                                                 # minutes per epoch
W, N1, N2, N3, REM = 0, 1, 2, 3, 4
SLEEP = (N1, N2, N3, REM)


def measures(y):
    """Architecture measures of one wall-clock hypnogram (-1 excluded)."""
    y = np.asarray(y)
    valid = y >= 0
    sleep = np.isin(y, SLEEP)
    n_sleep = int(sleep.sum())
    r = dict(tst_min=n_sleep * EP, se_pct=100.0 * n_sleep / max(valid.sum(), 1),
             rem_min=float((y == REM).sum()) * EP, n3_min=float((y == N3).sum()) * EP)
    r["rem_pct"] = 100.0 * r["rem_min"] / r["tst_min"] if n_sleep else np.nan
    r["n3_pct"] = 100.0 * r["n3_min"] / r["tst_min"] if n_sleep else np.nan
    if n_sleep == 0:
        r.update(waso_min=np.nan, n3_first_third_pct=np.nan, n3_centroid_min=np.nan, rem_centroid_min=np.nan,
                 awakenings_per_h=np.nan, transitions_per_h=np.nan)
        return r
    idx = np.flatnonzero(sleep)
    on, off = idx[0], idx[-1]                              # sleep period: first to last sleep epoch
    period = y[on:off + 1]
    r["waso_min"] = float((period == W).sum()) * EP
    t = np.arange(len(period)) * EP                        # minutes since sleep onset
    third = len(period) / 3.0
    n3 = period == N3
    r["n3_first_third_pct"] = 100.0 * n3[: int(round(third))].sum() / n3.sum() if n3.any() else np.nan
    r["n3_centroid_min"] = float(t[n3].mean()) if n3.any() else np.nan
    rem = period == REM
    r["rem_centroid_min"] = float(t[rem].mean()) if rem.any() else np.nan
    s = period[period >= 0]                                # scored epochs of the sleep period
    change = s[1:] != s[:-1]
    r["awakenings_per_h"] = float(((s[1:] == W) & change).sum()) / (n_sleep * EP / 60.0)
    r["transitions_per_h"] = float(change.sum()) / (n_sleep * EP / 60.0)
    return r


MEASURES = ["tst_min", "se_pct", "waso_min", "rem_min", "rem_pct", "n3_min", "n3_pct",
            "n3_first_third_pct", "n3_centroid_min", "rem_centroid_min", "awakenings_per_h", "transitions_per_h"]
NICE = {"tst_min": "Total sleep time (min)", "se_pct": "Sleep efficiency (%)", "waso_min": "WASO (min)",
        "rem_min": "REM duration (min)", "rem_pct": "REM (% of TST)", "n3_min": "N3 duration (min)", "n3_pct": "N3 (% of TST)",
        "n3_first_third_pct": "N3 in first third of sleep period (%)", "n3_centroid_min": "N3 centroid (min after sleep onset)",
        "rem_centroid_min": "REM centroid (min after sleep onset)", "awakenings_per_h": "Awakenings per hour of sleep",
        "transitions_per_h": "Stage transitions per hour of sleep"}


def load_arm(prefix, arm, split, folds):
    rows = []
    for f in range(folds):
        d = f"{RUNS}/{prefix}_{arm}_seed0_fold{f}of10/stage_b"
        m = pd.read_csv(f"{d}/{split}/per_recording_metrics.csv")
        if "external" in split:
            m = m[m.cohort.isin(external_cohorts())]
        for r in m.itertuples():
            z = np.load(f"{d}/predictions_{split}/{r.rec_id}.npz")
            yt, yp = z["y_true"].astype(int), z["y_pred"].astype(int)
            yp = np.where(yt < 0, -1, yp)                  # unscored epochs stay unscored on both sides
            mt, mp = measures(yt), measures(yp)
            row = dict(key=f"{f}:{r.rec_id}", rec_id=r.rec_id, cohort=r.cohort, subject_id=r.subject_id, fold=f,
                       kappa=r.kappa, rol_true=r.rol_true_min, rol_pred=r.rol_pred_min, rol_ae=r.rol_ae_min)
            for k in MEASURES:
                row[f"{k}_true"], row[f"{k}_pred"] = mt[k], mp[k]
                row[f"{k}_ae"] = abs(mp[k] - mt[k])
                row[f"{k}_me"] = mp[k] - mt[k]
            rows.append(row)
    return pd.DataFrame(rows)


def boot(m, col_a, col_b):
    a = m[["key", "cohort", "subject_id"]].copy(); a["rec_id"] = m["key"]; a["v"] = m[col_a].to_numpy(float)
    b = pd.DataFrame({"rec_id": m["key"], "v": m[col_b].to_numpy(float)})
    ok = ~(np.isnan(a.v) | np.isnan(b.v))
    return paired_bootstrap(a[ok.to_numpy()], b[ok.to_numpy()], "v", n_boot=10000, seed=0)


def contrast_table(A, B, label):
    """Absolute error of every measure, aware (A) minus naive (B)."""
    m = A.merge(B, on="key", suffixes=("_a", "_b"))
    m["cohort"], m["subject_id"] = m.cohort_a, m.subject_id_a
    rows = []
    for k in MEASURES:
        bs = boot(m, f"{k}_ae_a", f"{k}_ae_b")
        rows.append(dict(measure=NICE[k], n=bs["n"], expert_mean=m[f"{k}_true_a"].mean(),
                         naive_mean=m[f"{k}_pred_b"].mean(), aware_mean=m[f"{k}_pred_a"].mean(),
                         naive_mae=m[f"{k}_ae_b"].mean(), aware_mae=m[f"{k}_ae_a"].mean(),
                         naive_me=m[f"{k}_me_b"].mean(), aware_me=m[f"{k}_me_a"].mean(),
                         d_mae=bs["mean_diff"], ci_low=bs["ci_low"], ci_high=bs["ci_high"], p=bs["p_value"]))
    t = pd.DataFrame(rows)
    t["q"] = fdr_bh(t.p.to_numpy())
    t.insert(0, "split", label)
    return t, m


def stratum_contrast(m, mask, name, n_label):
    """ROL within 10 min, ROL AE and kappa, aware minus naive, in one stratum."""
    s = m[mask].copy()
    s["w10_a"] = (s.rol_ae_a <= 10).astype(float); s["w10_b"] = (s.rol_ae_b <= 10).astype(float)
    both = s.rol_ae_a.notna() & s.rol_ae_b.notna()
    out = dict(stratum=name, n_rec=int(len(s)), n_subj=int(s.subject_id.nunique()), n_rol=int(both.sum()),
               kappa_naive=s.kappa_b.mean(), kappa_aware=s.kappa_a.mean())
    for lab, ca, cb, sub in (("w10", "w10_a", "w10_b", s[both]), ("rol_ae", "rol_ae_a", "rol_ae_b", s[both]), ("kappa", "kappa_a", "kappa_b", s)):
        bs = boot(sub, ca, cb)
        scale = 100.0 if lab == "w10" else 1.0
        out[f"{lab}_naive"] = scale * sub[cb].mean(); out[f"{lab}_aware"] = scale * sub[ca].mean()
        out[f"d_{lab}"] = scale * bs["mean_diff"]; out[f"d_{lab}_lo"] = scale * bs["ci_low"]; out[f"d_{lab}_hi"] = scale * bs["ci_high"]; out[f"p_{lab}"] = bs["p_value"]
    out["n_label"] = n_label
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    # ---- external ensembles, internal fold models
    ext_a, ext_b = load_arm("ens", "full", "external_v2", 1), load_arm("ens", "zero", "external_v2", 1)
    int_a, int_b = load_arm("final", "full", "test", 10), load_arm("final", "zero", "test", 10)
    t_ext, m_ext = contrast_table(ext_a, ext_b, "external (ensembles, 308)")
    t_int, m_int = contrast_table(int_a, int_b, "internal (fold models, 973)")
    T = pd.concat([t_ext, t_int], ignore_index=True)
    T.to_csv(f"{OUT}/architecture_contrast.csv", index=False)
    m_ext.to_csv(f"{OUT}/per_recording_external.csv", index=False)
    print("=== Architecture measures, absolute error aware vs naive (Δ = aware − naive MAE)\n")
    print(T.round(2).to_string(index=False))

    # ---- strata: AHI (WSC external; CFS, HomePAP internal), DOD-O, fragmentation tertiles
    rows = []
    wsc = pd.read_csv(f"{TABLES}/wsc-dataset-0.3.0.csv", usecols=["wsc_id", "wsc_vst", "ahi"])
    ids = m_ext.rec_id_a.str.extract(r"WSC_(\d+)_v(\d)").astype(float)
    m_ext["ahi"] = pd.merge(ids.rename(columns={0: "wsc_id", 1: "wsc_vst"}), wsc, how="left", on=["wsc_id", "wsc_vst"])["ahi"].to_numpy()
    for name, mask in (("WSC, AHI < 5", m_ext.ahi < 5), ("WSC, AHI 5–15", (m_ext.ahi >= 5) & (m_ext.ahi < 15)),
                       ("WSC, AHI ≥ 15", m_ext.ahi >= 15), ("WSC, AHI ≥ 30", m_ext.ahi >= 30),
                       ("DOD-O (all OSA patients)", m_ext.cohort == "DODO"), ("DREAMS-P (sleep-disorder patients)", m_ext.cohort == "DREAMS-P"),
                       ("DREAMS-S (healthy)", m_ext.cohort == "DREAMS-S")):
        rows.append(stratum_contrast(m_ext, mask.to_numpy(), name, "external"))
    cfs = pd.read_csv(f"{TABLES}/cfs_import.csv", usecols=["nsrrid", "ahi"]); hp = pd.read_csv(f"{TABLES}/homepap_import.csv", usecols=["nsrrid", "ahi"])
    ahi_int = {**{f"CFS_{int(r.nsrrid)}": r.ahi for r in cfs.itertuples()}, **{f"HPAP_{int(r.nsrrid)}": r.ahi for r in hp.itertuples()}}
    m_int["ahi"] = m_int.rec_id_a.map(ahi_int)
    for name, lo, hi in (("CFS + HomePAP, AHI < 5", 0, 5), ("CFS + HomePAP, AHI 5–15", 5, 15), ("CFS + HomePAP, AHI 15–30", 15, 30), ("CFS + HomePAP, AHI ≥ 30", 30, 1e9)):
        rows.append(stratum_contrast(m_int, ((m_int.ahi >= lo) & (m_int.ahi < hi)).to_numpy(), name, "internal"))
    # fragmentation of the expert hypnogram: tertiles of awakenings per hour and of transitions per hour, pooled
    for key, lab in (("awakenings_per_h_true_a", "awakenings/h"), ("transitions_per_h_true_a", "transitions/h")):
        for M, split in ((m_ext, "external"), (m_int, "internal")):
            q = M[key].quantile([1 / 3, 2 / 3]).to_numpy()
            for name, mask in ((f"{split}, expert {lab} lowest tertile (< {q[0]:.1f})", M[key] < q[0]),
                               (f"{split}, expert {lab} middle tertile", (M[key] >= q[0]) & (M[key] < q[1])),
                               (f"{split}, expert {lab} highest tertile (≥ {q[1]:.1f})", M[key] >= q[1])):
                rows.append(stratum_contrast(M, mask.to_numpy(), name, split))
    for name, M, thr in (("internal, expert ROL ≤ 30 min", m_int, 30), ("internal, expert ROL ≤ 15 min", m_int, 15), ("internal, expert ROL ≤ 45 min", m_int, 45)):
        rows.append(stratum_contrast(M, (M.rol_true_a <= thr).to_numpy(), name, "internal"))
    S = pd.DataFrame(rows)
    S.to_csv(f"{OUT}/strata_contrast.csv", index=False)
    cols = ["stratum", "n_rec", "n_subj", "n_rol", "w10_naive", "w10_aware", "d_w10", "d_w10_lo", "d_w10_hi", "p_w10",
            "rol_ae_naive", "rol_ae_aware", "d_rol_ae", "p_rol_ae", "kappa_naive", "kappa_aware", "d_kappa", "d_kappa_lo", "d_kappa_hi", "p_kappa"]
    print("\n=== Time contrast by stratum (within 10 = % of recordings with ROL within 10 min)\n")
    print(S[cols].round(3).to_string(index=False))
    # ---- context for the 10-min tolerance: spread of the expert ROL
    e = m_ext.rol_true_a.dropna()
    print(f"\nExpert ROL, external: median {e.median():.0f} min, IQR {e.quantile(.25):.0f}–{e.quantile(.75):.0f}, SD {e.std():.0f}, n {len(e)}")
    e = m_int.rol_true_a.dropna()
    print(f"Expert ROL, internal: median {e.median():.0f} min, IQR {e.quantile(.25):.0f}–{e.quantile(.75):.0f}, SD {e.std():.0f}, n {len(e)}")
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
