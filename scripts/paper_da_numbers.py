# -*- coding: utf-8 -*-
"""Adaptation numbers at the level of the single fold models (every fold model
adapted and scored on its own, pooled over the ten). The paper reports the
ensemble values (scripts/ensemble_da_numbers.py); this script supplies the
class-prior comparison of Supplementary Table S4, which is defined per fold
model, and the fold-level counterpart of every ensemble number.

Reads the per-recording outputs of the adaptation runs in ``reproduce.sh``:

* ``analysis/domain_adaptation_da12_perrec.csv``       DA-G and the single
  anchor DA-A (Supplementary Table S12)
* ``analysis/domain_adaptation_da3_perrec.csv``        DA-MC (clusters with at
  least 50 epochs, missing clusters dropped from the fit)
* ``analysis/domain_adaptation_da3_oracle_perrec.csv`` DA-MC with expert labels
* ``analysis/da4/fold*_da_o/``                         DA-O (supervised head)

and the run directories the adaptation wrote for the REM-onset endpoints.
All values are prior-corrected and pooled over the ten fold models, with the
subject-clustered paired bootstrap. Writes ``analysis/paper/da_runs/
fold_kcurve.csv`` (the fold-model counterpart of Figure 4; the figure itself
reads current_kcurve.csv from ensemble_da_numbers.py).

    python scripts/paper_da_numbers.py
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import external_cohorts            # noqa: E402
from eval.stats import paired_bootstrap        # noqa: E402

ROOT = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging"))
A = f"{ROOT}/analysis"
RUNS = f"{ROOT}/runs"
MAN = f"{ROOT}/prepared/fs100_band0.2-30/manifest.csv"
C1 = "da1/{p}/all+prior/affine/c0.7"
C2 = "da2/{p}/pred-conf+prior/affine/c0.7"
C3 = "da2/{p}/pred-conf+prior/multicluster-drop/c0.7"
CO = "da2/{p}/true+prior/multicluster-drop/c0.7"
KS = (("loso1", "1"), ("loso2", "2"), ("loso5", "5"), ("loso10", "10"), ("loso15", "15"), ("loso", "all"))
NB = 10000
#: the expert-label comparison and the one-subject DA-MC details use 5,000 resamples
NB_S9 = 5000


def subj_map():
    m = pd.read_csv(MAN).drop_duplicates("rec_id")
    return dict(zip(m.rec_id.astype(str), m.subject_id.astype(str)))


SUBJ = subj_map()


def load(path, cohorts):
    d = pd.read_csv(path)
    d = d[d.cohort.isin(cohorts)].copy()
    d["rec_id"] = d.rec_id.astype(str)
    base = d.rec_id.str.split("|").str[0]
    d["subject_id"] = base.map(SUBJ).fillna(base)
    return d


def boot(d, a, b, nb=None):
    """paired bootstrap of column a minus column b, subject-clustered"""
    if len(d) == 0:
        return dict(d=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=0)
    X = d[["rec_id", "cohort", "subject_id"]].copy(); X["m"] = d[a].values
    Y = d[["rec_id"]].copy(); Y["m"] = d[b].values
    r = paired_bootstrap(X, Y, "m", n_boot=nb or NB, seed=0)
    return dict(d=r["mean_diff"], lo=r["ci_low"], hi=r["ci_high"], p=r["p_value"], n=len(d))


def delta(d, metric, scale=1.0, nb=None):
    r = boot(d, metric, metric + "_base", nb)
    return {k: (v * scale if k in ("d", "lo", "hi") else v) for k, v in r.items()}


def fmt(r, nd=4):
    return f"{r['d']:+.{nd}f} [{r['lo']:+.{nd}f}, {r['hi']:+.{nd}f}] p={r['p']:.4f} n={r['n']}"


def cond(d, c):
    return d[d.condition == c]


def per_cohort(d, metric="kappa"):
    return {coh: delta(g, metric) for coh, g in d.groupby("cohort")}


def within10(d, nb=None):
    """within-10-min ROL change from the per-recording table"""
    x = d.dropna(subset=["rol_ae_min", "rol_ae_min_base"]).copy()
    x["w"] = (x.rol_ae_min <= 10).astype(float)
    x["w_base"] = (x.rol_ae_min_base <= 10).astype(float)
    return delta(x, "w", 100, nb)


def clusters(d):
    """share of recording-folds aligned on 3, 2 and 1 clusters"""
    v = d.n_clusters.value_counts(normalize=True)
    return "  ".join(f"{k} clusters {100 * v.get(k, 0):.1f}%" for k in (3, 2, 1))


def load_rundirs(pattern, cohorts):
    fr = []
    for p in sorted(glob.glob(pattern)):
        f = int(re.search(r"fold(\d+)", p).group(1))
        x = pd.read_csv(p); x = x[x.cohort.isin(cohorts)].copy(); x["fold"] = f
        fr.append(x)
    d = pd.concat(fr, ignore_index=True)
    d["key"] = d.rec_id.astype(str) + "|f" + d.fold.astype(str)
    d["subject_id"] = d.rec_id.astype(str).map(SUBJ).fillna(d.rec_id.astype(str))
    return d


def rol_endpoints(a, base):
    """within-10 change (paired, subject-clustered) and false early REM under
    any fold, on recording-folds with a defined true latency"""
    m = a[["key", "rec_id", "cohort", "subject_id", "rol_ae_min", "rol_pred_min", "rol_true_min"]].merge(
        base[["key", "rol_ae_min", "rol_pred_min"]], on="key", suffixes=("", "_base"))
    m = m.dropna(subset=["rol_true_min"])
    m["w"] = (m.rol_ae_min <= 10).astype(float); m["w_base"] = (m.rol_ae_min_base <= 10).astype(float)
    X = m[["key", "cohort", "subject_id"]].rename(columns={"key": "rec_id"}); X["m"] = m.w.values
    Y = m[["key"]].rename(columns={"key": "rec_id"}); Y["m"] = m.w_base.values
    r = paired_bootstrap(X, Y, "m", n_boot=NB, seed=0)
    m["e"] = (m.rol_pred_min <= 15) & (m.rol_true_min > 15)
    m["e_base"] = (m.rol_pred_min_base <= 15) & (m.rol_true_min > 15)
    rec = m.groupby("rec_id")[["e", "e_base"]].any()
    return dict(w10=100 * m.w.mean(), w10_base=100 * m.w_base.mean(), d=100 * r["mean_diff"],
                lo=100 * r["ci_low"], hi=100 * r["ci_high"], p=r["p_value"],
                early=int(rec.e.sum()), early_base=int(rec.e_base.sum()), n_rec=len(rec), m=m)


def main():
    coh = external_cohorts()
    print("cohorts:", coh)
    M = load(f"{A}/domain_adaptation_da12_perrec.csv", coh)
    M3 = load(f"{A}/domain_adaptation_da3_perrec.csv", coh)
    src = {"DA-G": (M, C1), "DA-A": (M, C2), "DA-MC": (M3, C3)}

    print("\n=== POOLED, leave-one-subject-out (fold models; Table 6 reports the ensembles)")
    pooled = {}
    for lab, (D, c) in src.items():
        g = cond(D, c.format(p="loso")); pooled[lab] = g
        pk = per_cohort(g); worst = min(pk.items(), key=lambda kv: kv[1]["d"])
        print(f"{lab}: kappa {fmt(delta(g, 'kappa'))}")
        print(f"      acc pp {fmt(delta(g, 'accuracy', 100), 2)}  F1 pp {fmt(delta(g, 'f1_macro', 100), 2)}")
        print(f"      ROL AE min {fmt(delta(g, 'rol_ae_min'), 2)}  smallest cohort {worst[0]} {worst[1]['d']:+.4f}"
              f"{'' if worst[1]['p'] < 0.05 else ' n.s.'}")

    print("\n=== PER COHORT kappa (fold models; Table 7 reports the ensembles), p<0.05 unless n.s.")
    for lab, g in pooled.items():
        pk = per_cohort(g)
        print(lab, "  ".join(f"{k}:{v['d']:+.4f}{'' if v['p'] < 0.05 else ' n.s.'}" for k, v in sorted(pk.items())))

    print("\n=== DA-A minus DA-MC (kappa, paired on the same recording-folds)")
    h = pooled["DA-A"][["rec_id", "cohort", "subject_id", "kappa"]].merge(
        pooled["DA-MC"][["rec_id", "kappa"]], on="rec_id", suffixes=("", "_3"))
    print("pooled", fmt(boot(h, "kappa", "kappa_3")))
    marg = {}
    for c_, g in h.groupby("cohort"):
        r = boot(g, "kappa", "kappa_3"); marg[c_] = r["d"]
        print(f"  {c_:12s} {r['d']:+.4f} p={r['p']:.4f}")

    print("\n=== WITHOUT the class-prior correction (Supplementary Table S4)")
    for lab, (D, c) in src.items():
        g = cond(D, c.format(p="loso").replace("+prior", ""))
        print(f"{lab}: kappa {fmt(delta(g, 'kappa'))}  F1 pp {fmt(delta(g, 'f1_macro', 100), 2)}")
    for lab, g in pooled.items():
        print(f"{lab} corrected: F1 pp {delta(g, 'f1_macro', 100)['d']:+.2f}")

    print("\n=== TRANSDUCTIVE (whole cohort pooled) against leave-one-subject-out")
    for lab, (D, c) in src.items():
        gt = cond(D, c.format(p="cohort")); gl = pooled[lab]
        pt, pl = per_cohort(gt), per_cohort(gl)
        md = max(abs(pt[k]["d"] - pl[k]["d"]) for k in pt)
        print(f"{lab}: transductive {delta(gt, 'kappa')['d']:+.4f}  loso {delta(gl, 'kappa')['d']:+.4f}"
              f"  max per-cohort difference {md:.4f}")

    print("\n=== DATA REQUIREMENT (fold models; Figure 4 shows the ensembles), pooled kappa change by number of reference subjects")
    curve = []
    for lab, (D, c) in src.items():
        row = []
        for p, K in KS:
            g = cond(D, c.format(p=p)); dk = delta(g, "kappa")
            row.append(f"{K}:{dk['d']:+.4f} p={dk['p']:.3f} (n={len(g)})")
            curve.append(dict(series=lab, K=K, dk=dk["d"], n=len(g)))
        print(lab, " ".join(row))
    print("DA-G at one subject: acc pp", f"{delta(cond(M, C1.format(p='loso1')), 'accuracy', 100)['d']:+.2f}")
    for p in ("loso1", "loso2", "loso5"):
        g = cond(M3, C3.format(p=p))
        print(f"DA-MC {p}: {fmt(delta(g, 'kappa', nb=NB_S9))}  {clusters(g)}")
    g1 = cond(M3, C3.format(p="loso1"))
    print("DA-MC one subject, per cohort:", "  ".join(f"{k}:{v['d']:+.4f}" for k, v in sorted(per_cohort(g1).items())))
    print(f"DA-MC one subject: within-10 pp {fmt(within10(g1, NB_S9), 2)}")

    print("\n=== REM-ONSET ENDPOINTS under adaptation (run directories, leave-one-subject-out)")
    pat = lambda name: os.path.join(RUNS, "final_full_seed0_fold*of10", "stage_b", name, "per_recording_metrics.csv")
    base = load_rundirs(pat("da_da2_loso_pred-conf_pre_n2_multicluster-drop_c0.7_prior_baseline"), coh)
    rd = {}
    for lab, name in (("DA-G", "da_da1_loso_all_pre_n2_affine_c0.7_prior"),
                      ("DA-A", "da_da2_loso_pred-conf_pre_n2_affine_c0.7_prior"),
                      ("DA-MC", "da_da2_loso_pred-conf_pre_n2_multicluster-drop_c0.7_prior")):
        x = load_rundirs(pat(name), coh); rd[lab] = x
        kk = x[["key", "kappa"]].merge(pooled[lab][["rec_id", "kappa"]].rename(columns={"rec_id": "key"}), on="key")
        e = rol_endpoints(x, base)
        print(f"{lab}: within-10 {e['w10_base']:.1f} -> {e['w10']:.1f}  d {e['d']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}]"
              f" p={e['p']:.4f}  false early {e['early_base']} -> {e['early']} of {e['n_rec']}"
              f"  (run directory matches the table: max|d kappa| {np.abs(kk.kappa_x - kk.kappa_y).max():.1e})")
    a, b = rd["DA-MC"], rd["DA-A"]
    m = a.merge(b[["key", "rol_ae_min"]], on="key", suffixes=("", "_2")).dropna(
        subset=["rol_true_min", "rol_ae_min", "rol_ae_min_2"])
    m["w3"] = (m.rol_ae_min <= 10).astype(float); m["w2"] = (m.rol_ae_min_2 <= 10).astype(float)
    mm = m.assign(rec_id=m.key)
    r = boot(mm, "w3", "w2"); r2 = boot(mm, "rol_ae_min", "rol_ae_min_2")
    ahead = sum(int(g.w3.mean() > g.w2.mean()) for _, g in mm.groupby("cohort"))
    print(f"DA-MC minus DA-A: within-10 {100 * r['d']:+.2f} [{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}] p={r['p']:.4f},"
          f" ahead in {ahead} of {mm.cohort.nunique()} cohorts;  ROL AE {r2['d']:+.2f} p={r2['p']:.4f}")

    print("\n=== DA-A lead against quantities available at deployment (Spearman over cohorts)")
    g3 = pooled["DA-MC"]
    mg = pd.Series(marg)
    for lab, s in (("unadapted kappa", g3.groupby("cohort").kappa_base.mean()),
                   ("domain gap", g3.groupby("cohort").gap.mean()),
                   ("fit residual", g3.groupby("cohort").resid.mean())):
        rho, p = stats.spearmanr(mg[s.index], s)
        print(f"  {lab}: rho {rho:+.2f} p={p:.2f}")

    print("\n=== DA-MC WITH EXPERT LABELS (fold models; Supplementary Table S5 reports the ensembles)")
    O = load(f"{A}/domain_adaptation_da3_oracle_perrec.csv", coh)
    for p, K in KS:
        o, q = cond(O, CO.format(p=p)), cond(M3, C3.format(p=p))
        j = o[["rec_id", "cohort", "subject_id", "kappa"]].merge(q[["rec_id", "kappa"]], on="rec_id", suffixes=("", "_ps"))
        print(f"K={K:>3}: expert labels {fmt(delta(o, 'kappa', nb=NB_S9))}  pseudo-labels {delta(q, 'kappa')['d']:+.4f}"
              f"  paired {fmt(boot(j, 'kappa', 'kappa_ps', NB_S9))}")

    print("\n=== DA-O, classifier head fine-tuned on expert labels (halves protocol)")
    fr = []
    for d in sorted(glob.glob(f"{A}/da4/fold*_da_o")):
        f = int(re.search(r"fold(\d+)_", d).group(1))
        x = pd.read_csv(f"{d}/per_recording.csv"); x["fold"] = f; fr.append(x)
    H = pd.concat(fr)
    H = H[H.cohort.isin(coh) & H.condition.str.startswith("oracle/") & H.condition.str.endswith("+prior")].copy()
    H["K"] = H.condition.str.extract(r"/k(\d+)_")[0].fillna("half")
    H["rec_id"] = H.rec_id.astype(str) + "|f" + H.fold.astype(str)
    b0 = H.rec_id.str.split("|").str[0]; H["subject_id"] = b0.map(SUBJ).fillna(b0)
    d3 = pooled["DA-MC"].set_index("rec_id").kappa
    res = {}
    for K in ("1", "2", "5", "10", "15", "half"):
        g = H[H.K == K].copy()
        r = delta(g, "kappa"); res[K] = r["d"]
        curve.append(dict(series="DA-O", K="all" if K == "half" else K, dk=r["d"], n=len(g)))
        g["k3"] = g.rec_id.map(d3); gg = g.dropna(subset=["k3"])
        v = boot(gg, "kappa", "k3")
        print(f"K={K:>4}: {fmt(r)}  minus DA-MC (all subjects) {v['d']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] p={v['p']:.4f}")
    g = H[H.K == "half"]
    print(f"   half: acc pp {delta(g, 'accuracy', 100)['d']:+.2f}  F1 pp {delta(g, 'f1_macro', 100)['d']:+.2f}"
          f"  ROL AE {fmt(delta(g, 'rol_ae_min'), 2)}  smallest cohort {min(v['d'] for v in per_cohort(g).values()):+.4f}")
    for K in ("5", "10", "15"):
        dk3 = delta(cond(M3, C3.format(p="loso" + K)), "kappa")["d"]
        print(f"   K={K}: {100 * res[K] / res['half']:.0f}% of the half-cohort value;"
              f" {res[K] / dk3:.1f} times DA-MC from the same {K} subjects")
    print("   per cohort by K (fold models; Supplementary Table S13 reports the ensembles):")
    pc = H.assign(dk=H.kappa - H.kappa_base).groupby(["cohort", "K"]).dk.mean().unstack("K")[["1", "2", "5", "10", "15", "half"]]
    pc.loc["Pooled"] = [res[k] for k in pc.columns]
    print(pc.round(3).to_string())
    hb = load_rundirs(f"{A}/da4/fold*_da_o/baseline/per_recording_metrics.csv", coh)
    for K in ("1", "2", "5", "10", "15", "half"):
        sub = f"oracle_halves_k{K}_*_prior" if K != "half" else "oracle_halves_s*_prior"
        x = load_rundirs(f"{A}/da4/fold*_da_o/{sub}/per_recording_metrics.csv", coh)
        e = rol_endpoints(x, hb)
        print(f"   DA-O K={K:>4}: within-10 {e['w10_base']:.1f} -> {e['w10']:.1f}  d {e['d']:+.2f}"
              f" [{e['lo']:+.2f}, {e['hi']:+.2f}] p={e['p']:.4f}  false early {e['early_base']} -> {e['early']}")

    out = f"{A}/paper/da_runs/fold_kcurve.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(curve).to_csv(out, index=False)
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
