# -*- coding: utf-8 -*-
"""Statistics (§3.15).

Primary inference: RECORDING-LEVEL paired comparison, resampled by SUBJECT
within cohort strata (and a paired permutation/sign-flip alternative). NOT
paired t-tests over repeated random splits: those are not independent
experimental units.

The estimand is the mean paired difference per RECORDING; the resampling unit is
the SUBJECT. The two differ because some cohorts contain repeat nights from one
person (BOAS 128 recordings / 100 people, WSC 102 / 96), and two nights from the
same person are not independent evidence. Treating each recording as its own
cluster overstates n and shrinks the interval. Subjects with a single recording
are unaffected, so this only ever widens intervals where it should.

Every comparison reports the effect size (mean paired difference, Cohen's d
of the differences) with a 95% CI. Benjamini–Hochberg FDR is applied across
the macrostructure metric family; the primary endpoint is ROL AE and is
reported uncorrected as primary, everything else secondary and corrected.

Usage: build per-recording metric tables for two arms with
eval.epoch_metrics.per_recording_epoch_metrics /
eval.macrostructure.macrostructure_table, join on rec_id, then call
``compare_arms``.
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def _paired_frame(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str
                  ) -> pd.DataFrame:
    cols = ["rec_id", "cohort", metric]
    if "subject_id" in df_a.columns:
        cols.insert(2, "subject_id")
    a = df_a[cols].rename(columns={metric: "a"})
    b = df_b[["rec_id", metric]].rename(columns={metric: "b"})
    merged = a.merge(b, on="rec_id", how="inner")
    if "subject_id" not in merged.columns:
        merged["subject_id"] = merged["rec_id"]
    # A duplicated rec_id turns this inner join into a per-record cartesian
    # product, inflating n and shrinking every CI with nothing downstream
    # detecting it. Cheap to check, so check.
    for name, df in (("a", df_a), ("b", df_b)):
        dup = df["rec_id"][df["rec_id"].duplicated()].unique()
        if len(dup):
            raise ValueError(
                f"arm {name} has {len(dup)} duplicated rec_id(s), e.g. "
                f"{list(dup[:3])}. Pooling val with test, or the same fold "
                "aggregated twice, would do this.")
    merged["diff"] = merged["a"] - merged["b"]
    return merged.dropna(subset=["diff"])


def collapse_to_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Average each subject's recordings into one row, keyed by subject.

    Use when the ESTIMAND should be per subject rather than per recording. The
    subject-clustered bootstrap already gets the uncertainty right either way,
    but the point estimate differs: without collapsing, a person with three
    nights contributes three times the weight of a person with one.

    ``rec_id`` is set to the subject id so the existing pairing machinery works
    unchanged. Label columns that are constant within a subject (``cohort``,
    ``arm``, ``montage_family``) are carried through with 'first' rather than
    dropped -- callers group and annotate by them downstream, and silently losing
    them turns into a KeyError far from here.
    """
    if "subject_id" not in df.columns:
        return df
    metrics = df.select_dtypes("number").columns.tolist()
    labels = [c for c in ("cohort", "arm", "montage_family")
              if c in df.columns and c not in metrics]
    grouped = df.groupby("subject_id", as_index=False).agg(
        {**{m: "mean" for m in metrics}, **{c: "first" for c in labels}})
    grouped["rec_id"] = grouped["subject_id"]
    return grouped


def paired_bootstrap(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str,
                     n_boot: int = 10000, seed: int = 0,
                     alpha: float = 0.05) -> Dict[str, float]:
    """Cluster bootstrap of the mean paired difference: SUBJECTS resampled with
    replacement within each cohort stratum, pairs kept intact.

    A subject enters or leaves a resample with all of its recordings at once, so
    the interval reflects the number of independent people rather than the number
    of nights. Two-sided p-value from the bootstrap distribution's sign
    crossings.
    """
    merged = _paired_frame(df_a, df_b, metric)
    if len(merged) == 0:
        return {"metric": metric, "n": 0}
    rng = np.random.default_rng(seed)

    # Per cohort, reduce each subject to (sum of diffs, n recordings). The
    # resampled mean over recordings is then sum(sums)/sum(counts), which is
    # exact and lets the whole bootstrap be drawn as one index matrix per
    # cohort instead of a Python loop over n_boot x n_subjects.
    tot_sum = np.zeros(n_boot)
    tot_n = np.zeros(n_boot)
    n_subjects = 0
    for _, g in merged.groupby("cohort"):
        by_subj = g.groupby("subject_id")["diff"]
        sums = by_subj.sum().to_numpy(dtype=float)
        counts = by_subj.size().to_numpy(dtype=float)
        n_subjects += len(sums)
        idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
        tot_sum += sums[idx].sum(axis=1)
        tot_n += counts[idx].sum(axis=1)
    boots = tot_sum / tot_n

    observed = float(merged["diff"].mean())
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p: how often the resampled mean crosses zero
    p = 2.0 * min(np.mean(boots <= 0.0), np.mean(boots >= 0.0))
    sd = float(merged["diff"].std(ddof=1))
    return {
        "metric": metric,
        "n": int(len(merged)),
        "n_subjects": int(n_subjects),
        "mean_diff": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "cohens_d": observed / sd if sd > 0 else np.nan,
        "method": "paired_bootstrap_subject_cluster",
    }


def paired_permutation(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str,
                       n_perm: int = 10000, seed: int = 0
                       ) -> Dict[str, float]:
    """Sign-flip permutation test on the paired differences (exchangeability of
    the arm labels within each SUBJECT).

    The flip is drawn per subject and applied to all of that subject's
    recordings, matching the bootstrap's clustering: flipping a person's two
    nights independently would treat them as two independent exchanges.
    """
    merged = _paired_frame(df_a, df_b, metric)
    if len(merged) == 0:
        return {"metric": metric, "n": 0}
    diffs = merged["diff"].to_numpy()
    subj_codes = pd.factorize(merged["subject_id"])[0]
    n_subj = int(subj_codes.max()) + 1 if len(subj_codes) else 0
    rng = np.random.default_rng(seed)
    observed = float(np.mean(diffs))
    flips = rng.choice([-1.0, 1.0], size=(n_perm, n_subj))[:, subj_codes]
    null = (flips * diffs[None, :]).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    sd = float(np.std(diffs, ddof=1))
    return {
        "metric": metric,
        "n": int(len(diffs)),
        "n_subjects": n_subj,
        "mean_diff": observed,
        "p_value": float(p),
        "cohens_d": observed / sd if sd > 0 else np.nan,
        "method": "paired_permutation_subject_cluster",
    }


def fdr_bh(p_values: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (NaNs pass through)."""
    p = np.asarray(p_values, float)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    out[valid] = adj
    return out


PRIMARY_ENDPOINT = "rol_ae_min"
MACRO_FAMILY = ["rol_ae_min", "rol_me_min", "swsol_ae_min", "swsol_me_min"]
EPOCH_FAMILY = ["accuracy", "f1_macro", "kappa"]


def compare_arms(df_a: pd.DataFrame, df_b: pd.DataFrame,
                 metrics: Optional[Sequence[str]] = None,
                 n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Compare two arms' per-recording tables (paired on rec_id).

    FDR correction is applied ACROSS THE MACROSTRUCTURE FAMILY; the primary
    endpoint (ROL AE) is flagged and its uncorrected p is the primary
    inference.
    """
    if metrics is None:
        metrics = [m for m in EPOCH_FAMILY + MACRO_FAMILY
                   if m in df_a.columns and m in df_b.columns]
    rows = [paired_bootstrap(df_a, df_b, m, n_boot=n_boot, seed=seed)
            for m in metrics]
    out = pd.DataFrame(rows)
    out["primary_endpoint"] = out["metric"] == PRIMARY_ENDPOINT
    in_family = out["metric"].isin(MACRO_FAMILY)
    out["p_fdr_macro_family"] = np.nan
    if in_family.any():
        out.loc[in_family, "p_fdr_macro_family"] = fdr_bh(
            out.loc[in_family, "p_value"].to_numpy())
    return out
