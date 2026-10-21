# -*- coding: utf-8 -*-
"""Macrostructure metrics (§3.14) — first-class outputs, computed wherever
epoch metrics are.

Definitions (checked against the Keras ``onset_latencies`` — differences are
flagged in the README, not silently changed):

* Sleep onset: first epoch of any SLEEP stage (N1/N2/N3/REM). The Keras code
  used "first epoch != wake", which also fires on unknown epochs; with -1
  present in the axis the two differ — the spec definition is used here.
* Latency to persistent sleep (sensitivity variant): first epoch starting
  ``persistent_min`` (10) consecutive minutes of sleep; excluded (-1) epochs
  break the run and can never define the onset.
* ROL / SWSOL: elapsed time from sleep onset to the first epoch scored
  REM / N3. Ground truth and prediction each use THEIR OWN sleep onset —
  an early false sleep onset propagating into the latency is part of the
  error being measured.
* Time axis is WALL CLOCK: arrays must keep excluded epochs as -1 in place
  (never compacted), so removal cannot compress a latency. Excluded epochs
  cannot themselves define an onset.
* Censoring: no imputation. A recording without REM (or N3) in a given
  hypnogram has undefined latency -> excluded from latency aggregates, with
  counts and a 2×2 presence agreement table reported.

SOREMP: threshold ground-truth and predicted ROL (default 15 min, ICSD-3
narcolepsy criterion, plus a 10/15/20 sweep); sensitivity/specificity/
confusion reported, and the per-cohort SOREMP-positive count is computed
early so sparse-positive cohorts are visible.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from data.readers.base import N3, REM, VALID_STAGES

SLEEP_STAGES = (1, 2, 3, 4)  # N1, N2, N3, REM
EPOCH_SEC = 30.0


# ---------------------------------------------------------------------------
# Onsets and latencies for a single hypnogram (wall-clock indexed)
# ---------------------------------------------------------------------------

def sleep_onset_index(y: np.ndarray, mode: str = "first_sleep",
                      persistent_min: float = 10.0,
                      epoch_sec: float = EPOCH_SEC) -> Optional[int]:
    """Index of sleep onset, or None if the hypnogram contains no sleep.

    ``y`` uses -1 for excluded/unscored epochs, which never define an onset.
    """
    y = np.asarray(y)
    is_sleep = np.isin(y, SLEEP_STAGES)
    if mode == "first_sleep":
        idx = np.flatnonzero(is_sleep)
        return int(idx[0]) if idx.size else None
    if mode == "persistent":
        need = int(np.ceil(persistent_min * 60.0 / epoch_sec))
        run = 0
        for i, s in enumerate(is_sleep):
            run = run + 1 if s else 0  # -1 and W both break the run
            if run >= need:
                return int(i - need + 1)
        return None
    raise ValueError(f"Unknown sleep-onset mode '{mode}'")


def stage_onset_latency_min(y: np.ndarray, stage: int, onset_idx: Optional[int],
                            epoch_sec: float = EPOCH_SEC) -> float:
    """Minutes from sleep onset to first epoch of ``stage``; NaN when the
    stage is absent or there is no sleep onset."""
    if onset_idx is None:
        return float("nan")
    y = np.asarray(y)
    idx = np.flatnonzero(y == stage)
    if idx.size == 0:
        return float("nan")
    return (int(idx[0]) - onset_idx) * epoch_sec / 60.0


def count_stage_entries(y: np.ndarray, stage: int) -> int:
    """Number of entries into ``stage`` (cycle count proxy). Ported from the
    Keras ``count_cycles``; an entry from an excluded (-1) epoch counts."""
    y = np.asarray(y)
    if y.size == 0:
        return 0
    entries = int(np.sum((y[1:] == stage) & (y[:-1] != stage)))
    if y[0] == stage:
        entries += 1
    return entries


def record_macrostructure(y_true: np.ndarray, y_pred: np.ndarray,
                          epoch_sec: float = EPOCH_SEC,
                          onset_mode: str = "first_sleep",
                          persistent_min: float = 10.0) -> Dict[str, float]:
    """All macrostructure quantities for one recording.

    ``y_true``/``y_pred`` MUST be wall-clock aligned full-length arrays with
    -1 in place for excluded epochs. Predictions at epochs where the ground
    truth is excluded should also be set to -1 by the caller if those epochs
    are to be excluded on the prediction side as well.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be wall-clock aligned")

    t_on = sleep_onset_index(y_true, onset_mode, persistent_min, epoch_sec)
    p_on = sleep_onset_index(y_pred, onset_mode, persistent_min, epoch_sec)

    row = {
        "n_epochs": int(y_true.size),
        "n_excluded_true": int(np.sum(~np.isin(y_true, VALID_STAGES))),
        "sleep_onset_true_min": (np.nan if t_on is None
                                 else t_on * epoch_sec / 60.0),
        "sleep_onset_pred_min": (np.nan if p_on is None
                                 else p_on * epoch_sec / 60.0),
    }
    for stage, name in ((REM, "rol"), (N3, "swsol")):
        t_lat = stage_onset_latency_min(y_true, stage, t_on, epoch_sec)
        p_lat = stage_onset_latency_min(y_pred, stage, p_on, epoch_sec)
        row[f"{name}_true_min"] = t_lat
        row[f"{name}_pred_min"] = p_lat
        row[f"{name}_me_min"] = p_lat - t_lat          # NaN-propagating
        row[f"{name}_ae_min"] = abs(p_lat - t_lat)
        stage_key = "rem" if stage == REM else "n3"
        row[f"{stage_key}_present_true"] = bool(np.any(y_true == stage))
        row[f"{stage_key}_present_pred"] = bool(np.any(y_pred == stage))
        row[f"{stage_key}_entries_true"] = count_stage_entries(y_true, stage)
        row[f"{stage_key}_entries_pred"] = count_stage_entries(y_pred, stage)
    return row


def macrostructure_table(records: List[dict], epoch_sec: float = EPOCH_SEC,
                         onset_mode: str = "first_sleep",
                         persistent_min: float = 10.0) -> pd.DataFrame:
    """Per-recording macrostructure rows. ``records`` as in epoch_metrics
    (y_true/y_pred wall-clock aligned, plus rec_id/cohort/montage_family/
    n_channels)."""
    rows = []
    for r in records:
        row = {"rec_id": r["rec_id"], "cohort": r["cohort"],
               # falls back to rec_id, which makes every recording its own
               # cluster -- i.e. exactly the old recording-level behaviour
               "subject_id": r.get("subject_id", r["rec_id"]),
               "montage_family": r["montage_family"],
               "n_channels": r["n_channels"]}
        row.update(record_macrostructure(r["y_true"], r["y_pred"], epoch_sec,
                                         onset_mode, persistent_min))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation (per stratum): AE/ME distributions, censoring, 2x2, SOREMP
# ---------------------------------------------------------------------------

def _within_rates(errors: np.ndarray, ks: Sequence[float]) -> Dict[str, float]:
    errors = errors[~np.isnan(errors)]
    out = {}
    for k in ks:
        out[f"within_{int(k)}min"] = (float(np.mean(np.abs(errors) <= k))
                                      if errors.size else np.nan)
    return out


def summarize_latency(df: pd.DataFrame, name: str,
                      ks: Sequence[float] = (5.0, 10.0, 15.0, 30.0)) -> dict:
    """Aggregate one latency family (``name`` in {rol, swsol}) over a
    stratum's per-recording table. Undefined latencies are excluded with
    counts reported — no imputation.

    REPORT THE MEDIAN AND THE TAIL, not just the mean. These errors are heavily
    skewed: on the E5 baseline, ROL had a mean AE of 19.5 min against a MEDIAN of
    1.5 min, so the mean is a statement about a minority of catastrophic
    recordings rather than about typical accuracy. The distinction is not
    cosmetic — the time branch moved mean ROL AE by -5.2 min while leaving the
    median at exactly 0.00, i.e. it repairs the tail and does nothing for the
    typical night. A comparison on means alone would have been read as a general
    accuracy gain.

    ``within_k`` is |predicted - true| <= k minutes: agreement with the true
    latency, NOT a threshold on the latency itself.
    """
    me = df[f"{name}_me_min"].to_numpy(float)
    ae = df[f"{name}_ae_min"].to_numpy(float)
    defined = ~np.isnan(me)
    stage_key = "rem" if name == "rol" else "n3"
    tp = df[f"{stage_key}_present_true"].to_numpy(bool)
    pp = df[f"{stage_key}_present_pred"].to_numpy(bool)

    out = {
        "metric": name.upper(),
        "n_recordings": int(len(df)),
        "n_defined": int(defined.sum()),
        "n_undefined": int((~defined).sum()),
        # 2x2 presence agreement (stage present in truth × in prediction)
        "present_both": int(np.sum(tp & pp)),
        "present_true_only": int(np.sum(tp & ~pp)),
        "present_pred_only": int(np.sum(~tp & pp)),
        "present_neither": int(np.sum(~tp & ~pp)),
        "ae_mean_min": float(np.nanmean(ae)) if defined.any() else np.nan,
        "ae_median_min": float(np.nanmedian(ae)) if defined.any() else np.nan,
        "me_mean_min": float(np.nanmean(me)) if defined.any() else np.nan,
        "me_median_min": float(np.nanmedian(me)) if defined.any() else np.nan,
        "entries_diff_mean": float(np.mean(
            df[f"{stage_key}_entries_pred"] - df[f"{stage_key}_entries_true"])),
    }
    out.update(_within_rates(me, ks))
    ae_defined = ae[~np.isnan(ae)]
    if ae_defined.size:
        out["ae_median_min"] = float(np.median(ae_defined))
        out["ae_p75_min"] = float(np.percentile(ae_defined, 75))
        out["ae_p90_min"] = float(np.percentile(ae_defined, 90))
    else:
        out["ae_median_min"] = out["ae_p75_min"] = out["ae_p90_min"] = np.nan
    return out


def soremp_metrics(df: pd.DataFrame,
                   thresholds_min: Sequence[float] = (10.0, 15.0, 20.0)
                   ) -> pd.DataFrame:
    """SOREMP detection: threshold BOTH true and predicted ROL.

    Only recordings where both latencies are defined enter the confusion
    matrix; positives counts per threshold are reported so sparse-positive
    cohorts (expected in healthy data) are visible early — if positives are
    sparse this is a descriptive secondary analysis, not an endpoint.
    """
    rows = []
    t = df["rol_true_min"].to_numpy(float)
    p = df["rol_pred_min"].to_numpy(float)
    both = ~np.isnan(t) & ~np.isnan(p)
    for thr in thresholds_min:
        pos_t = t[both] <= thr
        pos_p = p[both] <= thr
        tp = int(np.sum(pos_t & pos_p))
        fn = int(np.sum(pos_t & ~pos_p))
        fp = int(np.sum(~pos_t & pos_p))
        tn = int(np.sum(~pos_t & ~pos_p))
        rows.append({
            "threshold_min": thr,
            "n_evaluable": int(both.sum()),
            "n_positive_true": int(pos_t.sum()),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        })
    return pd.DataFrame(rows)


def stratified_macrostructure(macro_df: pd.DataFrame,
                              ks: Sequence[float] = (10.0, 15.0, 20.0)
                              ) -> pd.DataFrame:
    """Latency summaries per cohort (primary), montage family, and pooled."""
    rows = []

    def add(stratum_type, stratum, sub):
        for name in ("rol", "swsol"):
            row = {"stratum_type": stratum_type, "stratum": stratum}
            row.update(summarize_latency(sub, name, ks))
            rows.append(row)

    for key in ("cohort", "montage_family", "n_channels"):
        for value, sub in macro_df.groupby(key, sort=True):
            add(key, str(value), sub)
    add("pooled", "all", macro_df)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detection and agreement analyses (Experiment 6)
# ---------------------------------------------------------------------------

def presence_metrics(df: pd.DataFrame, stage_key: str) -> Dict[str, float]:
    """Did the stage occur at all? Sensitivity / specificity / PPV / NPV.

    This is the analysis that COVERS THE CENSORED RECORDINGS. A night with no
    REM has an undefined ROL and drops out of every latency aggregate, which
    silently conflates "no REM occurred" with "REM occurred and we measured it
    well". On this corpus that is 3 recordings for REM but 22 for N3, so for
    slow-wave sleep it is a material fraction rather than a rounding error.
    """
    t = df[f"{stage_key}_present_true"].astype(bool).to_numpy()
    p = df[f"{stage_key}_present_pred"].astype(bool).to_numpy()
    tp = int(np.sum(t & p)); fn = int(np.sum(t & ~p))
    fp = int(np.sum(~t & p)); tn = int(np.sum(~t & ~p))
    return {
        "stage": stage_key, "n": int(len(t)),
        "n_present_true": int(t.sum()), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
    }


def latency_threshold_metrics(df: pd.DataFrame, name: str,
                              thresholds_min: Sequence[float] = (10.0, 15.0,
                                                                 20.0)
                              ) -> pd.DataFrame:
    """Confusion matrix from thresholding BOTH true and predicted latency.

    Generalises ``soremp_metrics`` to any latency. Note the two are not equally
    informative on this corpus: SOREMP (ROL <= 15 min) has ZERO positives out of
    260 evaluable recordings — median ROL is 104 min, and SOREMPs are a
    narcolepsy marker absent from a largely non-narcoleptic population — whereas
    SWSOL <= 15 min splits 27% / 73%. ``n_positive_true`` is reported so a
    degenerate task is visible rather than presented as a perfect specificity.
    """
    rows = []
    t = pd.to_numeric(df[f"{name}_true_min"], errors="coerce").to_numpy(float)
    p = pd.to_numeric(df[f"{name}_pred_min"], errors="coerce").to_numpy(float)
    both = ~np.isnan(t) & ~np.isnan(p)
    for thr in thresholds_min:
        pos_t, pos_p = t[both] <= thr, p[both] <= thr
        tp = int(np.sum(pos_t & pos_p)); fn = int(np.sum(pos_t & ~pos_p))
        fp = int(np.sum(~pos_t & pos_p)); tn = int(np.sum(~pos_t & ~pos_p))
        rows.append({
            "latency": name, "threshold_min": thr,
            "n_evaluable": int(both.sum()), "n_censored": int((~both).sum()),
            "n_positive_true": int(pos_t.sum()),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
            "auroc": _latency_auroc(t[both], p[both], thr),
        })
    return pd.DataFrame(rows)


def _latency_auroc(true_min: np.ndarray, pred_min: np.ndarray,
                   threshold: float) -> float:
    """AUROC using the PREDICTED latency as a continuous score.

    Threshold-free: it asks whether the model could screen for the condition at
    SOME operating point, which a fixed-threshold confusion matrix cannot. The
    score is negated because a SHORT predicted latency indicates the positive
    class. Needs no probabilities, so it applies retrospectively to runs that
    logged only argmax predictions.
    """
    y = (true_min <= threshold).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan                      # degenerate: one class only
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, -pred_min))
    except Exception:                      # noqa: BLE001
        return np.nan


def latency_agreement(df: pd.DataFrame, name: str) -> Dict[str, float]:
    """ICC(2,1), Bland-Altman and Spearman for one latency.

    AE and ME say how wrong the model is; none of them say whether it TRACKS
    between-subject variation. A model that predicted the cohort median for
    everyone would post a respectable AE and an ICC of zero.

    ICC(2,1) is the two-way random-effects, absolute-agreement, single-measure
    form — the one appropriate for method comparison, and stricter than
    consistency ICC because it penalises systematic bias.
    """
    t = pd.to_numeric(df[f"{name}_true_min"], errors="coerce")
    p = pd.to_numeric(df[f"{name}_pred_min"], errors="coerce")
    ok = t.notna() & p.notna()
    t, p = t[ok].to_numpy(float), p[ok].to_numpy(float)
    out = {"latency": name, "n_evaluable": int(len(t)),
           "n_censored": int((~ok).sum())}
    if len(t) < 3:
        return {**out, "icc": np.nan, "bias_min": np.nan,
                "loa_low_min": np.nan, "loa_high_min": np.nan, "spearman": np.nan}
    diff = p - t
    out["bias_min"] = float(diff.mean())
    sd = float(diff.std(ddof=1))
    out["loa_low_min"] = out["bias_min"] - 1.96 * sd
    out["loa_high_min"] = out["bias_min"] + 1.96 * sd

    # ICC(2,1): two-way random effects, absolute agreement, single measures
    n, k = len(t), 2
    m = np.column_stack([t, p])
    grand = m.mean()
    ms_r = k * ((m.mean(axis=1) - grand) ** 2).sum() / (n - 1)       # subjects
    ms_c = n * ((m.mean(axis=0) - grand) ** 2).sum() / (k - 1)       # raters
    ss_tot = ((m - grand) ** 2).sum()
    ss_err = ss_tot - ms_r * (n - 1) - ms_c * (k - 1)
    ms_e = ss_err / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    out["icc"] = float((ms_r - ms_e) / denom) if denom > 0 else np.nan

    from scipy import stats as _st
    out["spearman"] = float(_st.spearmanr(t, p).statistic)
    return out
