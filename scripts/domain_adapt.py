# -*- coding: utf-8 -*-
"""Label-free feature-space alignment for cross-cohort deployment.

Reproduces the three progressively refined alignment strategies on the current
model. After training, latent representations are taken from the concatenation
entering the classifier and corrected by a per-feature affine map::

    T(x) = ((x - mu_tgt) / (sigma_tgt + eps)) * sigma_src + mu_src,  eps = 1e-6

The adapted representation is then passed through the FROZEN classifier. No
weight update, no gradient, no retraining, no leave-one-dataset-out: external
data never enters train/val/test, so the model is unchanged and only inference
is involved.

ALIGNMENT POINT. Statistics are computed on the head's INPUT -- the 391-d
concatenation of both GRU outputs, the fused embedding and the raw time
harmonics -- not on the 64-d representation after the head's dense layer. Those
are different tensors, and the former is the one the method is defined on.
``models.model.SleepStager.adapter_at("head_in")`` is that injection point.

STRATEGIES

``da1``  global. mu/sigma from ALL epochs on both sides. Corrects average
         amplitude and variance offsets between acquisition systems, but
         conflates technical shift with macrostructural differences arising
         from population-level variation in stage composition.
``da2``  N2-anchored. Statistics from epochs the PRE-ADAPTATION model predicts
         as N2, on both sides. N2 is the most prevalent stage across the adult
         lifespan, the least altered by normal ageing, and reliably identified
         before adaptation. Restricting the anchor removes the macrostructural
         confound while the correction still applies to all epochs.
``da3``  N2-anchored, age-matched. As da2, but the source reference is
         restricted to internal validation subjects within ``--age-band`` years
         of the target subject's age -- sharper than histogram binning, and the
         natural companion to per-recording alignment.

REFERENCE POPULATION. ``--protocol`` selects what "target" means:

``recording``  statistics from the single recording being scored. Transductive
               by construction, but DEPLOYMENT-REALISTIC: at inference you have
               exactly that recording, so there is no held-out set to leak from.
``cohort``     statistics pooled over the whole target cohort. Transductive in
               the objectionable sense, and reported as such.
``split``      estimate on a held-out subset of the cohort, report on the rest.
               Only meaningful for the larger cohorts.

N2 ANCHOR SOURCE. ``--n2-source`` controls how anchor epochs are chosen:
``pred`` (the label-free default), ``pred-conf`` (predicted N2 above
``--n2-conf``, trading anchor purity against sample size), or ``true`` -- an
ORACLE upper bound that uses ground-truth N2, not a deployable method.

    python scripts/domain_adapt.py --strategy da2 --protocol recording
    python scripts/domain_adapt.py --strategy da3 --protocol cohort --n2-source true
"""

import argparse
import json
import os
import sys
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config                            # noqa: E402
from data import manifest as mf                           # noqa: E402
from data.dataset import entries_from_manifest            # noqa: E402
from data.preprocess import EPOCH_SAMPLES, apply_normalization  # noqa: E402
from data.sampling import apply_cohort_caps               # noqa: E402
from eval.stats import paired_bootstrap                   # noqa: E402
from models.model import build_model, reconcile_head_time_skip  # noqa: E402
from scripts.run_experiment import evaluate_split, write_metrics  # noqa: E402

EPS = 1e-6

#: Half-width of the temporal-stability window for the ``stable`` anchor: an
#: epoch qualifies only if every epoch within this many positions on both sides
#: is also predicted as an anchor stage.
STABLE_RADIUS = 1

#: Named epoch clusters for multi-cluster alignment. NREM is grouped because
#: what matters for an anchor is that an epoch IS NREM, not which NREM stage:
#: measured externally, 93.3% of predicted-NREM epochs really are NREM against
#: 84.2% for predicted-N2, and 11 of the 16 points of N2 contamination are the
#: adjacent NREM stages. Grouping converts most of that error into a
#: within-cluster relabelling that the anchor does not care about.
CLUSTERS = {"nrem": (1, 2, 3), "wake": (0,), "rem": (4,), "n2": (2,)}

#: Relative floor on the TARGET standard deviation, below which a dimension is
#: treated as constant and left unscaled.
#:
#: The published correction divides by (sigma_tgt + 1e-6). That guard is far too
#: small for this representation: the fused block is post-ReLU and therefore
#: SPARSE, so a unit that never fires in the target recording has sigma_tgt
#: exactly 0 and picks up a scale factor of sigma_src * 1e6. Measured on real
#: recordings, the fused block reached scale factors of 1.5e5 to 8.4e5 while its
#: median stayed near 2 -- and since the head's dense layer mixes all 391
#: dimensions, a handful of exploded ones destroy the logits. That is what drove
#: the N2-anchored variants to kappa 0.53 and ROL 95 min. The GRU blocks are
#: tanh-bounded and never show it; only the rectified block does.
#:
#: A dimension with no variance in the target carries no scale information, so
#: the honest transform leaves its scale at 1. Its MEAN is still matched, which
#: is well defined -- a constant feature is simply relocated.
SIGMA_FLOOR_REL = 1e-3

N2 = 2

#: Cohorts with no per-subject age in the manifest. The Dreem Open Datasets
#: paper (arXiv:1911.03221) reports DOD-H 35.3 +/- 7.51 y (25 healthy
#: volunteers) and DOD-O 45.6 +/- 16.5 y (55 OSA patients). For da3 these
#: cohorts are matched on the published cohort MEAN rather than a per-subject
#: age -- a coarser match, and rows derived from it should be marked as such.
COHORT_AGE_FALLBACK = {"DODH": (35.3, 7.51), "DODO": (45.6, 16.5)}


# ----------------------------------------------------------------- features
def collect_features(model, split_df, cfg, device, trim_wake=None):
    """{rec_id: (head_input (T, D) float32, pred (T,), proba (T, 5), y (T,))}.

    One forward pass per recording, no gradients. The head input is the tensor
    the alignment is defined on; predictions and probabilities come from the
    same pass so the N2 anchor costs no extra inference.
    """
    from scripts.run_experiment import predict_recording  # local: heavy import
    norm = cfg.get("normalization", {})
    out = {}
    entries = entries_from_manifest(split_df)
    model.eval()
    for entry, (_, row) in zip(entries, split_df.iterrows()):
        arrays = mf.open_arrays(cfg["data_dir"], entry.rec_id, mmap=True)
        lo, hi = 0, entry.n_epochs - 1
        if trim_wake is not None:
            from data.dataset import wake_trim_bounds
            lo, hi = wake_trim_bounds(np.asarray(arrays["y"])[:entry.n_epochs], 1,
                                      trim_wake[0], trim_wake[1], entry.rec_id,
                                      cfg.get("seed", 0))
        n = hi - lo + 1
        C, S = entry.n_channels, EPOCH_SAMPLES
        x_all = np.asarray(arrays["x"][lo * S:(hi + 1) * S], dtype=np.float32)
        if entry.channel_index is not None:
            x_all = x_all[:, entry.channel_index]
        x_all = apply_normalization(x_all, norm.get("mode", "fixed"),
                                    entry.stats, norm).astype(np.float32)
        x_all = x_all.reshape(n, S, C).transpose(0, 2, 1)

        c_max = cfg.get("c_max", 2)
        chunk = cfg.get("eval", {}).get("inference_chunk_epochs", 512) or n
        feats, preds, probas, aggs, tsos = [], [], [], [], []
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            T = stop - start
            x = np.zeros((1, T, c_max, S), dtype=np.float32)
            x[0, :, :C] = x_all[start:stop]
            mask = np.zeros((1, T, c_max), dtype=bool)
            mask[0, :, :C] = True
            div = (float(n) if cfg.get("tso_norm", "elapsed") == "fraction"
                   else cfg.get("tso_divisor", 1000.0))
            tso = (start + np.arange(T, dtype=np.float32)) / div
            with torch.no_grad():
                xb = torch.from_numpy(x).to(device)
                mb = torch.from_numpy(mask).to(device)
                tb = torch.from_numpy(tso[None]).to(device)
                o = model(xb, tb, channel_mask=mb, return_features=True)
                # the channel-aggregated EEG representation, cached so an
                # EEG-level correction can be scored without re-running the CNN
                B_, T_, C_, S_ = xb.shape
                e = model.cnn(xb.reshape(B_ * T_ * C_, 1, S_)).reshape(B_, T_, C_, -1)
                a_, _ = model.agg(e, mb)
                aggs.append(a_[0].float().cpu().numpy())
                tsos.append(tso.copy())
            lg = o["logits"][0]
            feats.append(o["head_input"][0].float().cpu().numpy())
            preds.append(lg.argmax(-1).cpu().numpy())
            probas.append(torch.softmax(lg, -1).cpu().numpy())
        y = np.asarray(arrays["y"], dtype=np.int64)[lo:hi + 1]
        out[entry.rec_id] = (np.concatenate(feats), np.concatenate(preds),
                             np.concatenate(probas), y,
                             np.concatenate(aggs), np.concatenate(tsos))
    return out


def anchor_weights(pred, proba, y, source, conf, stages=(N2,)):
    """Per-epoch weight for the alignment statistics.

    ``soft`` weights every epoch by the model's posterior for the anchor
    stage(s) instead of selecting a hard subset. No threshold to tune, and it
    uses the whole posterior rather than discarding the information in it --
    which matters because the oracle anchor is worth roughly twice the
    hard-predicted one (+0.0115 vs +0.0063 kappa), so anchor purity is the
    binding constraint on this family.
    """
    if source == "soft":
        return proba[:, list(stages)].sum(1).astype(np.float64)
    m = anchor_mask(pred, proba, y, source, conf, stages)
    return m.astype(np.float64)


def anchor_mask(pred, proba, y, source, conf, stages=(N2,)):
    """Which epochs contribute to the alignment statistics.

    ``stages`` is the anchor class set. N2 alone is the published choice -- most
    prevalent across the adult lifespan, least altered by ageing, reliably
    identified before adaptation. Adding N3 roughly doubles the anchor but
    brings in the stage whose prevalence changes MOST with age, which is the
    macrostructural confound the anchor exists to avoid; whether the extra
    sample is worth that is an empirical question, so it is a flag.
    """
    if source == "all":
        return np.ones(len(pred), bool)
    if source == "true":
        return np.isin(y, stages)            # oracle, not deployable
    m = np.isin(pred, stages)
    if source == "pred-conf":
        m &= proba[:, list(stages)].sum(1) >= conf
    elif source in ("stable", "stable-conf"):
        # Keep only epochs whose NEIGHBOURS are also anchor epochs. Roughly half
        # of all residual error sits on the ~22% of epochs adjacent to a stage
        # transition, so an anchor drawn from mid-bout epochs is far purer than
        # one drawn from anywhere the model happens to call N2. This excludes
        # transition boundaries by construction rather than by confidence proxy
        # -- and confidence is a poor proxy for it, since a model can be
        # confidently wrong right at a boundary.
        stable = m.copy()
        for k in range(1, STABLE_RADIUS + 1):
            stable[k:] &= m[:-k]             # k epochs earlier also anchor
            stable[:-k] &= m[k:]             # k epochs later also anchor
        stable[:STABLE_RADIUS] = False       # undefined at the edges
        stable[-STABLE_RADIUS:] = False
        m = stable
        if source == "stable-conf":
            m &= proba[:, list(stages)].sum(1) >= conf
    return m


def build_groups(tgt_df, protocol, split_frac=0.5, seed=0):
    """rec_id -> the recordings whose epochs estimate ITS target statistics."""
    groups = defaultdict(list)
    if protocol == "recording":
        for r in tgt_df.rec_id:
            groups[r] = [r]
    elif protocol == "cohort":
        for c, g in tgt_df.groupby("cohort"):
            for r in g.rec_id:
                groups[r] = list(g.rec_id)
    elif protocol.startswith("loso"):
        # Leave-one-SUBJECT-out: every other subject of the cohort forms the
        # reference. Excluding by subject rather than by recording matters
        # because a handful of subjects contribute two nights, and leaking a
        # subject's own second night would reinstate exactly the transductive
        # advantage this protocol exists to remove.
        n_ref = int(protocol[4:]) if protocol[4:] else 0
        for c, g in tgt_df.groupby("cohort"):
            sids = g.subject_id.astype(str)
            for r, sid in zip(g.rec_id, sids):
                pool = g[sids != sid]
                if n_ref > 0:
                    others = np.asarray(
                        sorted(pool.subject_id.astype(str).unique()))
                    s_ = (zlib.crc32(f"{c}|{sid}".encode()) + int(seed)) % (2 ** 32)
                    rg = np.random.default_rng(s_)
                    rg.shuffle(others)
                    pool = pool[pool.subject_id.astype(str).isin(
                        set(others[:n_ref]))]
                groups[r] = list(pool.rec_id)
    else:
        rng = np.random.default_rng(seed)
        for c, g in tgt_df.groupby("cohort"):
            subs = np.asarray(sorted(g.subject_id.astype(str).unique()))
            rng.shuffle(subs)
            est = set(subs[:max(1, int(len(subs) * split_frac))])
            ref = list(g[g.subject_id.astype(str).isin(est)].rec_id)
            for r in g.rec_id:
                groups[r] = ref
    return groups


def _attach_diag(df, fit_diag):
    """Carry the per-recording fit diagnostics into the metrics frame."""
    for k in ("resid", "gap", "unexplained", "n_clusters"):
        df[k] = [fit_diag.get(r, {}).get(k, np.nan) for r in df.rec_id]
    return df


def multicluster_affine(src_stats, tgt_stats, cluster_w=None, return_diag=False):
    """One diagonal affine that aligns SEVERAL class-conditional clusters.

    A single anchor fixes the transform from one cluster's first two moments.
    With K clusters -- e.g. NREM, wake, REM -- each contributes two constraints
    per dimension::

        mu_t[k]*s + b = mu_s[k]        (means align)
        sd_t[k]*s     = sd_s[k]        (spreads align)

    which is 2K equations in the 2 unknowns (s, b) per dimension: an
    overdetermined system solved by least squares. It reduces exactly to the
    single-anchor solution when K=1, and with K>1 it asks the correction to be
    consistent across the sleep-wake spectrum rather than fitted to one stage.
    That is a stronger constraint, so it should be less sensitive to error in
    any one pseudo-labelled cluster -- the failure mode that has capped every
    single-anchor variant so far.

    ``cluster_w`` optionally weights clusters (e.g. by epoch count), since a
    cluster estimated from few epochs deserves less influence.
    """
    K = len(src_stats)
    D = len(src_stats[0][0])
    w = np.ones(K) if cluster_w is None else np.asarray(cluster_w, float)
    w = np.sqrt(np.maximum(w, 1e-12) / max(w.sum(), 1e-12))
    scale = np.ones(D, dtype=np.float64)
    shift = np.zeros(D, dtype=np.float64)
    for d in range(D):
        A = np.empty((2 * K, 2)); y = np.empty(2 * K)
        for k in range(K):
            mu_s, sd_s = src_stats[k][0][d], src_stats[k][1][d]
            mu_t, sd_t = tgt_stats[k][0][d], tgt_stats[k][1][d]
            A[2 * k] = (mu_t * w[k], w[k]); y[2 * k] = mu_s * w[k]
            A[2 * k + 1] = (sd_t * w[k], 0.0); y[2 * k + 1] = sd_s * w[k]
        try:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            scale[d], shift[d] = sol
        except np.linalg.LinAlgError:
            pass
    if not return_diag:
        return scale.astype(np.float32), shift.astype(np.float32)

    # ---- goodness of fit -------------------------------------------------
    # With K>1 the system is overdetermined, so a PERFECT fit is impossible
    # unless the domain shift really is one diagonal affine shared by every
    # cluster. The leftover misfit is therefore a label-free test of the
    # method's own modelling assumption -- available at deployment time,
    # because it needs no ground truth, only the pseudo-labelled clusters.
    #
    # Both residual and gap are expressed in units of the SOURCE spread, so
    # they are comparable across dimensions and across cohorts:
    #   gap    -- how far the target sits from the source before correction
    #   resid  -- how much of that a single diagonal affine cannot remove
    #   unexplained = resid/gap -- the fraction left behind (0 = perfect fit)
    # Note a single anchor (K=1) is exactly determined, so its residual is
    # identically zero and carries no information.
    mus = np.array([s_[0] for s_ in src_stats], dtype=float)   # (K, D)
    sds = np.array([s_[1] for s_ in src_stats], dtype=float)
    mut = np.array([t_[0] for t_ in tgt_stats], dtype=float)
    sdt = np.array([t_[1] for t_ in tgt_stats], dtype=float)
    denom = np.sqrt(np.maximum((sds ** 2).mean(0), 1e-12))      # (D,)
    r_mu = (mut * scale + shift - mus) / denom
    r_sd = (sdt * scale - sds) / denom
    g_mu = (mut - mus) / denom
    g_sd = (sdt - sds) / denom
    diag = {"resid": float(np.sqrt((r_mu ** 2 + r_sd ** 2).mean())),
            "gap": float(np.sqrt((g_mu ** 2 + g_sd ** 2).mean()))}
    diag["unexplained"] = diag["resid"] / max(diag["gap"], 1e-9)
    return scale.astype(np.float32), shift.astype(np.float32), diag


def coral_from(mu_s, cov_s, mu_t, cov_t, shrink=1e-3):
    """Full-covariance alignment: x' = (x - mu_t) @ Ct^-1/2 @ Cs^1/2 + mu_s.

    The diagonal correction can rescale each feature but cannot ROTATE the
    representation, and a rotation is a plausible component of montage shift --
    DOD-O is a different electrode derivation entirely. CORAL matches the full
    second moment instead.

    Only tractable because the alignment moved to the 64-d pre-ReLU layer: a
    391x391 covariance from the same epochs would be badly conditioned, and the
    post-ReLU block is sparse (singular). Both matrices get a ridge of
    ``shrink`` x mean-eigenvalue before the square roots.
    """
    d = cov_s.shape[0]
    Cs = cov_s + shrink * (np.trace(cov_s) / d) * np.eye(d)
    Ct = cov_t + shrink * (np.trace(cov_t) / d) * np.eye(d)
    et, Vt = np.linalg.eigh(Ct)
    es, Vs = np.linalg.eigh(Cs)
    et = np.maximum(et, 1e-12)
    es = np.maximum(es, 1e-12)
    Ct_isqrt = Vt @ np.diag(et ** -0.5) @ Vt.T
    Cs_sqrt = Vs @ np.diag(es ** 0.5) @ Vs.T
    W = Ct_isqrt @ Cs_sqrt
    b = mu_s - mu_t @ W
    return W.astype(np.float32), b.astype(np.float32)


def affine_from(mu_s, sd_s, mu_t, sd_t, degenerate_mode="shift",
                transform="affine"):
    """(scale, shift, n_degenerate) for T(x) = x*scale + shift.

    ``transform`` selects which half of the correction is applied:

    ``affine``  both, the published form.
    ``scale``   GAIN ONLY, shift forced to 0. The principled choice on a
                POST-ReLU layer: those activations are non-negative and sparse,
                a zero means "this feature is absent", and a mean shift lifts
                every exact zero onto one constant -- fabricating activity and
                destroying the sparsity pattern. Scaling leaves zeros exactly
                where they are, since 0 * s = 0, so it recalibrates gain
                without inventing signal.
    ``shift``   mean only, scale forced to 1.

    Dimensions whose target standard deviation falls below a relative floor are
    treated as constant: scale 1, mean still matched. See SIGMA_FLOOR_REL for
    why the published (sigma + 1e-6) guard is insufficient here.
    """
    ref = float(np.median(sd_t))
    floor = max(SIGMA_FLOOR_REL * ref, 1e-8) if ref > 0 else 1e-8
    degenerate = sd_t < floor
    scale = np.where(degenerate, 1.0, sd_s / (sd_t + EPS))
    shift = mu_s - mu_t * scale
    if transform == "scale":
        shift = np.zeros_like(shift)
    elif transform == "shift":
        scale = np.ones_like(scale)
        shift = mu_s - mu_t
    if degenerate_mode == "leave":
        # Touch nothing in a dimension we have no information about. Matching
        # the mean of a unit that never fires in the target fabricates activity
        # that is not there; leaving it alone is the conservative reading.
        shift = np.where(degenerate, 0.0, shift)
    return scale.astype(np.float32), shift.astype(np.float32), int(degenerate.sum())


MIN_EPOCHS = 200


def conf_for(conf, i):
    """Per-cluster confidence: a scalar applies to every cluster, a tuple
    (parsed from --cluster-conf) gives cluster i its own threshold."""
    return conf[i] if isinstance(conf, tuple) else conf


def conf_tag(conf):
    return ("-".join(f"{c:g}" for c in conf) if isinstance(conf, tuple)
            else f"{conf:g}")


def stats_from(feature_map, keys, source, conf, min_epochs=None,
               stages=(N2,), weights=None, want_cov=False, want_n=False):
    """(mu, sigma) over the pooled anchor epochs of `keys`, or None.

    ``weights`` maps a key to a non-negative importance weight. This is how
    age matching is applied: rather than SUBSAMPLING the source to the target's
    age profile -- which discards most of the pool and was the dominant cost in
    the earlier implementation -- every source recording contributes, weighted
    by the target/source age-density ratio for its bin. The resulting weighted
    moments correspond to the same age distribution while keeping the full
    sample.
    """
    if min_epochs is None:
        min_epochs = MIN_EPOCHS
    rows, wts = [], []
    for k in keys:
        f, p, pr, y = feature_map[k][:4]
        ew = anchor_weights(p, pr, y, source, conf, stages)
        rw = 1.0 if weights is None else float(weights.get(k, 0.0))
        if rw <= 0 or not np.any(ew > 0):
            continue
        keep = ew > 0
        rows.append(f[keep])
        wts.append(ew[keep] * rw)
    if not rows:
        return None
    X = np.concatenate(rows)
    w = np.concatenate(wts)
    if w.sum() <= 0 or (w > 0).sum() < min_epochs:
        return None
    w = w / w.sum()
    mu = (X * w[:, None]).sum(0)
    d = X - mu
    var = (w[:, None] * d ** 2).sum(0)
    sd = np.sqrt(np.maximum(var, 0.0))
    if want_n:
        # number of anchor epochs, the quantity cluster weights are built from
        return mu, sd, int((w > 0).sum())
    if not want_cov:
        return mu, sd
    cov = (d * w[:, None]).T @ d          # weighted covariance
    return mu, sd, cov


# --------------------------------------------------------------- age matching
def age_range_pool(src_df, target_ages, lo_pct=10, hi_pct=90):
    """Source rec_ids whose age lies in the target's p10-p90 age range.

    Simpler and more robust than histogram matching or importance weighting:
    take the central 80% of the target cohort's age distribution and keep every
    internal recording inside it. No bins, no per-recording band, and no
    reweighting -- the source is simply restricted to the age range the target
    actually occupies, discarding only the tails.
    """
    a = src_df["age"].astype(float)
    t = np.asarray([x for x in target_ages if x is not None and np.isfinite(x)],
                   dtype=float)
    if t.size == 0 or not a.notna().any():
        return list(src_df["rec_id"])
    lo, hi = np.percentile(t, [lo_pct, hi_pct])
    keep = src_df.loc[(a >= lo) & (a <= hi), "rec_id"]
    return list(keep) if len(keep) else list(src_df["rec_id"])


def age_match_weights(src_df, target_ages, n_bins=8):
    """{rec_id: weight} reweighting the source to the target's age profile.

    EXACT distribution matching with no data discarded: the weight of a source
    recording is the ratio of target to source density in its age bin, so the
    weighted source has the target's age profile while every recording still
    contributes. This replaces histogram SUBSAMPLING, which cut the pool from
    196 recordings to 27-93 and whose variance cost was measured to outweigh the
    bias it removed (age matching lost ~0.19 pp of accuracy, p<0.0001, in 8 of
    10 datasets).

    No age band is involved. Source recordings with no recorded age get zero
    weight; if none has an age, weighting is skipped.
    """
    src_age = src_df["age"].astype(float).to_numpy()
    ids = src_df["rec_id"].to_numpy()
    ok = np.isfinite(src_age)
    t = np.asarray([a for a in target_ages if a is not None and np.isfinite(a)],
                   dtype=float)
    if not ok.any() or t.size == 0:
        return {r: 1.0 for r in ids}
    lo = float(min(src_age[ok].min(), t.min()))
    hi = float(max(src_age[ok].max(), t.max()))
    if hi <= lo:
        return {r: 1.0 for r in ids}
    edges = np.linspace(lo, hi, n_bins + 1)
    t_frac = np.histogram(t, bins=edges)[0].astype(float)
    t_frac /= max(t_frac.sum(), 1e-12)
    s_bin = np.clip(np.digitize(src_age, edges[1:-1]), 0, n_bins - 1)
    s_cnt = np.array([int(((s_bin == b) & ok).sum()) for b in range(n_bins)],
                     dtype=float)
    s_frac = s_cnt / max(s_cnt.sum(), 1e-12)
    out = {}
    for r, b, good in zip(ids, s_bin, ok):
        if not good or s_frac[b] <= 0:
            out[r] = 0.0
        else:
            out[r] = float(t_frac[b] / s_frac[b])
    return out if any(v > 0 for v in out.values()) else {r: 1.0 for r in ids}


def age_matched_pool(src_df, target_ages, n_bins=8, rng=None):
    """Subsample source recordings to match a target cohort's AGE DISTRIBUTION.

    This is the cohort-level form of age matching, and it is a different
    operation from the per-recording one below. Here the reference is the whole
    target cohort, so the source is downsampled by histogram over `n_bins`
    equally spaced bins until its age profile matches the target's -- retaining
    as many recordings as that allows. For a deliberately age-banded target
    this concentrates the reference in the matching band instead of averaging
    over a 20-89 spread.

    Using the per-recording +/-3-year rule at cohort level would be strictly
    worse: it would estimate every recording's source statistics from ~8-11
    peers rather than from a matched pool of ~98.
    """
    rng = rng or np.random.default_rng(0)
    src_age = src_df["age"].astype(float)
    ok = src_age.notna().to_numpy()
    t = np.asarray([a for a in target_ages if a is not None and np.isfinite(a)],
                   dtype=float)
    if not ok.any() or t.size == 0:
        return list(src_df["rec_id"])
    lo = float(min(src_age[ok].min(), t.min()))
    hi = float(max(src_age[ok].max(), t.max()))
    if hi <= lo:
        return list(src_df["rec_id"])
    edges = np.linspace(lo, hi, n_bins + 1)
    t_frac = np.histogram(t, bins=edges)[0].astype(float)
    t_frac /= t_frac.sum()
    s_bin = np.clip(np.digitize(src_age.to_numpy(), edges[1:-1]), 0, n_bins - 1)

    # largest pool whose bin proportions equal the target's
    counts = np.array([int(((s_bin == b) & ok).sum()) for b in range(n_bins)])
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where(t_frac > 0, counts / np.where(t_frac > 0, t_frac, 1), np.inf)
    total = float(np.min(cap[t_frac > 0])) if (t_frac > 0).any() else 0.0
    keep = []
    ids = src_df["rec_id"].to_numpy()
    for b in range(n_bins):
        want = int(round(total * t_frac[b]))
        if want <= 0:
            continue
        pool = ids[(s_bin == b) & ok]
        if len(pool) <= want:
            keep.extend(pool.tolist())
        else:
            keep.extend(rng.choice(pool, size=want, replace=False).tolist())
    return keep or list(src_df["rec_id"])


def age_peers(src_df, target_age, band):
    """Source rec_ids within +/- band years of target_age."""
    a = src_df["age"].astype(float)
    return list(src_df.loc[(a >= target_age - band) & (a <= target_age + band),
                           "rec_id"])


def target_age_for(row):
    """Per-subject age, or the published cohort mean for the DOD cohorts."""
    if pd.notna(row.get("age")):
        return float(row["age"]), False
    ref = COHORT_AGE_FALLBACK.get(row["cohort"])
    return (ref[0] if ref else None), True


def cohort_target_ages(cohort, group):
    """Ages describing a target cohort, for distribution matching.

    Where per-subject ages exist, those. Where they do not (DODH, DODO) the
    published mean AND standard deviation are used to draw a synthetic sample of
    the same size -- using the mean alone would make the "distribution" a spike
    and collapse the matched pool to a handful of recordings, which is a worse
    approximation than the published spread. Deterministic: fixed seed, and
    truncated to a plausible adult range.
    """
    ref = COHORT_AGE_FALLBACK.get(cohort)
    if ref is not None:
        mu, sd = ref
        draw = np.random.default_rng(0).normal(mu, sd, size=max(len(group), 200))
        return list(np.clip(draw, 18.0, 90.0))
    return [float(a) for a in group["age"] if pd.notna(a)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default=os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/runs"))
    ap.add_argument("--prefix", default="final")
    ap.add_argument("--config-prefix", default="final")
    ap.add_argument("--arm", default="full")
    ap.add_argument("--folds", type=int, default=10)
    # Lists, not single values: feature extraction dominates the runtime and
    # does NOT depend on the condition, so every requested combination is
    # evaluated from ONE extraction per fold. Running them as separate
    # invocations would re-extract the same tensors once per condition.
    ap.add_argument("--strategy", nargs="+", default=["da2"],
                    choices=("da1", "da2", "da3"))
    ap.add_argument("--protocol", nargs="+", default=["recording"],
                    help="recording | cohort | split | loso[K]. loso holds the "
                         "SUBJECT out: the alignment reference is every OTHER "
                         "subject of the same cohort, so the scored recording "
                         "never contributes to its own statistics. This is the "
                         "inductive, deployment-realistic protocol -- cohort is "
                         "transductive and optimistic by comparison. loso5 caps "
                         "the reference at 5 randomly drawn other subjects, "
                         "which traces how much target data adaptation needs.")
    ap.add_argument("--min-epochs", type=int, default=200,
                    help="floor on anchor epochs; a cluster below it is "
                         "unusable and the recording goes unadapted")
    ap.add_argument("--n2-source", nargs="+", default=["pred"],
                    choices=("pred", "pred-conf", "soft", "stable",
                             "stable-conf", "hybrid", "true"))
    ap.add_argument("--n2-conf", nargs="+", type=float, default=[0.7],
                    help="confidence threshold(s) on the summed anchor-class "
                         "posterior. Swept as a condition dimension. For a "
                         "grouped anchor this thresholds p(NREM), so an epoch "
                         "confidently NREM but ambiguous BETWEEN N1/N2/N3 still "
                         "qualifies -- which is the right criterion when the "
                         "anchor does not care which NREM stage it is.")
    ap.add_argument("--clusters", nargs="+", default=["nrem", "wake", "rem"],
                    help="clusters aligned jointly by transform=multicluster")
    ap.add_argument("--cluster-conf", nargs="*", default=[],
                    help="transform=multicluster only: per-cluster confidence "
                         "thresholds as comma-separated values in --clusters "
                         "order, e.g. 0.9,0.5,0.8. Each entry is one "
                         "condition and replaces --n2-conf for that transform. "
                         "The clusters have very different confidence "
                         "profiles (wake is pure at 0.5, REM only near 0.9), "
                         "so one shared threshold trades them off blindly.")
    ap.add_argument("--mc-weight", nargs="+", default=["equal"],
                    choices=("equal", "sqrt", "count"),
                    help="transform=multicluster: least-squares weight of each "
                         "cluster's constraints. equal (the configuration "
                         "behind every reported result), sqrt = square root of "
                         "the cluster's target epoch count, count = the count "
                         "itself. Swept as a condition dimension.")
    ap.add_argument("--mc-missing", nargs="+", default=["skip"],
                    choices=("skip", "drop"),
                    help="transform=multicluster: what happens when a cluster "
                         "has fewer than --min-epochs epochs. skip leaves the "
                         "recording unadapted (reported results); drop fits "
                         "the affine on the clusters that remain, so a "
                         "recording without enough REM gets a two-cluster "
                         "(N2 + wake) fit, and only a recording with no usable "
                         "cluster at all goes unadapted.")
    ap.add_argument("--age-band", type=float, default=3.0)
    ap.add_argument("--split-frac", type=float, default=0.5,
                    help="protocol=split: fraction used to ESTIMATE")
    ap.add_argument("--where", nargs="+", default=["head_in"],
                    choices=("eeg", "head_in", "pre", "post"),
                    help="which activations to align. head_in is the 391-d "
                         "concatenation entering the head (the point the method "
                         "is defined on); pre and post are the 64-d hidden "
                         "layer inside the head, before and after its ReLU.")
    ap.add_argument("--prior-correct", action="store_true",
                    help="after alignment, restore the predicted class marginal "
                         "to the unadapted model's, per cohort. Label-free.")
    ap.add_argument("--exclude-time", action="store_true",
                    help="at where=head_in, leave the trailing raw-time "
                         "harmonics at identity and correct only the GRU and "
                         "fused blocks. The time features are a deterministic "
                         "function of the clock, identical in every domain, so "
                         "there is nothing about them to align -- and any "
                         "correction applied to them is pure distortion of the "
                         "signal the macrostructure result depends on.")
    ap.add_argument("--transform", nargs="+", default=["affine"],
                    choices=("affine", "scale", "shift", "coral",
                             "multicluster"),
                    help="which half of the correction to apply. scale-only is "
                         "the appropriate form on a post-ReLU layer, where a "
                         "mean shift would lift exact zeros onto a constant.")
    ap.add_argument("--anchor-stages", nargs="+", default=["N2"],
                    choices=("N1", "N2", "N3"),
                    help="anchor class set for DA-2/DA-3. N2 alone is the "
                         "published choice. All NREM (N1 N2 N3) is far purer as "
                         "a pseudo-labelled set -- 93.3%% of predicted-NREM "
                         "epochs really are NREM against 84.2%% for N2 -- "
                         "because ~11 of the 16 points of N2 contamination are "
                         "the adjacent NREM stages. The cost is that NREM "
                         "composition varies with age, which is what age "
                         "matching the source (DA-3) is there to control.")
    ap.add_argument("--degenerate", default="shift", choices=("shift", "leave"),
                    help="what to do with a target dimension that has no "
                         "variance: still match its mean, or leave it untouched")
    ap.add_argument("--min-peers", type=int, default=5,
                    help="da3: fail rather than silently widen the age band")
    ap.add_argument("--out-name", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    globals()['MIN_EPOCHS'] = args.min_epochs
    for _p in args.protocol:
        if not (_p in ('recording', 'cohort', 'split')
                or (_p.startswith('loso')
                    and (not _p[4:] or _p[4:].isdigit()))):
            ap.error(f'bad --protocol {_p}')
    cluster_confs = []
    for text in args.cluster_conf:
        vals = tuple(float(v) for v in text.split(","))
        if len(vals) != len(args.clusters):
            ap.error(f"--cluster-conf {text!r} needs {len(args.clusters)} values")
        cluster_confs.append(vals)
    STAGE_ID = {"N1": 1, "N2": 2, "N3": 3}
    stages = tuple(STAGE_ID[x] for x in args.anchor_stages)
    stage_tag = "".join(args.anchor_stages).lower()
    # multicluster variants are encoded in the transform name, e.g.
    # multicluster-wsqrt-drop, so every output tag stays distinct and the
    # default name (equal weights, skip) is the reported configuration
    transforms = []
    for tr in args.transform:
        if tr != "multicluster":
            transforms.append(tr); continue
        for w_ in args.mc_weight:
            for m_ in args.mc_missing:
                transforms.append("multicluster" + ("" if w_ == "equal" else f"-w{w_}")
                                  + ("" if m_ == "skip" else f"-{m_}"))
    combos = [(st, pr, n2, wh, tr, cf) for st in args.strategy
              for pr in args.protocol
              for n2 in (args.n2_source if st != "da1" else ["all"])
              for wh in args.where for tr in transforms
              for cf in (cluster_confs if tr.startswith("multicluster") and cluster_confs
                         else args.n2_conf if n2 in ("pred-conf", "stable-conf")
                         or tr.startswith("multicluster") else [args.n2_conf[0]])]
    print(f"{len(combos)} condition(s) x {args.folds} fold(s), "
          f"one feature extraction per fold\n")
    acc = {c: {"adapted": [], "base": []} for c in combos}

    for fold in range(args.folds):
        run_dir = f"{args.runs_root}/{args.prefix}_{args.arm}_seed0_fold{fold}of10"
        ckpt_path = os.path.join(run_dir, "stage_b.pt")
        if not os.path.exists(ckpt_path):
            print(f"fold {fold}: no checkpoint, skipping", flush=True)
            continue
        cfg = load_config(f"configs/{args.config_prefix}_{args.arm}.yaml")
        manifest = mf.load_manifest(cfg["data_dir"])
        if cfg.get("require_qc_pass", True):
            manifest = manifest[manifest["qc_pass"] == 1].reset_index(drop=True)
        with open(os.path.join(run_dir, "split_manifest.json")) as fh:
            split_ids = json.load(fh)["splits"]
        # Source = this fold's val AND test. Both are INTERNAL recordings whose
        # labels we hold; the pretence of ignorance applies only to the external
        # target. Withholding the test split would halve the reference pool for
        # no methodological gain, and the pool size is the binding constraint on
        # age-matched alignment -- matching a narrow age band against 98
        # recordings leaves ~15, against ~196 it leaves ~30.
        internal_ids = list(split_ids["val"]) + list(split_ids["test"])
        src_df = manifest[manifest.rec_id.isin(internal_ids)].reset_index(drop=True)
        tgt_df = manifest[manifest.cohort.isin(cfg["external_cohorts"])]
        caps = cfg.get("cohort_caps") or {}
        if caps:
            tgt_df = apply_cohort_caps(tgt_df, caps, seed=cfg.get("seed", 0))
        tgt_df = tgt_df.reset_index(drop=True)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        mcfg = reconcile_head_time_skip(dict(load_config(
            os.path.join(run_dir, "resolved_config.yaml"))["model"]),
            ckpt["state_dict"], os.path.basename(run_dir))
        model = build_model(mcfg).to(args.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        trim = cfg.get("trim_wake") or {}
        trim_wake = ((float(trim.get("min_minutes", 20.0)),
                      float(trim.get("max_minutes", 40.0)))
                     if trim.get("enabled") else None)

        print(f"fold {fold}: extracting features "
              f"({len(src_df)} source, {len(tgt_df)} target)", flush=True)
        src_feats_all = collect_features(model, src_df, cfg, args.device, trim_wake)
        tgt_feats_all = collect_features(model, tgt_df, cfg, args.device, None)
        head = model.seq_head
        min_purity = (cfg.get("purity") or {}).get("min_label_purity")

        # y_true once per recording; identical across conditions
        y_cache = {}
        for rid in tgt_feats_all:
            y = np.asarray(tgt_feats_all[rid][3], dtype=np.int64)
            if min_purity is not None:
                pur = np.asarray(mf.open_arrays(cfg["data_dir"], rid,
                                                mmap=True)["purity"])[:len(y)]
                y = np.where(pur < min_purity, -1, y)
            y_cache[rid] = y

        for (strategy, protocol, n2_source, where, transform, conf) in combos:
            src_kind = "all" if strategy == "da1" else n2_source
            # The SOURCE side may use ground truth even in a deployable method:
            # the internal corpus is ours and its labels are known. Only the
            # TARGET must be label-free. `hybrid` uses true N2 internally and
            # predicted N2 externally -- still deployable, but with a clean
            # reference anchor.
            if n2_source == "hybrid":
                src_kind_src, src_kind_tgt = "true", "pred-conf"
            else:
                src_kind_src = src_kind_tgt = src_kind
            # `head_in` aligns the concatenation itself; `pre`/`post` align the
            # 64-d hidden layer inside the head, so the stored head input must
            # be pushed through seq_head.dense (and its ReLU for `post`) first.
            def project(fm):
                if where in ("head_in", "eeg"):
                    # `eeg` aligns the cached aggregated representation itself
                    if where == "eeg":
                        return {k: (v[4], v[1], v[2], v[3], v[4], v[5])
                                for k, v in fm.items()}
                    return fm
                out = {}
                for k, (f, p_, pr_, y_, ag_, ts_) in fm.items():
                    with torch.no_grad():
                        t = head.dense(torch.from_numpy(f).to(args.device))
                        if where == "post":
                            t = head.act(t)
                    out[k] = (t.float().cpu().numpy(), p_, pr_, y_, ag_, ts_)
                return out
            src_feats, tgt_feats = project(src_feats_all), project(tgt_feats_all)
            groups = build_groups(tgt_df, protocol,
                                  args.split_frac, cfg.get("seed", 0))


            need_cov = transform == "coral"
            cl_sets = [CLUSTERS[c] for c in args.clusters]
            mc_src = None
            is_mc = transform.startswith("multicluster")
            mc_weight = ("sqrt" if "-wsqrt" in transform
                         else "count" if "-wcount" in transform else "equal")
            mc_drop = transform.endswith("-drop")
            if is_mc:
                src_keys = (list(src_feats) if strategy != "da3" else None)
                if strategy == "da3":
                    mc_src = {}          # filled per cohort below
                else:
                    mc_src = [stats_from(src_feats, src_keys, src_kind_src,
                                         conf_for(conf, i), stages=cs)
                              for i, cs in enumerate(cl_sets)]
            src_global = (stats_from(src_feats, list(src_feats), src_kind_src,
                                     conf_for(conf, 0), stages=stages,
                                     want_cov=need_cov)
                          if strategy != "da3" else None)
            da3_cohort_src = {}
            if strategy == "da3":
                rng_a = np.random.default_rng(cfg.get("seed", 0))
                for c, g in tgt_df.groupby("cohort"):
                    ages = cohort_target_ages(c, g)
                    pool = age_range_pool(src_df, ages)
                    st = stats_from(src_feats, pool, src_kind_src, conf_for(conf, 0),
                                    stages=stages, want_cov=(transform == "coral"))
                    da3_cohort_src[c] = st
                    if is_mc:
                        mc_src[c] = [stats_from(src_feats, pool, src_kind_src,
                                                conf_for(conf, i), stages=cs)
                                     for i, cs in enumerate(cl_sets)]
                    lo, hi = np.percentile([x for x in ages if x is not None], [10, 90])
                    print(f"      da3 {c}: p10-p90 = {lo:.0f}-{hi:.0f} y -> "
                          f"{len(pool)}/{len(src_df)} source recordings",
                          flush=True)
            recs, base_recs, skipped = [], [], []
            fit_diag = {}
            flat_dims = 0
            for _, row in tgt_df.iterrows():
                rid = row["rec_id"]
                tgt = stats_from(tgt_feats, groups[rid], src_kind_tgt,
                                 conf_for(conf, 0), stages=stages, want_cov=need_cov)
                from_mean = False
                if strategy == "da3":
                    # cohort / split: match the target COHORT's age DISTRIBUTION
                    src = da3_cohort_src.get(row["cohort"])
                    from_mean = row["cohort"] in COHORT_AGE_FALLBACK
                    if src is None:
                        skipped.append((rid, "no age-matched pool")); continue
                else:
                    src = src_global
                # with mc_drop the single N2 anchor is not required: the
                # cluster loop below decides what can be fitted
                if src is None or (tgt is None and not (is_mc and mc_drop)):
                    skipped.append((rid, "too few anchor epochs")); continue
                if is_mc:
                    ms = mc_src[row["cohort"]] if strategy == "da3" else mc_src
                    mt = [stats_from(tgt_feats, groups[rid], src_kind_tgt,
                                     conf_for(conf, i), stages=cs, want_n=True)
                          for i, cs in enumerate(cl_sets)]
                    use = [i for i in range(len(cl_sets))
                           if ms[i] is not None and mt[i] is not None]
                    if not use or (len(use) < len(cl_sets) and not mc_drop):
                        skipped.append((rid, "a cluster had too few epochs")); continue
                    n_ep = np.array([mt[i][2] for i in use], dtype=float)
                    cw = (None if mc_weight == "equal"
                          else np.sqrt(n_ep) if mc_weight == "sqrt" else n_ep)
                    scale, shift, _d = multicluster_affine(
                        [(ms[i][0], ms[i][1]) for i in use],
                        [(mt[i][0], mt[i][1]) for i in use],
                        cluster_w=cw, return_diag=True)
                    _d["n_clusters"] = len(use)
                    fit_diag[rid] = _d
                    n_flat = 0
                elif transform == "coral":
                    mu_s, _, cov_s = src
                    mu_t, _, cov_t = tgt
                    scale, shift = coral_from(mu_s, cov_s, mu_t, cov_t)
                    n_flat = 0
                else:
                    mu_s, sd_s = src[0], src[1]
                    mu_t, sd_t = tgt[0], tgt[1]
                    scale, shift, n_flat = affine_from(mu_s, sd_s, mu_t, sd_t,
                                                       args.degenerate, transform)
                    if args.exclude_time and where == "head_in":
                        # trailing TSO_FEATURE_DIM dims are the raw harmonics
                        scale[-7:] = 1.0
                        shift[-7:] = 0.0
                flat_dims += n_flat
                f = tgt_feats[rid][0]
                y_true = y_cache[rid]
                with torch.no_grad():
                    t = torch.from_numpy(f).to(args.device)
                    sc = torch.tensor(scale, dtype=torch.float32, device=args.device)
                    sh = torch.tensor(shift, dtype=torch.float32, device=args.device)
                    apply_T = ((lambda z: z @ sc + sh) if transform == "coral"
                               else (lambda z: z * sc + sh))
                    if where == "eeg":
                        # correct the EEG representation, then let fusion, the
                        # GRUs and the head run on the corrected input
                        tso_t = torch.from_numpy(
                            tgt_feats_all[rid][5]).to(args.device)[None]
                        lg_a = model.forward_from_agg(
                            apply_T(t)[None], tso_t)["logits"][0]
                        lg_b = model.forward_from_agg(t[None], tso_t)["logits"][0]
                    elif where == "head_in":
                        lg_a, lg_b = head(apply_T(t)), head(t)
                    else:
                        # t is already the 64-d hidden representation
                        post = t if where == "post" else head.act(t)
                        post_a = (apply_T(t) if where == "post"
                                  else head.act(apply_T(t)))
                        lg_a, lg_b = head.fc(post_a), head.fc(post)
                for store, lg in ((recs, lg_a), (base_recs, lg_b)):
                    pred = lg.argmax(-1).cpu().numpy()
                    store.append({
                        "rec_id": rid, "cohort": row["cohort"],
                        "subject_id": row.get("subject_id", rid),
                        "montage_family": row["montage_family"],
                        "n_channels": row["n_channels"], "y_true": y_true,
                        "y_pred": np.where(y_true == -1, -1, pred),
                        "proba": torch.softmax(lg, -1).cpu().numpy(),
                        "age_from_cohort_mean": from_mean})
            if not recs:
                continue

            tag = (f"da_{strategy}_{protocol}_{n2_source}_{where}_{stage_tag}"
                   f"_{transform}_c{conf_tag(conf)}")
            # ---- optional label-free prior correction
            # Aligning the target's features to the source implicitly imposes
            # the SOURCE's class prior. Measured on the best configuration, four
            # of five classes improve and N1 alone loses 3.5 F1 points -- N1 has
            # the weakest feature evidence of any stage (it is largely "not
            # clearly W, not yet N2"), so its predictions are the first to be
            # absorbed by W and N2 when the decision boundary moves.
            #
            # This restores the predicted class marginal to what the UNADAPTED
            # model produced on the same target, using only the model's own
            # outputs -- no labels. It keeps the alignment's effect on WHICH
            # epochs get which label while removing its effect on HOW MANY.
            #
            # Diagnostic as much as fix: if kappa survives, the alignment
            # genuinely improved discrimination; if kappa collapses, the gain
            # WAS the prior shift and adaptation is only re-tuning the
            # operating point.
            if args.prior_correct:
                prior_recs = []
                by_cohort = defaultdict(lambda: [[], []])
                for ra, rb in zip(recs, base_recs):
                    by_cohort[ra["cohort"]][0].append(ra["proba"])
                    by_cohort[ra["cohort"]][1].append(rb["proba"])
                adjust = {}
                for c, (pa, pb) in by_cohort.items():
                    pi_post = np.concatenate(pa).mean(0)
                    pi_pre = np.concatenate(pb).mean(0)
                    adjust[c] = np.log(np.maximum(pi_pre, 1e-8) /
                                       np.maximum(pi_post, 1e-8))
                for ra in recs:
                    lp = np.log(np.maximum(ra["proba"], 1e-12)) + adjust[ra["cohort"]]
                    lp -= lp.max(1, keepdims=True)
                    pr = np.exp(lp); pr /= pr.sum(1, keepdims=True)
                    yt = ra["y_true"]
                    prior_recs.append({**ra, "proba": pr,
                                       "y_pred": np.where(yt == -1, -1, pr.argmax(1))})
                jp = _attach_diag(write_metrics(prior_recs,
                                   os.path.join(run_dir, "stage_b",
                                                f"{tag}_prior"),
                                   cfg.get("eval", {})), fit_diag)
                jp["fold"] = fold
                key = (strategy, protocol, n2_source + "+prior", where,
                       transform, conf)
                acc.setdefault(key, {"adapted": [], "base": []})
                acc[key]["adapted"].append(jp)
                acc[key]["base"].append(
                    write_metrics(base_recs, os.path.join(
                        run_dir, "stage_b", f"{tag}_prior_baseline"),
                        cfg.get("eval", {})).assign(fold=fold))
            j_ = _attach_diag(write_metrics(
                recs, os.path.join(run_dir, "stage_b", tag),
                cfg.get("eval", {})), fit_diag)
            b_ = write_metrics(base_recs,
                               os.path.join(run_dir, "stage_b", tag + "_baseline"),
                               cfg.get("eval", {}))
            j_["fold"] = b_["fold"] = fold
            acc[(strategy, protocol, n2_source, where, transform,
                 conf)]["adapted"].append(j_)
            acc[(strategy, protocol, n2_source, where, transform,
                 conf)]["base"].append(b_)
            note = f"  ({len(skipped)} unadapted)" if skipped else ""
            if flat_dims:
                note += f"  [{flat_dims / max(len(recs), 1):.1f} degenerate dims/rec]"
            print(f"    {tag:34s} n={len(j_):4d} kappa {j_.kappa.mean():.4f} "
                  f"(base {b_.kappa.mean():.4f}, {j_.kappa.mean()-b_.kappa.mean():+.4f})"
                  f"{note}", flush=True)

    # ------------------------------------------------------------- reporting
    summary = []
    for combo, d in acc.items():
        if not d["adapted"]:
            continue
        A = pd.concat(d["adapted"], ignore_index=True)
        B = pd.concat(d["base"], ignore_index=True)
        for x in (A, B):
            x["rec_id"] = x["rec_id"].astype(str) + "|f" + x["fold"].astype(str)
        row = {"strategy": combo[0], "protocol": combo[1], "n2_source": combo[2],
               "where": combo[3], "transform": combo[4], "conf": conf_tag(combo[5]),
               "anchor": stage_tag,
               "n_rec": A.rec_id.nunique()}
        for m in ("kappa", "accuracy", "f1_macro", "rol_ae_min"):
            r = paired_bootstrap(A, B, m, n_boot=10000, seed=0)
            row[f"{m}_base"] = B[m].mean()
            row[f"{m}_adapted"] = A[m].mean()
            row[f"{m}_delta"] = r["mean_diff"]
            row[f"{m}_p"] = r["p_value"]
        summary.append(row)
        row["_A"], row["_B"] = A, B

    print(f"\n{'='*94}\nSUMMARY (delta = adapted - baseline; "
          f"positive kappa/acc = adaptation helps)\n{'='*94}")
    print(f"{'strategy':9s} {'protocol':10s} {'anchor':10s} {'where':6s} {'xform':12s} {'conf':4s} {'n':>5s} | "
          f"{'kappa':>17s} {'p':>7s} | {'accuracy':>17s} {'p':>7s} | {'ROL AE':>15s}")
    for row in summary:
        print(f"{row['strategy']:9s} {row['protocol']:10s} {row['n2_source']:10s} "
              f"{row['where']:6s} {row['transform']:12s} {row['conf']:>12s} "
              f"{row['n_rec']:5d} | {row['kappa_base']:.4f}->{row['kappa_adapted']:.4f} "
              f"{row['kappa_delta']:+.4f} {row['kappa_p']:7.4f} | "
              f"{row['accuracy_base']:.4f}->{row['accuracy_adapted']:.4f} "
              f"{row['accuracy_delta']:+.4f} {row['accuracy_p']:7.4f} | "
              f"{row['rol_ae_min_base']:6.2f}->{row['rol_ae_min_adapted']:6.2f} "
              f"{row['rol_ae_min_delta']:+.2f}")

    print(f"\nPER-COHORT kappa delta")
    cohorts = sorted(summary[0]["_A"].cohort.unique())
    hdr = "".join(f"{c[:11]:>12s}" for c in cohorts)
    print(f"{'condition':38s}{hdr}")
    for row in summary:
        A, B = row["_A"], row["_B"]
        cells = ""
        for c in cohorts:
            a, b = A[A.cohort == c], B[B.cohort == c]
            cells += f"{a.kappa.mean()-b.kappa.mean():+12.4f}" if len(a) else f"{'--':>12s}"
        lbl = (row['strategy'] + '/' + row['protocol'] + '/' +
               row['n2_source'] + '/' + row['transform'] +
               '/c' + row['conf'])
        print(f"{lbl:52s}{cells}")

    out = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/analysis/domain_adaptation"
                             + (f"_{args.out_name}" if args.out_name else "")
                             + ".csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in summary]).to_csv(out, index=False)
    print(f"\nwrote {out}")

    per_rec = []
    for r in summary:
        A, B = r["_A"], r["_B"]
        keep = [c for c in ("rec_id", "cohort", "fold", "kappa", "accuracy",
                            "f1_macro", "rol_ae_min", "resid", "gap",
                            "unexplained", "n_clusters") if c in A.columns]
        m = A[keep].merge(
            B[["rec_id", "kappa", "accuracy", "f1_macro", "rol_ae_min"]],
            on="rec_id", suffixes=("", "_base"))
        m["condition"] = (f"{r['strategy']}/{r['protocol']}/{r['n2_source']}/"
                          f"{r['transform']}/c{r['conf']}")
        per_rec.append(m)
    if per_rec:
        out2 = out.replace(".csv", "_perrec.csv")
        pd.concat(per_rec, ignore_index=True).to_csv(out2, index=False)
        print(f"wrote {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
