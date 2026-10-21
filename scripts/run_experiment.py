# -*- coding: utf-8 -*-
"""ENTRY POINT 2 — config -> trained model(s) + metrics in a run directory.

Builds the model, runs Stage A and/or Stage B, and writes checkpoints,
per-recording predictions, epoch + macrostructure metrics (stratified by
cohort / montage family / channel count), SOREMP tables, complexity profile,
split manifest and the resolved config. All experiment arms (TSO ablation,
aggregation mode, normalization mode) are selected by config, not by editing
code.

Usage:
    python scripts/run_experiment.py --config configs/final_full.yaml
    python scripts/run_experiment.py --config configs/final_full.yaml --seed 3
"""

import argparse
import json
import logging
import os
import shutil
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config, save_resolved  # noqa: E402
from data import manifest as mf  # noqa: E402
from data.dataset import (AugmentConfig, SleepSequenceDataset,  # noqa: E402
                          entries_from_manifest, make_loader,
                          sequence_length_for_duration, TSO_DIVISOR,
                          wake_trim_bounds)
from data.preprocess import (EPOCH_SAMPLES, apply_normalization,  # noqa: E402
                             reference_iqr_scale)
from data.sampling import (apply_cohort_caps, fold_of_k,  # noqa: E402
                           save_split_manifest, split_by_subject)
from eval.curves import (plot_learning_curves, summarize_per_recording,  # noqa: E402
                         write_histories, write_summary)
from eval.epoch_metrics import (per_recording_epoch_metrics,  # noqa: E402
                                stage_ranking_metrics,
                                stratified_epoch_metrics)
from eval.macrostructure import (latency_agreement,  # noqa: E402
                                 latency_threshold_metrics,
                                 macrostructure_table, presence_metrics,
                                 soremp_metrics, stratified_macrostructure)
from models.model import build_model  # noqa: E402
from models.profile import profile_model  # noqa: E402
from training.stages import run_stage_a, run_stage_b  # noqa: E402

log = logging.getLogger("run_experiment")


def _fail(message: str) -> int:
    log.error(message)
    return 1


def set_determinism(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # older torch without warn_only
            pass


# ---------------------------------------------------------------------------
# Whole-recording inference (wall-clock aligned predictions)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_recording(model, arrays, entry, norm_mode: str, norm_cfg: dict,
                      c_max: int, device, chunk_epochs=512, trim_wake=None,
                      tso_norm: str = "elapsed",
                      seed: int = 0):
    """Predict every epoch of one recording; returns (y_pred, alpha or None).

    The full hypnogram is produced in wall-clock order, -1 nowhere — masking
    against excluded ground truth happens in the metric code.

    ``chunk_epochs`` blocks are fed as ONE sequence each, so inference context is
    the block length, not the training ``seq_len``. Set it to None (or <= 0) to
    run the whole recording in a single pass. That matters because the GRUs are
    bidirectional: at each block boundary the encoder restarts from zero state
    and the last epoch of a block has no future context, and at the default 512 a
    typical 900-1100 epoch night crosses exactly one such seam. Whole-recording
    inference removes it; the cost is peak memory, since the CNN sees
    ``T x C`` epochs folded into one batch.
    """
    # Same wake trimming as training, when the split is an internal one. The
    # bounds must match SleepSequenceDataset exactly or the model is scored on a
    # span it was never trained for.
    lo, hi = 0, entry.n_epochs - 1
    if trim_wake is not None:
        lo, hi = wake_trim_bounds(np.asarray(arrays["y"])[:entry.n_epochs], 1,
                                  trim_wake[0], trim_wake[1], entry.rec_id, seed)
    n_epochs = hi - lo + 1
    step = int(chunk_epochs) if chunk_epochs and int(chunk_epochs) > 0 \
        else n_epochs
    C = entry.n_channels
    S = EPOCH_SAMPLES
    x_all = np.asarray(arrays["x"][lo * S:(hi + 1) * S], dtype=np.float32)
    if entry.channel_index is not None:
        # must mirror SleepSequenceDataset exactly: a model trained on the
        # surviving channels would otherwise be scored on the dead one
        x_all = x_all[:, entry.channel_index]
    x_all = apply_normalization(x_all, norm_mode, entry.stats,
                                norm_cfg).astype(np.float32)
    x_all = x_all.reshape(n_epochs, S, C).transpose(0, 2, 1)  # (T, C, S)

    preds, alphas, probas = [], [], []
    model.eval()
    for start in range(0, n_epochs, step):
        stop = min(start + step, n_epochs)
        T = stop - start
        x = np.zeros((1, T, c_max, S), dtype=np.float32)
        x[0, :, :C] = x_all[start:stop]
        mask = np.zeros((1, T, c_max), dtype=bool)
        mask[0, :, :C] = True
        # start is already relative to the trimmed span; the divisor is the
        # constant TSO_DIVISOR (elapsed time) or the usable length (fraction)
        div = float(n_epochs) if tso_norm == "fraction" else TSO_DIVISOR
        tso = (start + np.arange(T, dtype=np.float32)) / div

        out = model(torch.from_numpy(x).to(device),
                    torch.from_numpy(tso[None]).to(device),
                    channel_mask=torch.from_numpy(mask).to(device),
                    return_alpha=True)
        logits = out["logits"][0]
        preds.append(logits.argmax(-1).cpu().numpy())
        probas.append(torch.softmax(logits, dim=-1).cpu().numpy())
        if "alpha" in out:
            alphas.append(out["alpha"][0].cpu().numpy())

    y_pred = np.concatenate(preds)
    alpha = np.concatenate(alphas) if alphas else None
    proba = np.concatenate(probas)
    return y_pred, alpha, proba, (lo, hi)


def evaluate_split(model, split_df, data_dir, norm_mode, norm_cfg, c_max,
                   device, pred_dir=None, min_label_purity=None,
                   chunk_epochs=512, trim_wake=None, seed: int = 0,
                   tso_norm: str = "elapsed"):
    """Per-recording predictions + record dicts for the metric modules."""
    entries = entries_from_manifest(split_df)
    records = []
    for entry, (_, row) in zip(entries, split_df.iterrows()):
        arrays = mf.open_arrays(data_dir, entry.rec_id, mmap=True)
        y_pred, alpha, proba, (lo, hi) = predict_recording(
            model, arrays, entry, norm_mode, norm_cfg, c_max, device,
            chunk_epochs=chunk_epochs, trim_wake=trim_wake, seed=seed,
            tso_norm=tso_norm)
        y_true = np.asarray(arrays["y"], dtype=np.int64)[lo:hi + 1]
        if min_label_purity is not None:
            purity = np.asarray(arrays["purity"])[lo:hi + 1]
            y_true = np.where(purity < min_label_purity, -1, y_true)
        # wall-clock alignment: keep -1 in place, never compact the axis
        y_pred_masked = np.where(y_true == -1, -1, y_pred)

        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)
            # float32, not float16: ~3 decimal digits would tie large numbers
            # of epochs together and bias any AUC computed from the ranking.
            # An eval pass is a few MB either way.
            payload = {"y_true": y_true.astype(np.int8),
                       "y_pred": y_pred.astype(np.int8),
                       "proba": proba.astype(np.float32)}
            if alpha is not None:
                payload["alpha"] = alpha.astype(np.float32)
            np.savez_compressed(
                os.path.join(pred_dir, f"{entry.rec_id}.npz"), **payload)

        records.append({
            "rec_id": entry.rec_id,
            "cohort": row["cohort"],
            # carried through to the per-recording tables so the paired
            # statistics can cluster on the person rather than the night
            "subject_id": row.get("subject_id", entry.rec_id),
            "montage_family": row["montage_family"],
            "n_channels": row["n_channels"],
            "y_true": y_true,
            "y_pred": y_pred_masked,
            "proba": proba,
        })
    return records


def write_metrics(records, out_dir, eval_cfg):
    os.makedirs(out_dir, exist_ok=True)
    per_rec = per_recording_epoch_metrics(records)
    macro = macrostructure_table(
        records,
        onset_mode=eval_cfg.get("onset_mode", "first_sleep"),
        persistent_min=eval_cfg.get("persistent_min", 10.0))
    # per-recording table joining epoch + macrostructure metrics: the unit
    # of statistical analysis (eval/stats.compare_arms consumes this)
    joined = per_rec.merge(
        macro.drop(columns=["cohort", "subject_id", "montage_family",
                            "n_channels"]),
        on="rec_id")
    joined.to_csv(os.path.join(out_dir, "per_recording_metrics.csv"),
                  index=False)
    stratified_epoch_metrics(records).to_csv(
        os.path.join(out_dir, "epoch_metrics_stratified.csv"), index=False)
    stratified_macrostructure(macro).to_csv(
        os.path.join(out_dir, "macrostructure_stratified.csv"), index=False)
    # SOREMP: pooled and per cohort, with the threshold sweep
    thresholds = eval_cfg.get("soremp_thresholds_min", [10.0, 15.0, 20.0])
    tables = [soremp_metrics(macro, thresholds).assign(cohort="all")]
    for cohort, sub in macro.groupby("cohort"):
        tables.append(soremp_metrics(sub, thresholds).assign(cohort=cohort))
    pd.concat(tables).to_csv(os.path.join(out_dir, "soremp.csv"), index=False)

    # --- Experiment 6 additions -------------------------------------------
    # Stage-level ranking, from the logged probabilities. Skipped silently for
    # runs that did not log them, so old prediction dirs stay readable.
    ranking = stage_ranking_metrics(records)
    if len(ranking):
        ranking.to_csv(os.path.join(out_dir, "stage_ranking.csv"), index=False)

    # Event presence — the analysis that covers CENSORED recordings, which the
    # latency tables necessarily drop
    pd.DataFrame([presence_metrics(macro, k) for k in ("rem", "n3")]).to_csv(
        os.path.join(out_dir, "stage_presence.csv"), index=False)

    # Threshold detection for both latencies. Reported for ROL too, but expect
    # it to be degenerate: SOREMP has ZERO positives on this corpus.
    pd.concat([latency_threshold_metrics(macro, n,
                                         eval_cfg.get("latency_thresholds_min",
                                                      [10.0, 15.0, 20.0]))
               for n in ("rol", "swsol")]).to_csv(
        os.path.join(out_dir, "latency_detection.csv"), index=False)

    # Agreement: ICC / Bland-Altman / Spearman. AE says how wrong, this says
    # whether the model tracks between-subject variation at all.
    pd.DataFrame([latency_agreement(macro, n) for n in ("rol", "swsol")]).to_csv(
        os.path.join(out_dir, "latency_agreement.csv"), index=False)

    # persistent-sleep-onset sensitivity variant
    macro_persistent = macrostructure_table(
        records, onset_mode="persistent",
        persistent_min=eval_cfg.get("persistent_min", 10.0))
    stratified_macrostructure(macro_persistent).to_csv(
        os.path.join(out_dir, "macrostructure_stratified_persistent_onset.csv"),
        index=False)
    return joined


def build_datasets(cfg, data_dir):
    norm = cfg.get("normalization", {})
    aug_cfg = cfg.get("augment", {})
    augment = AugmentConfig(
        channel_dropout_p=aug_cfg.get("channel_dropout_p", 0.4),
        artifact_p=aug_cfg.get("artifact_p", 0.3),
    )
    trim = cfg.get("trim_wake") or {}
    trim_wake = ((float(trim.get("min_minutes", 20.0)),
                  float(trim.get("max_minutes", 40.0)))
                 if trim.get("enabled") else None)
    common = dict(
        data_dir=data_dir,
        trim_wake=trim_wake,
        c_max=cfg.get("c_max", 2),
        normalization=norm.get("mode", "fixed"),
        normalization_cfg=norm,
        tso_jitter=cfg.get("tso_jitter", 0.025),
        tso_divisor=float(cfg.get("tso_divisor", TSO_DIVISOR)),
        tso_norm=cfg.get("tso_norm", "elapsed"),
        min_label_purity=(cfg.get("purity") or {}).get("min_label_purity"),
        seed=cfg.get("seed", 0),
    )

    def dataset(split_df, seq_len, stride, train):
        return SleepSequenceDataset(
            entries=entries_from_manifest(split_df), seq_len=seq_len,
            stride=stride, train=train,
            augment=augment if train else None, **common)

    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="override the config seed (for repeated runs)")
    parser.add_argument("--fold", type=int, default=None,
                        help="k-fold index; test=fold, val=(fold+1)%%k. Omit to "
                             "use the config's single random split instead.")
    parser.add_argument("--n-folds", type=int, default=None,
                        help="k for --fold (default: config split.n_folds, 10)")
    parser.add_argument("--stage-a-from", default=None,
                        help="Path to an existing stage_a.pt to REUSE instead "
                             "of training Stage A. Only valid when the Stage A "
                             "configuration is identical to the source run; "
                             "the architecture is checked, the config is not.")
    parser.add_argument("--stage-a-epochs", type=int, default=None,
                        help="cap Stage A epochs (smoke runs)")
    parser.add_argument("--stage-b-epochs", type=int, default=None,
                        help="cap Stage B epochs (smoke runs)")
    parser.add_argument("--tag", default=None,
                        help="extra suffix on the run directory name")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg.get("seed", 0))
    set_determinism(seed, cfg.get("deterministic", True))

    split_cfg = cfg.get("split", {})
    n_folds = args.n_folds or split_cfg.get("n_folds") or 10
    fold = args.fold
    for stage_key, override in (("stage_a", args.stage_a_epochs),
                                ("stage_b", args.stage_b_epochs)):
        if override is not None:
            cfg.setdefault(stage_key, {})["epochs"] = int(override)
            log.info("%s epochs capped at %d by --%s-epochs", stage_key,
                     override, stage_key.replace("_", "-"))

    name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    # The fold MUST appear in the path: without it the k runs of one arm
    # overwrite each other's metrics and the aggregation silently sees one fold
    # k times.
    parts = [name, f"seed{seed}"]
    if fold is not None:
        parts.append(f"fold{fold}of{n_folds}")
    if args.tag:
        parts.append(args.tag)
    run_dir = os.path.join(cfg.get("output_root", "runs"), "_".join(parts))
    os.makedirs(run_dir, exist_ok=True)

    # keep a full log with the run, not only on the terminal
    fh = logging.FileHandler(os.path.join(run_dir, "train.log"))
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    log.info("run dir: %s", run_dir)

    device = torch.device(cfg.get("device", "cuda" if
                                  torch.cuda.is_available() else "cpu"))

    # ---- corpus: manifest -> QC filter -> caps -> subject-level split ----
    data_dir = cfg["data_dir"]
    manifest = mf.load_manifest(data_dir)
    if len(manifest) == 0:
        log.error("empty manifest in %s — run scripts/prepare_data.py first",
                  data_dir)
        return 1
    if cfg.get("require_qc_pass", True):
        before = manifest.groupby("cohort").size().to_dict()
        manifest = manifest[manifest["qc_pass"] == 1].reset_index(drop=True)
        after = manifest.groupby("cohort").size().to_dict()
        log.info("QC filter per cohort: %s -> %s", before, after)
    # External cohorts are held out of the pool ENTIRELY and scored by every
    # fold's model. They are taken from the manifest before caps and before the
    # pool filter, because the pool filter is include-only: a cohort dropped
    # there is invisible to eval too, which is how a "held-out" cohort ends up
    # never being scored at all.
    external_cohorts = cfg.get("external_cohorts") or []
    external_df = manifest[manifest["cohort"].isin(external_cohorts)] \
        .reset_index(drop=True)
    if external_cohorts:
        missing = sorted(set(external_cohorts) - set(manifest["cohort"]))
        if missing:
            log.warning("external cohorts absent from the manifest: %s", missing)
        log.info("external: %d recordings across %s", len(external_df),
                 sorted(external_df["cohort"].unique()))

    cohorts = cfg.get("cohorts")
    if cohorts:
        manifest = manifest[manifest["cohort"].isin(cohorts)].reset_index(
            drop=True)
    overlap = sorted(set(manifest["cohort"]) & set(external_cohorts))
    if overlap:
        return _fail(f"cohorts {overlap} are in BOTH the training pool and "
                     "external_cohorts; that would train on the external set")

    caps = cfg.get("cohort_caps") or {}
    manifest = apply_cohort_caps(manifest, caps, seed=seed)
    log.info("pool: %d recordings / %d subjects across %s", len(manifest),
             manifest["subject_id"].nunique(),
             dict(manifest["cohort"].value_counts()))

    if fold is not None:
        log.info("k-fold split: fold %d of %d (test=%d, val=%d)", fold,
                 n_folds, fold, (fold + 1) % n_folds)
        train_df, val_df, test_df = fold_of_k(manifest, fold, k=n_folds,
                                              seed=seed)
    else:
        log.info("single random split (no --fold given)")
        train_df, val_df, test_df = split_by_subject(
            manifest, split_cfg.get("val_frac", 0.1),
            split_cfg.get("test_frac", 0.1), seed=seed)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    if len(external_df):
        splits["external"] = external_df
    save_split_manifest(run_dir, splits, caps, seed)
    for k, df in splits.items():
        log.info("%s: %d recordings, %d subjects, %d epochs", k, len(df),
                 df["subject_id"].nunique(), int(df["n_epochs"].sum()))

    # 'iqr' mode is put on the same amplitude scale as 'fixed' so the arms
    # differ in ADAPTATION, not magnitude. Computed from the POOL (identical
    # across folds and arms for a given seed), never hardcoded, and written into
    # the resolved config saved with the run.
    norm_cfg = cfg.setdefault("normalization", {})
    if str(norm_cfg.get("scale_match", "")).lower() == "auto":
        if norm_cfg.get("mode") == "iqr":
            k_scale = reference_iqr_scale(
                manifest, scale=float(norm_cfg.get("scale", 50.0)))
            log.info("scale_match auto -> %.6f (clip %.1fxIQR lands at %.6f)",
                     k_scale, float(norm_cfg.get("clip", 20.0)),
                     float(norm_cfg.get("clip", 20.0)) / k_scale)
        else:
            k_scale = 1.0        # inert for 'fixed'; keeps one shared base config
        norm_cfg["scale_match"] = k_scale
    save_resolved(cfg, run_dir)

    dataset = build_datasets(cfg, data_dir)
    norm = cfg.get("normalization", {})
    norm_mode = norm.get("mode", "fixed")
    c_max = cfg.get("c_max", 2)
    eval_cfg = cfg.get("eval", {})
    min_purity = (cfg.get("purity") or {}).get("min_label_purity")
    model_cfg = cfg.get("model", {})
    chunk = eval_cfg.get("inference_chunk_epochs", 512)
    _tw = cfg.get("trim_wake") or {}
    trim_wake = ((float(_tw.get("min_minutes", 20.0)),
                  float(_tw.get("max_minutes", 40.0)))
                 if _tw.get("enabled") else None)
    if trim_wake:
        log.info("wake trimming ACTIVE on internal splits: keep sleep +/- "
                 "U(%.0f, %.0f) min; external scored untrimmed", *trim_wake)
    eval_splits = [k for k in ("val", "test", "external") if k in splits]
    log.info("eval splits: %s (inference chunk: %s epochs)", eval_splits,
             chunk if chunk and int(chunk) > 0 else "whole recording")
    histories, summary = {}, {}

    def evaluate_stage(model, stage: str):
        """Score every eval split and write metrics + summary for one stage."""
        summary[stage] = {}
        for split_name in eval_splits:
            # External cohorts are scored on the WHOLE recording: they stand in
            # for deployment, where nobody trims the night for you.
            split_trim = None if split_name == "external" else trim_wake
            records = evaluate_split(
                model, splits[split_name], data_dir, norm_mode, norm, c_max,
                device, pred_dir=os.path.join(run_dir, stage,
                                              f"predictions_{split_name}"),
                min_label_purity=min_purity, chunk_epochs=chunk,
                trim_wake=split_trim, seed=seed,
                tso_norm=cfg.get("tso_norm", "elapsed"))
            joined = write_metrics(
                records, os.path.join(run_dir, stage, split_name), eval_cfg)
            summary[stage][split_name] = summarize_per_recording(joined)
            m = summary[stage][split_name]
            log.info("%s %-8s: kappa %.4f  acc %.4f  f1 %.4f  (n=%d rec, "
                     "%d subj)", stage, split_name, m.get("kappa", float("nan")),
                     m.get("accuracy", float("nan")),
                     m.get("f1_macro", float("nan")), m["n_recordings"],
                     m.get("n_subjects", 0))
            if split_name == "external":
                # per-cohort too: one pooled external number hides the
                # between-site spread these cohorts are here to expose
                for cohort, sub in joined.groupby("cohort"):
                    cm = summarize_per_recording(sub)
                    summary[stage][f"external/{cohort}"] = cm
                    log.info("%s   %-14s kappa %.4f  acc %.4f  (n=%d)", stage,
                             cohort, cm.get("kappa", float("nan")),
                             cm.get("accuracy", float("nan")),
                             cm["n_recordings"])

    # ------------------------------- Stage A -------------------------------
    stage_a_cfg = cfg.get("stage_a", {})
    model_a = None
    if stage_a_cfg.get("enabled", True) and getattr(args, "stage_a_from", None):
        # Reuse, don't retrain. Two reasons this exists, both about the dropout
        # probe it was written for: Stage A is IDENTICAL across arms that differ
        # only in stage_b.dropout, so retraining it is pure waste; and Stage A
        # val accuracy varies ~0.004 run to run (0.8086-0.8124 across E8), which
        # is the same size as the Stage B effect being measured. Sharing one
        # Stage A removes that variance from the comparison instead of letting
        # it masquerade as a dropout effect.
        src = os.path.abspath(os.path.expanduser(args.stage_a_from))
        log.info("--- Stage A: REUSING %s (not training) ---", src)
        a_cfg = dict(model_cfg)
        a_cfg["with_sequence"] = False
        model_a = build_model(a_cfg)
        model_a.load_state_dict(torch.load(src, map_location="cpu")["state_dict"])
        model_a.to(device)
        shutil.copyfile(src, os.path.join(run_dir, "stage_a.pt"))
        histories["stage_a"] = {"reused_from": [src]}
        evaluate_stage(model_a, "stage_a")
    elif stage_a_cfg.get("enabled", True):
        log.info("--- Stage A (CNN-only) ---")
        bs = stage_a_cfg.get("batch_size", 256)
        nw = stage_a_cfg.get("num_workers", 8)
        train_ds = dataset(train_df, seq_len=1, stride=1, train=True)
        val_ds = dataset(val_df, seq_len=1, stride=1, train=False)
        log.info("Stage A index sizes: train=%d val=%d",
                 len(train_ds), len(val_ds))
        model_a, hist = run_stage_a(
            model_cfg,
            make_loader(train_ds, bs, shuffle=True, num_workers=nw),
            make_loader(val_ds, bs, shuffle=False, num_workers=nw),
            stage_a_cfg, run_dir)
        histories["stage_a"] = hist
        model_a.to(device)

        # Stage A is a separately evaluated model (§3.8) — full reports
        evaluate_stage(model_a, "stage_a")

    # ------------------------------- Stage B -------------------------------
    stage_b_cfg = cfg.get("stage_b", {})
    if stage_b_cfg.get("enabled", True):
        log.info("--- Stage B (sequence) ---")
        T = sequence_length_for_duration(
            stage_b_cfg.get("context_minutes", 42.5))
        stride = max(1, int(round(T * stage_b_cfg.get(
            "train_stride_frac", 0.75))))
        log.info("sequence length T=%d epochs, train stride=%d", T, stride)
        bs = stage_b_cfg.get("batch_size", 16)
        nw = stage_b_cfg.get("num_workers", 8)
        train_ds = dataset(train_df, seq_len=T, stride=stride, train=True)
        val_ds = dataset(val_df, seq_len=T, stride=T, train=False)
        log.info("Stage B index sizes: train=%d val=%d",
                 len(train_ds), len(val_ds))
        # Optional phase-1 loaders: the Bi-GRU-only warm-up trains on SHORT
        # sequences, so it needs its own pair built at that length.
        gru_cfg = stage_b_cfg.get("gru_warmup") or {}
        short_loaders = None
        if gru_cfg.get("enabled"):
            t_short = int(gru_cfg.get("context_epochs", 16))
            if t_short >= T:
                return _fail(f"gru_warmup.context_epochs ({t_short}) must be "
                             f"shorter than the Stage B length ({T})")
            s_stride = max(1, int(round(t_short * stage_b_cfg.get(
                "train_stride_frac", 0.75))))
            # keep epochs-per-batch roughly constant, so the warm-up sees a
            # comparable amount of signal per step despite the shorter window
            bs_short = int(gru_cfg.get("batch_size", bs * max(1, T // t_short)))
            tr_s = dataset(train_df, seq_len=t_short, stride=s_stride, train=True)
            va_s = dataset(val_df, seq_len=t_short, stride=t_short, train=False)
            log.info("Stage B warm-up: T=%d, stride=%d, batch=%d "
                     "(index sizes train=%d val=%d)", t_short, s_stride,
                     bs_short, len(tr_s), len(va_s))
            short_loaders = (
                make_loader(tr_s, bs_short, shuffle=True, num_workers=nw),
                make_loader(va_s, bs_short, shuffle=False, num_workers=nw))

        model_b, hist = run_stage_b(
            model_cfg,
            make_loader(train_ds, bs, shuffle=True, num_workers=nw),
            make_loader(val_ds, bs, shuffle=False, num_workers=nw),
            stage_b_cfg, run_dir, short_loaders=short_loaders)
        # the phase-1 curve rides along under 'warmup'; lift it out so it gets
        # its own history CSV and learning-curve panel
        warm = hist.pop("warmup", None) if isinstance(hist, dict) else None
        if warm is not None:
            histories["stage_b_warmup"] = warm
        histories["stage_b"] = hist
        model_b.to(device)

        evaluate_stage(model_b, "stage_b")

        # complexity profile of the deliverable model (§3.16)
        try:
            profile = profile_model(model_b, seq_len=T, n_channels=c_max)
            with open(os.path.join(run_dir, "profile.json"), "w") as f:
                json.dump(profile, f, indent=2)
        except Exception:  # noqa: BLE001 — profiling must not kill the run
            log.exception("profiling failed")

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(histories, f, indent=2)
    write_histories(histories, run_dir)
    write_summary({"run": os.path.basename(run_dir), "name": name, "seed": seed,
                   "fold": fold, "n_folds": n_folds if fold is not None else None,
                   "normalization": {k: norm.get(k) for k in
                                     ("mode", "scale", "clip", "scale_match")},
                   "tso_mode": model_cfg.get("tso_mode"),
                   "splits": {k: int(len(v)) for k, v in splits.items()},
                   "metrics": summary}, run_dir)
    fold_txt = "" if fold is None else f"  fold {fold}/{n_folds}"
    plot_learning_curves(histories, run_dir, summary=summary,
                         title=f"{name}{fold_txt}  (norm={norm_mode}, "
                               f"tso={model_cfg.get('tso_mode')})")
    log.info("done: %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
