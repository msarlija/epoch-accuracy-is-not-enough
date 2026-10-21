# -*- coding: utf-8 -*-
"""ENTRY POINT 1 — raw EDFs + annotations -> memmaps + labels + manifest.

Reads the configured cohorts, applies resampling (100 Hz), zero-phase
filtering (0.2–30 Hz on the FULL recording), 30 s label harmonization with
per-epoch purity, spectral QC, and writes per recording:

    <rec_id>_x.npy       (n_samples, C) float32 continuous filtered signal
    <rec_id>_y.npy       (n_epochs,)    int8 labels (-1 unknown)
    <rec_id>_purity.npy  (n_epochs,)    float32 label purity
    manifest.csv         one row per recording (appended as we go)

Runs once per preprocessing configuration (the filter cutoff is a
preprocessing parameter; changing it means re-running this script).
Resumable: recordings already in the manifest with arrays on disk are
skipped. Per-cohort counts before and after QC are logged at the end.

Usage:
    python scripts/prepare_data.py --config configs/preprocess.yaml
    python scripts/prepare_data.py --config configs/preprocess.yaml --cohorts MASS,WSC
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config  # noqa: E402
from data import manifest as mf  # noqa: E402
from data.cohorts import get_cohort  # noqa: E402
from data.preprocess import (EPOCH_SEC, RecordStats, TARGET_FS,  # noqa: E402
                             bandpass_filter, harmonize_labels,
                             resample_to_target, trim_to_labeled)
from data.qc import quality_kwargs, record_quality  # noqa: E402
from data.readers import get_reader  # noqa: E402
from data.readers.base import SKIPPED  # noqa: E402

log = logging.getLogger("prepare_data")


def process_recording(rec, preproc_cfg: dict, qc_cfg: dict):
    """RawRecording -> (signal (n,C) float32, labels int8, purity float32,
    stats, qc_flag, channel_flags)."""
    fs_out = float(preproc_cfg.get("target_fs") or TARGET_FS)
    epoch_sec = float(preproc_cfg.get("epoch_sec") or EPOCH_SEC)
    epoch_samples = int(round(fs_out * epoch_sec))
    filt = preproc_cfg.get("filter") or {}

    signal = resample_to_target(rec.signal, rec.fs, fs_out)
    signal = bandpass_filter(signal, fs_out,
                             low_hz=filt.get("low_hz", 0.2),
                             high_hz=filt.get("high_hz", 30.0),
                             order=filt.get("order", 4))

    harmonized = harmonize_labels(rec.hypnogram, epoch_sec,
                                  end_time=signal.shape[0] / fs_out)
    signal, harmonized = trim_to_labeled(signal, fs_out, harmonized)
    if harmonized.n_epochs == 0:
        raise ValueError("no labeled epochs after harmonization")

    stats = RecordStats.from_signal(signal)
    qc_flag, channel_flags = record_quality(
        signal, harmonized.labels, fs_out, epoch_samples,
        **quality_kwargs(qc_cfg))

    return signal, harmonized, stats, qc_flag, channel_flags


def iter_recordings_resilient(reader, spec, ccfg, counts):
    """Yield recordings, absorbing per-recording failures in the READER.

    ``main`` already guards ``process_recording``, but the reader itself is a
    generator, so an exception raised while *producing* a recording escapes that
    guard and kills the whole cohort. That is a live failure mode rather than a
    hypothetical: readers raise ``FileNotFoundError`` for a recording listed in
    an import table but absent from a partial download (MrOS, SOF and HPAP are
    all partial here), and ``LookupError`` when a file carries none of the
    cohort's candidate channel sets — 50 of the 53 MASS SS1 recordings have no
    Fp channel at all. Those belong in the ``failed`` tally the caller already
    keeps, not in a traceback that discards the rest of the cohort.

    A failure inside a generator closes it, so iteration cannot resume past the
    offending item; we log it and end the cohort cleanly with whatever was
    yielded. Readers that skip-and-continue internally are unaffected.
    """
    iterator = iter(reader(spec, ccfg))
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except Exception:  # noqa: BLE001 — one bad source file must not end the run
            counts["failed"] += 1
            log.exception(
                "%s: reader raised while producing a recording; ending this "
                "cohort with %d recording(s) yielded so far",
                spec.name, counts["seen"])
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohorts", default=None,
                        help="comma-separated subset overriding the config's "
                             "enabled flags")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    data_dir = cfg["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    preproc_cfg = {k: cfg.get(k) for k in ("target_fs", "epoch_sec", "filter")}
    qc_cfg = cfg.get("qc", {})
    ph = mf.preproc_hash({**preproc_cfg, "qc": qc_cfg})
    log.info("preprocessing hash: %s", ph)

    if args.cohorts:
        selected = [c.strip() for c in args.cohorts.split(",")]
    else:
        selected = [name for name, ccfg in cfg["cohorts"].items()
                    if ccfg.get("enabled", True)]

    min_age = cfg.get("min_age")
    if min_age is not None:
        log.info("inclusion criterion: age >= %s (unknown age is KEPT)", min_age)
    done = mf.existing_rec_ids(data_dir)
    counts = {}  # cohort -> dict(seen, written, skipped, failed, qc_fail)

    for cohort_name in selected:
        spec = get_cohort(cohort_name)
        ccfg = dict(cfg["cohorts"][cohort_name])
        # Readers that fetch over the network need to know what is already in
        # the corpus BEFORE they transfer anything: the resume check below runs
        # after the reader has yielded, which for an S3 cohort means the file has
        # already been downloaded. Re-running STAGES without this re-fetches
        # ~60 GB to write a handful of new recordings.
        ccfg["_done_rec_ids"] = done
        reader = get_reader(spec.reader)
        c = counts.setdefault(cohort_name, dict(seen=0, written=0, skipped=0,
                                                failed=0, qc_fail=0,
                                                unreadable=0, under_age=0))
        log.info("=== %s ===", cohort_name)
        SKIPPED[cohort_name] = 0   # recordings the reader itself had to drop

        for rec in iter_recordings_resilient(reader, spec, ccfg, c):
            c["seen"] += 1
            # Adult-only inclusion criterion. Applied here rather than in each
            # reader so it holds for every cohort, present and future.
            # UNKNOWN age is kept, not dropped: DOD-H ships no age at all, and
            # silently discarding 25 recordings for a missing field would be a
            # different decision from excluding children.
            if min_age is not None and rec.age is not None:
                try:
                    if float(rec.age) < float(min_age):
                        c["under_age"] += 1
                        log.debug("%s: age %.1f < %.1f, excluded",
                                  rec.rec_id, float(rec.age), float(min_age))
                        continue
                except (TypeError, ValueError):
                    pass
            paths = mf.array_paths(data_dir, rec.rec_id)
            if rec.rec_id in done and os.path.exists(paths["x"]):
                c["skipped"] += 1
                log.debug("%s already done, skipping", rec.rec_id)
                continue
            t0 = time.time()
            try:
                signal, harm, stats, qc_flag, ch_flags = process_recording(
                    rec, preproc_cfg, qc_cfg)
            except Exception:  # noqa: BLE001 — keep the corpus run alive
                c["failed"] += 1
                log.exception("%s FAILED", rec.rec_id)
                continue

            mf.save_arrays(data_dir, rec.rec_id, signal,
                           harm.labels, harm.purity)
            mf.append_row(data_dir, {
                "rec_id": rec.rec_id,
                "cohort": rec.cohort,
                "montage_family": rec.montage_family,
                "n_channels": rec.n_channels,
                "channel_names": rec.channel_names,
                "subject_id": rec.subject_id,
                "age": rec.age,
                "sex": rec.sex,
                "qc_pass": qc_flag,
                "qc_channel_flags": ch_flags,
                "source_path": rec.source_path,
                "n_epochs": harm.n_epochs,
                "n_samples": signal.shape[0],
                "mean_label_purity": float(np.mean(harm.purity)),
                "min_label_purity": float(np.min(harm.purity)),
                "median": stats.median,
                "iqr": stats.iqr,
                "preproc_hash": ph,
                "notes": rec.notes,
            })
            done.add(rec.rec_id)
            c["written"] += 1
            if not qc_flag:
                c["qc_fail"] += 1
            log.info("%s: %d epochs, C=%d, qc=%s (%.1fs)", rec.rec_id,
                     harm.n_epochs, rec.n_channels, qc_flag,
                     time.time() - t0)

        c["unreadable"] = SKIPPED[cohort_name]

    log.info("=== per-cohort counts (before / after QC) ===")
    for cohort_name, c in counts.items():
        after_qc = c["written"] - c["qc_fail"]
        log.info("%-8s seen=%d written=%d resumed=%d unreadable=%d failed=%d "
                 "under_age=%d | qc_fail=%d -> usable_new=%d",
                 cohort_name, c["seen"], c["written"], c["skipped"],
                 c["unreadable"], c["failed"], c.get("under_age", 0),
                 c["qc_fail"], after_qc)
        # A cohort that yields nothing exits 0 otherwise, which reads as success
        if c["written"] == 0 and c["skipped"] == 0:
            log.warning("%s produced NO recordings — check the paths and the "
                        "import table (%d unreadable, %d failed)",
                        cohort_name, c["unreadable"], c["failed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
