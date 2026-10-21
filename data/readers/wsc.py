# -*- coding: utf-8 -*-
"""WSC (Wisconsin Sleep Cohort) reader — streams from S3, one at a time.

This reader was a TODO in the Keras repo, which downloaded WSC files but never
read a channel out of one, so its guessed two-channel candidates were wrong.
WSC provides a SINGLE frontal lead, which makes this the only cohort that
exercises the C=1 path on real data.

S3 (bucket ``wisconsin-sleep-cohort``)::

    polysomnography/wsc-visit<v>-<id>-nsrr.edf
    polysomnography/wsc-visit<v>-<id>-nsrr.stg.txt   ('<epoch> <stage>' per 30 s)

Each recording is fetched to a fixed ``temp.edf`` / ``temp.txt`` and overwritten
by the next, so disk use stays at one recording however many are processed
(:mod:`data.readers.s3_stream`). Set ``data_path`` instead of ``bucket`` to read
an already-downloaded local copy.

stg stage codes: 0=W 1=N1 2=N2 3=N3 4=S4->N3 5=REM 6=movement 7=unscored.

Config keys::

    bucket / key_prefix / temp_dir   S3 mode (default bucket above)
    data_path                        local mode, if the files are on disk
    import_csv                       wsc-dataset CSV: wsc_id, wsc_vst, age, sex
    visits                           which wsc_vst values to accept; visit 1
                                     has no frontal lead at all
    limit                            cap; 100 for the external-validation subset
"""

import logging
import os
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from .base import (Hypnogram, RawRecording, N1, N2, N3, REM, WAKE,
                   UNKNOWN, format_record_id, log_skip, wall_clock_offset)
from .edf_utils import load_edf_channels
from .s3_stream import S3ObjectMissing, download_to, temp_paths

DEFAULT_BUCKET = "wisconsin-sleep-cohort"
DEFAULT_PREFIX = "polysomnography"

log = logging.getLogger(__name__)

STG_STAGE_MAP = {0: WAKE, 1: N1, 2: N2, 3: N3, 4: N3, 5: REM,
                 6: UNKNOWN, 7: UNKNOWN}
EPOCH_SEC = 30.0


def _read_stg(path: str) -> Hypnogram:
    epochs, labels = [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                epoch = int(float(parts[0]))
                code = int(float(parts[-1]))
            except ValueError:
                continue  # header line
            epochs.append(epoch)
            labels.append(STG_STAGE_MAP.get(code, UNKNOWN))
    epochs = np.asarray(epochs, int)
    labels = np.asarray(labels, int)
    # epoch numbering starts at 0 or 1 depending on export; normalize to 0
    if len(epochs) and epochs.min() == 1:
        epochs = epochs - 1
    onsets = epochs.astype(float) * EPOCH_SEC
    durations = np.full(len(onsets), EPOCH_SEC)
    return Hypnogram(onsets, durations, labels)


#: `.allscore.txt` writes stage TRANSITIONS as 'STAGE - <name>'.
ALLSCORE_STAGE_MAP = {"W": WAKE, "N1": N1, "N2": N2, "N3": N3, "N4": N3,
                      "R": REM, "REM": REM, "NO STAGE": UNKNOWN}


def _read_allscore(path: str, edf_start_sec: Optional[int]) -> Hypnogram:
    """Parse `.allscore.txt`, used from visit 3 onward instead of `.stg.txt`.

    Tab-separated ``HH:MM:SS.ss<TAB>EVENT``, where the stage rows are
    ``STAGE - <name>`` and every other row is an arousal, limb movement or
    respiratory event. Stages are TRANSITIONS, not per-epoch rows, so each one
    runs until the next -- which is why durations come from the gaps.

    Onsets carry sub-second precision and do not land on a 30 s grid; that is
    fine and deliberate, because ``harmonize_labels`` takes arbitrary
    onsets/durations and reports the resulting ``label_purity`` rather than
    pretending a transition happened on a grid boundary.
    """
    rows = []
    with open(path, "r", encoding="latin-1", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or not parts[1].startswith("STAGE - "):
                continue
            name = parts[1][len("STAGE - "):].strip().upper()
            label = ALLSCORE_STAGE_MAP.get(name)
            if label is None:
                continue
            try:
                h, m, sec = parts[0].strip().split(":")
                clock = int(h) * 3600 + int(m) * 60 + float(sec)
            except ValueError:
                continue
            rows.append((clock, label))
    if not rows:
        raise ValueError(f"{path}: no 'STAGE - ' rows")

    origin = edf_start_sec if edf_start_sec is not None else min(
        c for c, _ in rows)
    onsets = np.asarray([wall_clock_offset(c, origin) for c, _ in rows], float)
    labels = np.asarray([l for _, l in rows], int)
    order = np.argsort(onsets, kind="stable")
    onsets, labels = onsets[order], labels[order]
    # a transition lasts until the next one; the last gets one epoch
    durations = np.diff(np.append(onsets, onsets[-1] + EPOCH_SEC))
    keep = (durations > 0) & (onsets >= 0)   # pre-signal rows have no samples
    if not keep.any():
        raise ValueError(f"{path}: no usable stage transitions")
    return Hypnogram(onsets[keep], durations[keep], labels[keep])


def _edf_start_seconds(path: str) -> Optional[int]:
    """Recording start from the EDF header (bytes 176:184, 'hh.mm.ss')."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(184)
        h, m, sec = (int(v) for v in head[176:184].decode("ascii").split("."))
        return h * 3600 + m * 60 + sec
    except Exception:  # noqa: BLE001
        return None


def _read_annotations(path: str, edf_path: Optional[str] = None) -> Hypnogram:
    """Dispatch on content, not on filename: the same temp.txt slot holds either
    flavour, so the marker is the header/body rather than the extension.

    ``edf_path`` may be None, which yields correct LABELS but onsets relative to
    the first annotation rather than the recording start. That is enough to test
    whether a recording is scored at all, which is why the pre-check below can
    run before the EDF is fetched.
    """
    with open(path, "r", encoding="latin-1", errors="replace") as fh:
        head = fh.read(4096)
    if "STAGE - " in head:
        start = _edf_start_seconds(edf_path) if edf_path else None
        return _read_allscore(path, start)
    return _read_stg(path)


def _has_scored_stages(ann_path: str) -> bool:
    """Does this annotation file contain any real sleep stage?

    Worth checking before the EDF is fetched: 13 of the first 105 WSC recordings
    are coded unscored for EVERY epoch, and each one cost a full 80-165 MB
    download before the annotation was even read. The annotation is 7-8 KB.
    """
    try:
        return bool((_read_annotations(ann_path).labels >= 0).any())
    except Exception:  # noqa: BLE001 — unparseable counts as not scorable
        return False


def _fetch(cfg, stem: str) -> Tuple[str, str, str]:
    """(edf path, stg path, provenance) — from S3 into fixed slots, or local."""
    root = cfg.get("data_path")
    if root and os.path.isdir(root) and os.path.exists(
            os.path.join(root, stem + ".edf")):
        return (os.path.join(root, stem + ".edf"),
                os.path.join(root, stem + ".stg.txt"),
                os.path.join(root, stem + ".edf"))

    bucket = cfg.get("bucket", DEFAULT_BUCKET)
    prefix = cfg.get("key_prefix", DEFAULT_PREFIX).rstrip("/")
    region = cfg.get("region")
    slots = temp_paths(cfg.get("temp_dir") or root, ".edf", ".txt")

    # Annotation FIRST. It is 7-8 KB against an 80-165 MB EDF, and it decides
    # whether the EDF is worth fetching at all. Flavour depends on the visit
    # (.stg.txt through visit 2, .allscore.txt from visit 3) but we try both
    # rather than deriving it from the visit number, which would break on any
    # exception to the pattern.
    last = None
    got_ann = False
    for suffix in (".stg.txt", ".allscore.txt"):
        try:
            download_to(bucket, f"{prefix}/{stem}{suffix}", slots[".txt"], region)
            got_ann = True
            break
        except S3ObjectMissing as err:
            last = err
    if not got_ann:
        raise last
    if not _has_scored_stages(slots[".txt"]):
        raise ValueError(f"{stem}: annotation contains no scored stage; "
                         f"skipping before the EDF download")

    download_to(bucket, f"{prefix}/{stem}.edf", slots[".edf"], region)
    return slots[".edf"], slots[".txt"], f"s3://{bucket}/{prefix}/{stem}.edf"


def iter_recordings(spec, cfg) -> Iterator[RawRecording]:
    imports = pd.read_csv(cfg["import_csv"])

    # WSC's montage changed across visits, and the difference is total rather
    # than cosmetic: sampling EDF headers by ranged GET showed visit 1 carries
    # C3_M2 and O1_M2 with NO frontal lead at all (0 of 6 sampled), visit 2 is
    # mixed (3 of 6), and visits 3-5 reliably carry F3_M2 (14 of 14). Without
    # this filter the first 100 table rows are 42% visit 1, every one of which
    # would be fetched in full and then skipped for want of a channel.
    visits = cfg.get("visits")
    if visits:
        keep = imports["wsc_vst"].astype(int).isin([int(v) for v in visits])
        log.info("WSC: visits %s -> %d of %d table rows",
                 sorted(int(v) for v in visits), int(keep.sum()), len(imports))
        imports = imports[keep].reset_index(drop=True)

    limit = cfg.get("limit")
    if limit:
        imports = imports.iloc[:limit]
    log.info("WSC: %d recordings to attempt", len(imports))

    done = set(cfg.get("_done_rec_ids") or ())
    for _, row in imports.iterrows():
        # format_record_id, not str(): iterrows collapses the row to one dtype,
        # so numeric ids arrive as floats and str() would give '12345.0'.
        wsc_id = format_record_id(row["wsc_id"])
        visit = format_record_id(row["wsc_vst"])
        stem = f"wsc-visit{visit}-{wsc_id}-nsrr"
        if f"WSC_{wsc_id}_v{visit}" in done:
            continue                      # skip before transferring ~90 MB

        try:
            edf_path, stg_path, provenance = _fetch(cfg, stem)
            signal, names, fs = load_edf_channels(edf_path, spec)
            hypnogram = _read_annotations(stg_path, edf_path)
        except S3ObjectMissing as err:
            # Not every table row has an object in the bucket; expected.
            log.info("WSC %s: %s", stem, err)
            log_skip(spec.name, stem)
            continue
        except Exception:  # noqa: BLE001 — one bad file must not end the cohort
            log_skip(spec.name, stem)
            continue

        n_leads = signal.shape[1]
        yield RawRecording(
            rec_id=f"WSC_{wsc_id}_v{visit}",
            cohort=spec.name,
            subject_id=wsc_id,           # visits of one subject share this id
            montage_family=spec.montage_family,
            channel_names=names,
            signal=signal,
            fs=fs,
            hypnogram=hypnogram,
            source_path=provenance,
            age=row.get(cfg.get("age_column", "age")),
            sex=row.get(cfg.get("sex_column", "sex")),
            notes=f"{n_leads}_frontal_lead" + ("s" if n_leads != 1 else ""),
        )
