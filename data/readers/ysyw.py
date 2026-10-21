# -*- coding: utf-8 -*-
"""YSYW reader (PhysioNet/CinC 2018, "you snooze you win").

wfdb records under <root>/<record>/<record>.{hea,mat,arousal}. Frontal
derivations F3-M2 / F4-M1 are named signals in the record. Sleep stages are
change-point events in the 'arousal' annotation stream (aux_note in
{W, N1, N2, N3, R}); each stage runs until the next stage event.

Config keys: data_path, records_file (the RECORDS listing), optional
age_sex_csv (columns Record, Age, Sex), optional limit / offset.
"""

import os
from typing import Iterator

import numpy as np
import pandas as pd

from .base import (Hypnogram, RawRecording, N1, N2, N3, REM, WAKE,
                   UNKNOWN, log_skip)

STAGE_MAP = {"W": WAKE, "N1": N1, "N2": N2, "N3": N3, "R": REM}


def _stage_hypnogram(ann, fs: float, total_sec: float) -> Hypnogram:
    onsets, labels = [], []
    for sample, note in zip(ann.sample, ann.aux_note):
        note = note.strip().lstrip("(").rstrip(")")
        if note in STAGE_MAP:
            onsets.append(sample / fs)
            labels.append(STAGE_MAP[note])
    onsets = np.asarray(onsets, float)
    labels = np.asarray(labels, int)
    # each stage lasts until the next stage event; the last until record end
    ends = np.append(onsets[1:], total_sec)
    durations = ends - onsets
    keep = durations > 0
    return Hypnogram(onsets[keep], durations[keep], labels[keep])


def iter_recordings(spec, cfg) -> Iterator[RawRecording]:
    import wfdb

    root = cfg["data_path"]
    records = pd.read_csv(cfg["records_file"], header=None)[0]
    records = [str(r).strip().strip("/") for r in records]
    offset = cfg.get("offset", 0)
    limit = cfg.get("limit")
    records = records[offset:offset + limit] if limit else records[offset:]

    meta = None
    if cfg.get("age_sex_csv"):
        meta = pd.read_csv(cfg["age_sex_csv"]).set_index("Record")

    wanted = list(spec.channels.names)
    for rec in records:
        rec_path = os.path.join(root, rec, rec)
        try:
            # channel_names= is load-bearing for throughput, not just tidiness:
            # these records hold 13 signals in a ~134 MB .mat, and rdrecord
            # converts every one of them to physical float64 (~750 MB peak)
            # before we discard 11. Asking for two cuts the read ~6x.
            record = wfdb.rdrecord(rec_path, channel_names=wanted)
            fs = float(record.fs)
            if list(record.sig_name) != wanted:
                raise LookupError(
                    f"YSYW {rec}: asked for {wanted}, got {record.sig_name}")
            signal = record.p_signal.astype(np.float32)  # already µV

            ann = wfdb.rdann(rec_path, "arousal")
            hypnogram = _stage_hypnogram(ann, fs, len(signal) / fs)
        except Exception:  # noqa: BLE001 — one bad file must not end the cohort
            log_skip(spec.name, rec)
            continue

        age = sex = None
        if meta is not None and rec in meta.index:
            age = meta.loc[rec].get("Age")
            sex = meta.loc[rec].get("Sex")

        yield RawRecording(
            rec_id=f"YSYW_{rec}",
            cohort=spec.name,
            subject_id=rec,
            montage_family=spec.montage_family,
            channel_names=wanted,
            signal=signal,
            fs=fs,
            hypnogram=hypnogram,
            source_path=rec_path,
            age=age,
            sex=sex,
        )
