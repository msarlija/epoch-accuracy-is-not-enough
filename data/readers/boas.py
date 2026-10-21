# -*- coding: utf-8 -*-
"""BOAS (BitBrain Open Access Sleep) reader.

BIDS-ish layout:
    <root>/<participant_id>/eeg/<participant_id>_task-Sleep_acq-headband_eeg.edf
    <root>/<participant_id>/eeg/<participant_id>_task-Sleep_acq-psg_events.tsv

We load the HEADBAND channels (HB_1/HB_2) but the PSG-derived human scoring
('stage_hum' column), exactly like the old pipeline.

Stage coding in 'stage_hum': the old pipeline accepted both 4 and 5 as REM
(its global remap y==5 -> 4 made them equivalent); we map both to REM here.
Verify against the dataset docs on first run if BOAS scoring ever
distinguishes them.

Config keys: data_path, participants_csv (columns participant_id, pid, age, sex),
optional limit.
"""

import os
from typing import Iterator

import numpy as np
import pandas as pd

from .base import (Hypnogram, RawRecording, N1, N2, N3, REM, WAKE,
                   UNKNOWN, log_skip)
from .edf_utils import load_edf_channels

STAGE_MAP = {0: WAKE, 1: N1, 2: N2, 3: N3, 4: REM, 5: REM}


def iter_recordings(spec, cfg) -> Iterator[RawRecording]:
    root = cfg["data_path"]
    imports = pd.read_csv(cfg["participants_csv"])
    limit = cfg.get("limit")
    if limit:
        imports = imports.iloc[:limit]

    for _, row in imports.iterrows():
        rec = str(row["participant_id"])          # per-RECORDING id, 'sub-N'
        # participants.csv also carries `pid`, the PERSON. They are not 1:1:
        # 128 recordings come from 100 people, 20 of whom appear 2-3 times
        # (pid 83 -> sub-32, sub-35, sub-41). Using participant_id as the
        # subject would let split_by_subject place the same person in train and
        # test, which is the leak that machinery exists to prevent.
        person = row.get("pid")
        subject = f"pid{person}" if person is not None and str(person) != "nan" \
            else rec
        eeg_dir = os.path.join(root, rec, "eeg")
        edf_path = os.path.join(eeg_dir, f"{rec}_task-Sleep_acq-headband_eeg.edf")
        events_path = os.path.join(eeg_dir, f"{rec}_task-Sleep_acq-psg_events.tsv")

        try:
            signal, names, fs = load_edf_channels(edf_path, spec)

            events = pd.read_csv(events_path, sep="\t")
            onsets = events["onset"].to_numpy(dtype=float)
            codes = events["stage_hum"].to_numpy()
            labels = np.asarray(
                [STAGE_MAP.get(int(c), UNKNOWN) if pd.notna(c) else UNKNOWN
                 for c in codes], int)
            durations = np.full(len(onsets), spec.native_epoch_sec or 30.0)
            if len(onsets) > 1:
                durations[:-1] = np.diff(onsets)
                durations[-1] = float(np.median(np.diff(onsets)))
        except Exception:  # noqa: BLE001 — one bad file must not end the cohort
            log_skip(spec.name, rec)
            continue

        yield RawRecording(
            rec_id=f"BOAS_{rec}",
            cohort=spec.name,
            subject_id=subject,
            montage_family=spec.montage_family,
            channel_names=names,
            signal=signal,
            fs=fs,
            hypnogram=Hypnogram(onsets, durations, labels),
            source_path=edf_path,
            age=row.get("age"),
            sex=row.get("sex"),
        )
