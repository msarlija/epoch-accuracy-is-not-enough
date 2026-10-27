# -*- coding: utf-8 -*-
"""ANPHY reader.

Layout: <root>/<subject_id>/ containing the signals and one annotation TXT with
lines '<stage> <start_sec> ...'. Stage coding: W/N1/N2/N3/R, 'L' = movement /
unknown. Subjects CSV: 'Details information for healthy subjects.csv' with
columns 'Subjects ID', 'Age', 'Sex'.

Signals come in one of two forms and this reader accepts both:

* an EDF, read via the shared mne path like every other EDF cohort. This is
  ``<subject>_Fp-TP_250Hz.edf`` written by ``reextract_anphy.py``: the released
  EEG is referenced to the amplifier's central system reference, so the script
  re-references to Fp1-TP10 and Fp2-TP9 and resamples 1000 -> 250 Hz;
* ``<subject>_FP.csv`` — 'time,Fp1,Fp2' at 1000 Hz with the original central
  reference, from an older pipeline. The cohort spec no longer names these
  channels, so a folder holding only the CSV is skipped rather than silently
  prepared with the wrong reference. The EDF is preferred when present.
"""

import os
from typing import Iterator

import numpy as np
import pandas as pd

from .base import (Hypnogram, RawRecording, N1, N2, N3, REM, WAKE,
                   UNKNOWN, log_skip, resolve_channels)
from .edf_utils import load_edf_channels

STAGE_MAP = {"W": WAKE, "N1": N1, "N2": N2, "N3": N3, "R": REM, "L": UNKNOWN}


def _read_hypnogram(path: str, epoch_sec: float) -> Hypnogram:
    onsets, labels = [], []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels.append(STAGE_MAP.get(parts[0], UNKNOWN))
                onsets.append(float(parts[1]))
    onsets = np.asarray(onsets, float)
    durations = np.full(len(onsets), epoch_sec)
    if len(onsets) > 1:  # trust the file's own spacing where present
        diffs = np.diff(onsets)
        durations[:-1] = diffs
        durations[-1] = np.median(diffs)
    return Hypnogram(onsets, durations, np.asarray(labels, int))


def _read_fp_csv(path: str, spec):
    """Read a 'time,Fp1,Fp2' CSV, returning ``extract_channels``' contract:
    ``(signal (n,C) float32 µV with polarity applied, names, fs)``.

    ``fs`` is derived from the time column rather than assumed. That matters
    because ``resample_to_target`` computes ``Fraction(target_fs / fs)`` and
    nothing downstream can tell a wrong rate from a right one — a bad fs
    silently yields a whole corpus resampled to the wrong frequency. 1000 Hz
    here means an exact up 1 / down 10 to the 100 Hz target.
    """
    head = pd.read_csv(path, nrows=1000)
    resolved = resolve_channels(list(head.columns),
                               spec.candidate_channel_specs())
    if resolved is None:
        raise LookupError(
            f"No candidate channel set of cohort {spec.name} in {path}. "
            f"Available columns: {list(head.columns)}")
    if resolved.is_derivation:
        raise NotImplementedError(
            f"{spec.name}: CSV signals do not support reference derivations")
    names, polarity = list(resolved.names), resolved.polarity

    if "time" not in head.columns:
        raise ValueError(f"{path}: no 'time' column to derive fs from")
    dt = float(np.median(np.diff(head["time"].to_numpy(dtype=float))))
    if not dt > 0:
        raise ValueError(f"{path}: non-increasing time column (median dt={dt})")
    fs = 1.0 / dt
    fs_int = float(round(fs))
    if abs(fs - fs_int) > 1e-3 * fs_int:
        raise ValueError(f"{path}: irregular time column, inferred fs={fs:.4f}")

    # usecols drops the time column from the full read — a third of both the
    # parse cost and the peak memory on a 1.3 GB file.
    names = list(names)
    frame = pd.read_csv(path, usecols=names,
                        dtype={n: np.float32 for n in names})
    signal = frame[names].to_numpy(dtype=np.float32)
    signal *= np.asarray(polarity, dtype=np.float32)
    return signal, names, fs_int


def iter_recordings(spec, cfg) -> Iterator[RawRecording]:
    root = cfg["data_path"]
    imports = pd.read_csv(cfg["import_csv"])
    limit = cfg.get("limit")
    if limit:
        imports = imports.iloc[:limit]

    for _, row in imports.iterrows():
        subj = str(row["Subjects ID"])
        subj_dir = os.path.join(root, subj)

        try:
            entries = os.listdir(subj_dir)
            edfs = [f for f in entries if f.lower().endswith(".edf")]
            txts = [f for f in entries if f.lower().endswith(".txt")]
            if not txts:
                raise FileNotFoundError(
                    f"ANPHY {subj}: no annotation TXT in {subj_dir}")

            if edfs:
                source_path = os.path.join(subj_dir, edfs[0])
                signal, names, fs = load_edf_channels(source_path, spec)
            else:
                source_path = os.path.join(subj_dir, f"{subj}_FP.csv")
                signal, names, fs = _read_fp_csv(source_path, spec)

            hypnogram = _read_hypnogram(os.path.join(subj_dir, txts[0]),
                                        spec.native_epoch_sec or 30.0)
        except Exception:  # noqa: BLE001 — one bad subject must not end the cohort
            log_skip(spec.name, subj)
            continue

        yield RawRecording(
            rec_id=f"ANPHY_{subj}",
            cohort=spec.name,
            subject_id=subj,
            montage_family=spec.montage_family,
            channel_names=names,
            signal=signal,
            fs=fs,
            hypnogram=hypnogram,
            source_path=source_path,
            age=row.get("Age"),
            sex=row.get("Sex"),
        )
