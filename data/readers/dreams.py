# -*- coding: utf-8 -*-
"""DREAMS reader (subjects + patients databases).

EDF: <root>/DatabaseSubjects|DatabasePatients/<name>.edf
Hypnogram: HypnogramAASM_<name>.txt next to it — one integer label per 5 s,
first line is a header. Coding (ported verbatim from the Keras repo)::

    5 = wake, 4 = REM, 3 = N1, 2 = N2, 1 = N3, 0 = N3(S4-like),
    negative = unknown

Config keys: data_path, subjects_csv, patients_csv (';'-separated, columns
'Name', 'Age', 'Sex'), optional 'limit'.
"""

import os
from typing import Iterator

import numpy as np
import pandas as pd

from .base import (Hypnogram, RawRecording, N1, N2, N3, REM, WAKE,
                   UNKNOWN, log_skip)
from .edf_utils import load_edf_channels

STAGE_MAP = {"5": WAKE, "4": REM, "3": N1, "2": N2, "1": N3,
             "0": N3, "-1": UNKNOWN, "-2": UNKNOWN, "-3": UNKNOWN}
ANN_INTERVAL_SEC = 5.0


def _read_hypnogram(path: str) -> Hypnogram:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.read().split("\n")]
    lines = [ln for ln in lines[1:] if ln]  # drop header + trailing blanks
    labels = np.asarray([STAGE_MAP.get(ln, UNKNOWN) for ln in lines], int)
    return Hypnogram.from_grid(labels, ANN_INTERVAL_SEC)


def iter_recordings(spec, cfg) -> Iterator[RawRecording]:
    root = cfg["data_path"]
    frames = []
    for key, subdir in (("subjects_csv", "DatabaseSubjects"),
                        ("patients_csv", "DatabasePatients")):
        if cfg.get(key):
            df = pd.read_csv(cfg[key], sep=";")
            df["_subdir"] = subdir
            frames.append(df)
    imports = pd.concat(frames, axis=0).reset_index(drop=True)
    limit = cfg.get("limit")
    if limit:
        imports = imports.iloc[:limit]

    for _, row in imports.iterrows():
        name = str(row["Name"])
        subdir = row["_subdir"]
        edf_path = os.path.join(root, subdir, f"{name}.edf")
        ann_path = os.path.join(root, subdir, f"HypnogramAASM_{name}.txt")
        if not os.path.exists(edf_path):  # a few files live in the other dir
            other = ("DatabasePatients" if subdir == "DatabaseSubjects"
                     else "DatabaseSubjects")
            edf_path = os.path.join(root, other, f"{name}.edf")
            ann_path = os.path.join(root, other, f"HypnogramAASM_{name}.txt")

        try:
            signal, names, fs = load_edf_channels(edf_path, spec)
            hypnogram = _read_hypnogram(ann_path)
        except Exception:  # noqa: BLE001 — one bad file must not end the cohort
            log_skip(spec.name, name)
            continue

        yield RawRecording(
            rec_id=f"DREAMS_{name}",
            cohort=spec.name,
            subject_id=name,
            montage_family=spec.montage_family,
            channel_names=names,
            signal=signal,
            fs=fs,
            hypnogram=hypnogram,
            source_path=edf_path,
            age=row.get("Age"),
            sex=row.get("Sex"),
        )
