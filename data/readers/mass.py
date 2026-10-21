# -*- coding: utf-8 -*-
"""MASS reader.

Layout (as in the Keras repo):
    <root>/SS<k>_EDF/<nightcode> PSG.edf
    <root>/SS2_Annotations/<nightcode> Base.edf          (SS2 uses 'Base')
    <root>/SS<k>_Annotations/<nightcode> Annotations.edf (others)

Import CSV: columns 'Nightcode' (e.g. 01-01-0002), optional 'Age', 'Sex'.

``rec_prefix`` (default 'MASS') names the stored arrays. It exists so a second
MASS cohort can be prepared alongside the first without rec_id collisions: SS1
is read twice under different channel specs -- once as prefrontal (the 3 files
that carry Fp) and once as a frontal cohort over all 53 -- and the two must not
overwrite each other's arrays.
Nightcode digit 4 selects the subset (SS1..SS5). Subject id = nightcode
(one night per subject in MASS).
"""

import os
from typing import Iterator

import numpy as np
import pandas as pd

from .base import (Hypnogram, RawRecording, N1, N2, N3, REM, WAKE,
                   UNKNOWN, log_skip)
from .edf_utils import load_edf_channels

STAGE_MAP = {
    "Sleep stage ?": UNKNOWN,
    "Sleep stage W": WAKE,
    "Sleep stage 1": N1,
    "Sleep stage 2": N2,
    "Sleep stage 3": N3,
    "Sleep stage 4": N3,   # R&K S4 collapsed into N3
    "Sleep stage R": REM,
}


def _parse_annotations(anns) -> Hypnogram:
    """Keep only sleep-stage events; keep true onsets AND durations
    (MASS SS2/SS4/SS5 are scored on a 20 s R&K grid)."""
    onsets, durations, labels = [], [], []
    for onset, duration, desc in zip(anns.onset, anns.duration, anns.description):
        if "Sleep stage" not in desc:
            continue
        onsets.append(round(float(onset)))
        durations.append(round(float(duration)))
        labels.append(STAGE_MAP.get(desc, UNKNOWN))
    return Hypnogram(np.asarray(onsets, float), np.asarray(durations, float),
                     np.asarray(labels, int))


def iter_recordings(spec, cfg) -> Iterator[RawRecording]:
    import mne

    root = cfg["data_path"]
    imports = pd.read_csv(cfg["import_csv"])
    limit = cfg.get("limit")
    if limit:
        imports = imports.iloc[:limit]

    for _, row in imports.iterrows():
        nightcode = str(row["Nightcode"])
        subset = "SS" + nightcode[4]
        edf_path = os.path.join(root, f"{subset}_EDF", f"{nightcode} PSG.edf")
        ann_name = "Base" if subset == "SS2" else "Annotations"
        ann_path = os.path.join(root, f"{subset}_Annotations",
                                f"{nightcode} {ann_name}.edf")

        try:
            signal, names, fs = load_edf_channels(edf_path, spec, substring=True)
            anns = mne.read_annotations(ann_path)
        except Exception:  # noqa: BLE001 — one bad file must not end the cohort
            log_skip(spec.name, nightcode)
            continue

        yield RawRecording(
            rec_id=f"{cfg.get('rec_prefix', 'MASS')}_{nightcode}",
            cohort=spec.name,
            subject_id=nightcode,
            montage_family=spec.montage_family,
            channel_names=names,
            signal=signal,
            fs=fs,
            hypnogram=_parse_annotations(anns),
            source_path=edf_path,
            age=row.get("Age"),
            sex=row.get("Sex"),
        )
