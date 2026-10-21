# -*- coding: utf-8 -*-
"""Reader for NSRR-style cohorts: CFS, HomePAP (HPAP), MrOS, SOF.

These share the NSRR annotation XML ('annotations-events-nsrr') where sleep
stages appear as::

    <ScoredEvent>
      <EventType>Stages|Stages</EventType>
      <EventConcept>Stage 2 sleep|2</EventConcept>
      <Start>1230.0</Start>
      <Duration>300.0</Duration>
    </ScoredEvent>

The old repo parsed these with string offsets and expanded them to a fixed
grid; here we use a real XML parser and keep native onsets/durations for the
30 s harmonizer. Stage code 4 (R&K S4) maps into N3, code 5 is REM.

Config keys (per cohort, in preprocess.yaml)::

    import_csv:  CSV with one row per recording
    id_column:   column holding the recording id
    edf_pattern: format string -> EDF path,        e.g. ".../cfs-visit5-{id}.edf"
    ann_pattern: format string -> XML path
    id_format:   optional python format spec applied to the id (e.g. "05d"
                 zero-pads SOF ids the way the old code did with zfill(5))
    age_column / sex_column: optional metadata columns
    limit:       optional row cap
"""

import xml.etree.ElementTree as ET
from typing import Iterator

import numpy as np
import pandas as pd

from .base import (Hypnogram, RawRecording, N1, N2, N3, REM, WAKE, UNKNOWN,
                   format_record_id, log_skip)
from .edf_utils import load_edf_channels

NSRR_STAGE_MAP = {0: WAKE, 1: N1, 2: N2, 3: N3, 4: N3, 5: REM}


def parse_nsrr_stage_xml(path: str) -> Hypnogram:
    onsets, durations, labels = [], [], []
    root = ET.parse(path).getroot()
    for ev in root.iter("ScoredEvent"):
        etype = ev.findtext("EventType") or ""
        if "Stages" not in etype:
            continue
        concept = ev.findtext("EventConcept") or ""
        try:
            code = int(concept.rsplit("|", 1)[-1])
        except ValueError:
            code = None
        onsets.append(float(ev.findtext("Start")))
        durations.append(float(ev.findtext("Duration")))
        labels.append(NSRR_STAGE_MAP.get(code, UNKNOWN))
    order = np.argsort(onsets)
    return Hypnogram(np.asarray(onsets, float)[order],
                     np.asarray(durations, float)[order],
                     np.asarray(labels, int)[order])


def iter_recordings(spec, cfg) -> Iterator[RawRecording]:
    imports = pd.read_csv(cfg["import_csv"])
    limit = cfg.get("limit")
    if limit:
        imports = imports.iloc[:limit]
    id_col = cfg["id_column"]
    id_format = cfg.get("id_format")

    for _, row in imports.iterrows():
        rec = format_record_id(row[id_col], id_format)
        edf_path = cfg["edf_pattern"].format(id=rec)
        ann_path = cfg["ann_pattern"].format(id=rec)

        try:
            signal, names, fs = load_edf_channels(edf_path, spec)
            hypnogram = parse_nsrr_stage_xml(ann_path)
        except Exception:  # noqa: BLE001 — one bad file must not end the cohort
            log_skip(spec.name, rec)
            continue

        yield RawRecording(
            rec_id=f"{spec.name}_{rec}",
            cohort=spec.name,
            subject_id=rec,
            montage_family=spec.montage_family,
            channel_names=names,
            signal=signal,
            fs=fs,
            hypnogram=hypnogram,
            source_path=edf_path,
            age=row.get(cfg.get("age_column", "Age")),
            sex=row.get(cfg.get("sex_column", "Sex")),
        )
