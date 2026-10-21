# -*- coding: utf-8 -*-
"""DOD-H (Dreem Open Dataset — Healthy) reader.

Per the task spec: read the h5 files DIRECTLY — secondary sources disagree on
the channel list. Each record ships Fp1-M2 / Fp2-M1 among its EEG derivations
and five independent expert scorings.

h5 layout (dreem-learning-open distribution):
    /signals/eeg/<CHANNEL>   float arrays, attrs['fs'] = sampling rate
    /hypnogram               consensus labels, one per 30 s, -1 = unscored
Per-scorer hypnograms are distributed separately as JSON lists
(<scorers_dir>/scorer_<k>/<record>.json). This reader is defensive about the
exact group names because distributions differ; it searches the file for the
configured channel names.

Stage coding in DOD files: -1 unscored, 0 W, 1 N1, 2 N2, 3 N3, 4 REM —
already our common scheme.

Config keys: data_path (dir of .h5 files), optional scorers_dir, optional
scorer ('consensus' [default] = majority vote across scorer files when
present, else the /hypnogram dataset; or a scorer subdir name), optional
limit.
"""

import json
import logging
import os
from typing import Iterator, Optional

import numpy as np

from .base import Hypnogram, RawRecording, UNKNOWN

log = logging.getLogger(__name__)

EPOCH_SEC = 30.0


def _find_channel(h5file, name: str):
    """Depth-first search for a dataset whose leaf name matches ``name``
    (case-insensitive, '-'/'_' interchangeable)."""
    import h5py

    want = name.lower().replace("-", "_")
    found = []

    def visit(path, obj):
        if isinstance(obj, h5py.Dataset):
            leaf = path.rsplit("/", 1)[-1].lower().replace("-", "_")
            if leaf == want:
                found.append(path)

    h5file.visititems(visit)
    return found[0] if found else None


def _dataset_fs(h5file, path: str) -> Optional[float]:
    import h5py

    obj = h5file[path]
    for holder in (obj, obj.parent, h5file):
        if isinstance(holder, (h5py.Dataset, h5py.Group)) and "fs" in holder.attrs:
            return float(holder.attrs["fs"])
    return None


def _majority_vote(hypnograms: np.ndarray) -> np.ndarray:
    """(n_scorers, n_epochs) -> consensus; ties -> label of the
    lowest-numbered scorer among the tied winners; all-unknown -> -1."""
    n_scorers, n_epochs = hypnograms.shape
    out = np.full(n_epochs, UNKNOWN, int)
    for e in range(n_epochs):
        votes = hypnograms[:, e]
        votes_valid = votes[votes != UNKNOWN]
        if votes_valid.size == 0:
            continue
        vals, counts = np.unique(votes_valid, return_counts=True)
        winners = vals[counts == counts.max()]
        if len(winners) == 1:
            out[e] = winners[0]
        else:
            for v in votes:  # first scorer's vote among tied winners
                if v in winners:
                    out[e] = v
                    break
    return out


def iter_recordings(spec, cfg) -> Iterator[RawRecording]:
    import h5py

    root = cfg["data_path"]
    scorers_dir = cfg.get("scorers_dir")
    scorer = cfg.get("scorer", "consensus")
    files = sorted(f for f in os.listdir(root) if f.endswith(".h5"))
    limit = cfg.get("limit")
    if limit:
        files = files[:limit]

    for fname in files:
        rec = os.path.splitext(fname)[0]
        path = os.path.join(root, fname)
        with h5py.File(path, "r") as h5:
            resolved = None
            for cand in spec.candidate_channel_specs():
                paths = [_find_channel(h5, n) for n in cand.names]
                if all(p is not None for p in paths):
                    resolved = (cand, paths)
                    break
            if resolved is None:
                log.warning("DODH %s: none of the candidate channel sets "
                            "found; file structure keys: %s",
                            rec, list(h5.keys()))
                continue
            cand, paths = resolved
            fs = _dataset_fs(h5, paths[0]) or float(cfg.get("fs", 250.0))
            signal = np.stack(
                [np.asarray(h5[p]) * pol for p, pol in zip(paths, cand.polarity)],
                axis=1).astype(np.float32)

            consensus = (np.asarray(h5["hypnogram"], int)
                         if "hypnogram" in h5 else None)

        # per-scorer hypnograms
        extra = {}
        if scorers_dir and os.path.isdir(scorers_dir):
            for sub in sorted(os.listdir(scorers_dir)):
                sc_path = os.path.join(scorers_dir, sub, rec + ".json")
                if os.path.exists(sc_path):
                    with open(sc_path, "r") as f:
                        labels = np.asarray(json.load(f), int)
                    extra[sub] = Hypnogram.from_grid(labels, EPOCH_SEC)

        if scorer == "consensus":
            if extra:
                n = min(len(h.labels) for h in extra.values())
                stack = np.stack([h.labels[:n] for h in extra.values()])
                labels = _majority_vote(stack)
            elif consensus is not None:
                labels = consensus
            else:
                log.warning("DODH %s: no hypnogram found, skipping", rec)
                continue
        else:
            if scorer not in extra:
                log.warning("DODH %s: scorer '%s' not found, skipping",
                            rec, scorer)
                continue
            labels = extra[scorer].labels

        yield RawRecording(
            rec_id=f"{spec.name}_{rec}",
            cohort=spec.name,
            subject_id=rec,
            montage_family=spec.montage_family,
            channel_names=list(cand.names),
            signal=signal,
            fs=fs,
            hypnogram=Hypnogram.from_grid(labels, EPOCH_SEC),
            source_path=path,
            extra_hypnograms=extra,
        )
