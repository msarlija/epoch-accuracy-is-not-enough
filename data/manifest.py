# -*- coding: utf-8 -*-
"""Recording manifest: one row per recording, emitted DURING preprocessing.

Columns (per §3.16): rec_id, cohort, montage_family, n_channels,
channel_names, subject_id, age, sex, qc_pass, qc_channel_flags, source_path,
n_epochs, n_samples, mean_label_purity, min_label_purity, per-channel
median/iqr (JSON lists), preprocessing parameter hash, reader notes.

Stored as ``manifest.csv`` next to the arrays. Append-safe so
prepare_data.py can resume.
"""

import hashlib
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

MANIFEST_NAME = "manifest.csv"

COLUMNS = [
    "rec_id", "cohort", "montage_family", "n_channels", "channel_names",
    "subject_id", "age", "sex", "qc_pass", "qc_channel_flags", "source_path",
    "n_epochs", "n_samples", "mean_label_purity", "min_label_purity",
    "median", "iqr", "preproc_hash", "notes",
]


def preproc_hash(cfg: dict) -> str:
    """Stable short hash of the preprocessing parameters, stored per row so a
    stale array written under different parameters is detectable."""
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def manifest_path(data_dir: str) -> str:
    return os.path.join(data_dir, MANIFEST_NAME)


def load_manifest(data_dir: str) -> pd.DataFrame:
    path = manifest_path(data_dir)
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for col in ("median", "iqr", "qc_channel_flags", "channel_names"):
        if col in df:
            df[col] = df[col].apply(
                lambda v: json.loads(v) if isinstance(v, str) else v)
    return df


def append_row(data_dir: str, row: dict) -> None:
    path = manifest_path(data_dir)
    row = dict(row)
    for col in ("median", "iqr", "qc_channel_flags", "channel_names"):
        if col in row and not isinstance(row[col], str):
            val = row[col]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            row[col] = json.dumps(val)
    df = pd.DataFrame([row], columns=COLUMNS)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def existing_rec_ids(data_dir: str) -> set:
    df = load_manifest(data_dir)
    return set(df["rec_id"].astype(str)) if len(df) else set()


# ---------------------------------------------------------------------------
# Array file naming (one continuous float32 array + int8 labels + purity)
# ---------------------------------------------------------------------------

def array_paths(data_dir: str, rec_id: str) -> dict:
    base = os.path.join(data_dir, rec_id)
    return {"x": base + "_x.npy",        # (n_samples, C) float32, filtered µV
            "y": base + "_y.npy",        # (n_epochs,) int8
            "purity": base + "_purity.npy"}  # (n_epochs,) float32


def save_arrays(data_dir: str, rec_id: str, signal: np.ndarray,
                labels: np.ndarray, purity: np.ndarray) -> None:
    paths = array_paths(data_dir, rec_id)
    np.save(paths["x"], np.ascontiguousarray(signal, dtype=np.float32))
    np.save(paths["y"], np.ascontiguousarray(labels, dtype=np.int8))
    np.save(paths["purity"], np.ascontiguousarray(purity, dtype=np.float32))


def open_arrays(data_dir: str, rec_id: str, mmap: bool = True) -> dict:
    paths = array_paths(data_dir, rec_id)
    mode: Optional[str] = "r" if mmap else None
    return {"x": np.load(paths["x"], mmap_mode=mode),
            "y": np.load(paths["y"]),
            "purity": np.load(paths["purity"])}
