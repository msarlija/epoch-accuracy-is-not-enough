# -*- coding: utf-8 -*-
"""Cohort balancing (per-cohort record caps) and subject-level splits.

§3.11: cap each cohort at a comparable number of RECORDINGS so no dataset
holds a majority. Subsampling happens at the recording level BY SUBJECT
(never splitting a subject's nights across the cap boundary), with a logged
seed, random — not performance-based. Selected ids are recorded in the run's
split manifest. No additional sampler weighting: caps + global shuffle is
sufficient.
"""

import json
import logging
import os
import zlib
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def apply_cohort_caps(manifest: pd.DataFrame, caps: Dict[str, int],
                      seed: int) -> pd.DataFrame:
    """Return the capped manifest subset.

    ``caps``: {cohort: max_recordings}; cohorts absent from the dict are kept
    whole. Subjects are drawn in random order and each subject's recordings
    are taken as a block until the cap is reached — a subject whose nights
    would straddle the cap is skipped in favour of smaller ones, never split.
    """
    rng = np.random.default_rng(seed)
    kept = []
    for cohort, group in manifest.groupby("cohort", sort=True):
        cap = caps.get(cohort)
        if cap is None or len(group) <= cap:
            kept.append(group)
            continue
        subjects = np.asarray(sorted(group["subject_id"].astype(str).unique()))
        rng.shuffle(subjects)
        chosen, total = [], 0
        for subj in subjects:
            block = group[group["subject_id"].astype(str) == subj]
            if total + len(block) > cap:
                continue  # never split a subject's nights across the cap
            chosen.append(block)
            total += len(block)
            if total == cap:
                break
        kept.append(pd.concat(chosen) if chosen else group.iloc[0:0])
    return pd.concat(kept).reset_index(drop=True)


def split_by_subject(manifest: pd.DataFrame, val_frac: float,
                     test_frac: float, seed: int
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Subject-level train/val/test split, stratified by cohort.

    All recordings of one subject land in the same split.
    """
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, group in manifest.groupby("cohort", sort=True):
        subjects = np.asarray(sorted(group["subject_id"].astype(str).unique()))
        rng.shuffle(subjects)
        n = len(subjects)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test_s = set(subjects[:n_test])
        val_s = set(subjects[n_test:n_test + n_val])
        subj = group["subject_id"].astype(str)
        parts["test"].append(group[subj.isin(test_s)])
        parts["val"].append(group[subj.isin(val_s)])
        parts["train"].append(group[~subj.isin(test_s | val_s)])
    out = tuple(pd.concat(parts[k]).reset_index(drop=True)
                for k in ("train", "val", "test"))
    return out


def save_split_manifest(run_dir: str, splits: Dict[str, pd.DataFrame],
                        caps: Optional[Dict[str, int]], seed: int) -> str:
    """Write exact record ids per split (§3.16) + the sampling parameters."""
    payload = {
        "seed": seed,
        "caps": caps or {},
        "splits": {name: sorted(df["rec_id"].astype(str).tolist())
                   for name, df in splits.items()},
    }
    path = os.path.join(run_dir, "split_manifest.json")
    os.makedirs(run_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# k-fold, subject-level, cohort-stratified (Experiment 1)
# ---------------------------------------------------------------------------

def _cohort_rng(seed: int, cohort: str) -> np.random.Generator:
    """Per-cohort generator, so one cohort's fold assignment does not depend on
    which OTHER cohorts happen to be present in the manifest.

    ``zlib.crc32``, not ``hash()``: Python salts str hashing per process
    (PYTHONHASHSEED), so ``hash()`` would produce a different partition on every
    invocation and silently break the "identical partition for a given
    (k, seed)" guarantee that pairing the arms per recording depends on.
    """
    return np.random.default_rng([seed, zlib.crc32(cohort.encode())])


def subject_folds(manifest: pd.DataFrame, k: int = 10,
                  seed: int = 0) -> Dict[Tuple[str, str], int]:
    """``{(cohort, subject_id): fold}`` — subject-level, cohort-stratified.

    Keyed on (cohort, subject_id) rather than subject_id alone because several
    cohorts use bare numeric subject ids, which collide across cohorts.
    """
    if k < 3:
        raise ValueError(f"k must be >= 3 (train + val + test), got {k}")
    out: Dict[Tuple[str, str], int] = {}
    for cohort, group in manifest.groupby("cohort", sort=True):
        subjects = np.asarray(sorted(group["subject_id"].astype(str).unique()))
        _cohort_rng(seed, str(cohort)).shuffle(subjects)
        if len(subjects) < k:
            log.warning("cohort %s has %d subjects for k=%d: %d fold(s) will "
                        "contain none of it", cohort, len(subjects), k,
                        k - len(subjects))
        # Index-slicing a seeded shuffle, NOT int(round(n * frac)) as
        # split_by_subject does: the fraction form rounds to 0 for a small
        # cohort at k=10 and would drop it from the test stratum without
        # saying so. array_split keeps fold sizes within 1 of each other.
        for f, part in enumerate(np.array_split(subjects, k)):
            for subj in part:
                out[(str(cohort), str(subj))] = f
    return out


def fold_of_k(manifest: pd.DataFrame, fold: int, k: int = 10, seed: int = 0
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fold ``fold`` of a subject-level, cohort-stratified k-fold split.

    ``test = fold``, ``val = (fold + 1) % k``, ``train`` = the other k-2. Every
    recording of a subject moves together, every cohort appears in every fold,
    and the partition depends only on ``(k, seed)`` — never on ``fold`` or on the
    arm — so across the k runs each recording is tested exactly once and the
    arms pair exactly per recording.

    Returns the same ``(train, val, test)`` 3-tuple as :func:`split_by_subject`,
    so callers need no other change.
    """
    if not 0 <= fold < k:
        raise ValueError(f"fold must be in [0, {k}), got {fold}")
    folds = subject_folds(manifest, k=k, seed=seed)
    assigned = np.array([folds[(c, s)] for c, s in
                         zip(manifest["cohort"].astype(str),
                             manifest["subject_id"].astype(str))])
    test_f, val_f = fold, (fold + 1) % k
    parts = (manifest[(assigned != test_f) & (assigned != val_f)],
             manifest[assigned == val_f],
             manifest[assigned == test_f])
    return tuple(df.reset_index(drop=True) for df in parts)
