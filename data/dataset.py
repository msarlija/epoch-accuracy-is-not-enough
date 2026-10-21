# -*- coding: utf-8 -*-
"""Flat-index memmap Dataset + augmentations.

Replaces the old Keras DataGenerator. The index is simply::

    index = [(rec_id, start_epoch)
             for rec_id in recordings
             for start_epoch in range(0, n_epochs[rec_id] - T + 1, stride)]

with global shuffling delegated to ``DataLoader(shuffle=True)``. A batch
drawn from a shuffled index over millions of entries contains distinct
recordings with overwhelming probability, so the old generator's explicit
"≥4 recordings per batch" logic is intentionally NOT ported.

Sample layout returned by ``__getitem__`` (all torch tensors):

    x            (T, C_max, 3000) float32   zero-padded above n_channels
    channel_mask (T, C_max)       bool      False = absent/dropped channel
    tso          (T,)             float32   normalized time-since-onset
    y            (T,)             int64     -1 = ignore (CE ignore_index)
    purity       (T,)             float32

Channels are padded to ``C_max`` (=2) with a mask instead of using ragged
batches, so the default collate works and mixed C=1/C=2 batches (native or
from channel dropout) are handled uniformly. The mask rows are identical
across T for native channel count but channel DROPOUT is drawn per sample,
making C=1 an in-distribution regime (§3.6).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import json
import zlib

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import array_paths, open_arrays
from .preprocess import EPOCH_SAMPLES, RecordStats, apply_normalization

TSO_HARMONICS = 3  # t + 3x(sin,cos) -> 7-D feature (see models/tso.py)


# ---------------------------------------------------------------------------
# Augmentations (train only)
# ---------------------------------------------------------------------------

@dataclass
class AugmentConfig:
    #: probability of dropping one channel per SAMPLE (only when >=2 present)
    channel_dropout_p: float = 0.3
    #: Probability of corrupting one channel over a random sub-window. Applied
    #: ONLY when at least two clean channels remain, so a sample never loses all
    #: usable signal. Effective unconditional rate is therefore
    #: (1 - channel_dropout_p) * artifact_p = 0.21 for a C=2 recording, and 0
    #: for a C=1 one.
    artifact_p: float = 0.3
    artifact_types: Sequence[str] = ("flatline", "saturation", "line_noise",
                                     "drift", "emg_burst")
    #: sub-window length as a fraction of the clip, U(min, max)
    artifact_frac: Tuple[float, float] = (0.05, 0.5)
    line_noise_hz: Sequence[float] = (50.0, 60.0)


def _inject_artifact(x: np.ndarray, ch: int, fs: float, kind: str,
                     frac_range: Tuple[float, float],
                     line_hz: Sequence[float],
                     rng: np.random.Generator) -> None:
    """Corrupt channel ``ch`` of x (n_samples, C), in place, label unchanged.

    Operates in NORMALIZED units; amplitudes are scaled to the clip's own
    robust amplitude so severity is comparable across normalization modes.
    """
    n = x.shape[0]
    frac = rng.uniform(*frac_range)
    length = max(int(n * frac), int(fs))
    start = rng.integers(0, max(n - length, 1))
    sl = slice(start, start + length)
    amp = max(float(np.percentile(np.abs(x[:, ch]), 95)), 1e-3)

    if kind == "flatline":
        x[sl, ch] = rng.normal(0.0, 0.02 * amp, length)
    elif kind == "saturation":
        level = amp * rng.uniform(2.0, 4.0)
        x[sl, ch] = np.clip(x[sl, ch] * rng.uniform(5.0, 20.0), -level, level)
    elif kind == "line_noise":
        f0 = float(rng.choice(np.asarray(line_hz)))
        t = np.arange(length) / fs
        x[sl, ch] += amp * rng.uniform(1.0, 3.0) * np.sin(
            2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
    elif kind == "drift":
        f0 = rng.uniform(0.05, 0.3)
        t = np.arange(length) / fs
        drift = np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi))
        x[sl, ch] += amp * rng.uniform(3.0, 8.0) * drift
    elif kind == "emg_burst":
        burst = rng.normal(0.0, 1.0, length)
        env = np.hanning(length)
        x[sl, ch] += amp * rng.uniform(1.5, 4.0) * burst * env
    else:
        raise ValueError(f"Unknown artifact kind '{kind}'")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class RecordEntry:
    rec_id: str
    n_epochs: int
    n_channels: int
    stats: RecordStats
    cohort: str = ""
    montage_family: str = ""
    #: Which STORED columns to read. Normally every column, but a recording
    #: whose QC flagged a channel keeps only the good ones, so ``n_channels``
    #: is the surviving count and this says where they live on disk.
    channel_index: Optional[List[int]] = None


def entries_from_manifest(manifest_df,
                          use_channel_flags: bool = True) -> List[RecordEntry]:
    """Build entries, dropping QC-flagged channels unless told otherwise.

    ``qc_channel_flags`` has been written to the manifest since the corpus was
    built but was read by NOTHING -- a channel already identified as dead was
    still handed to the network as signal. Dropping it here turns such a
    recording into a genuine C=1 recording instead.

    That is safe because C=1 is already in-distribution: WSC is single-lead by
    construction, and channel dropout zeroes a channel on 40% of training
    samples. Rejecting the whole recording instead would discard a perfectly
    good channel -- and all 46 DREAMS recordings, whose FP2 is dead.
    """
    entries = []
    for _, row in manifest_df.iterrows():
        median = np.asarray(row["median"], float).ravel()
        iqr = np.asarray(row["iqr"], float).ravel()
        n_stored = int(row["n_channels"])
        keep = list(range(n_stored))

        if use_channel_flags and "qc_channel_flags" in row.index:
            flags = row["qc_channel_flags"]
            if isinstance(flags, str):
                try:
                    flags = json.loads(flags)
                except ValueError:
                    flags = None
            if flags is not None:
                flags = list(np.atleast_1d(flags))
                if len(flags) == n_stored:
                    good = [i for i, f in enumerate(flags) if int(f) == 1]
                    # never return a channel-less recording: if QC flagged
                    # everything, keep the columns and let the record-level
                    # qc_pass decide whether it is used at all
                    if good:
                        keep = good

        entries.append(RecordEntry(
            rec_id=str(row["rec_id"]),
            n_epochs=int(row["n_epochs"]),
            n_channels=len(keep),
            stats=RecordStats(median=median[keep], iqr=iqr[keep]),
            cohort=str(row.get("cohort", "")),
            montage_family=str(row.get("montage_family", "")),
            channel_index=keep if keep != list(range(n_stored)) else None,
        ))
    return entries


SLEEP_STAGES = (1, 2, 3, 4)

#: TSO divisor. A CONSTANT, not the recording length, so the feature measures
#: ELAPSED TIME rather than fractional position: at 30 s epochs, t = 1.0 is 1000
#: epochs = 8.33 h. Normalising per recording warps the axis -- a 6 h night and a
#: 10 h night would both span [0, 1], so the same t would mean two different
#: wall-clock times and the harmonics in ``models/tso.py`` would encode a
#: different period for every subject. 1000 is close to a typical night's epoch
#: count, so the numeric range stays near [0, 1] and the existing tso_jitter
#: (0.025 ~ 12.5 min) keeps its meaning.
TSO_DIVISOR = 1000.0


def wake_trim_bounds(labels, seq_len: int, pad_min: float, pad_max: float,
                     rec_id: str, seed: int, epoch_sec: float = 30.0):
    """Epoch range to keep: sleep, plus a random wake margin either side.

    Returns ``(lo, hi)`` inclusive. Recordings with no scored sleep, or too
    short to yield one window after trimming, are returned untrimmed.

    WHY. Sleep-onset latency measured from the first SCORED epoch differs ~18x
    across cohorts -- CFS median 124.5 min, MrOS 97.8, SOF 77.5 against MASS 6.8
    and YSYW 3.5 -- and the difference is protocol, not physiology: those three
    cohorts record ~2 h longer and their extra time is almost exactly their extra
    latency (CFS records 1.92 h more than MASS and its median SOL is 1.96 h
    longer). Left alone, a third of the training pool carries a two-hour block of
    trivially-classified wake at the start.

    The margin is RANDOM per recording rather than a fixed 30 min so the model
    cannot learn a constant offset between the recording edge and sleep onset --
    with a fixed margin, "epoch 60" would always be sleep onset.

    Costs nothing at batch time: called once per recording when the index is
    built, on an int8 array of ~1 kB.
    """
    labels = np.asarray(labels)
    n = len(labels)
    sleep = np.flatnonzero(np.isin(labels, SLEEP_STAGES))
    if sleep.size == 0:
        return 0, n - 1
    rng = np.random.default_rng(
        [int(seed), zlib.crc32(str(rec_id).encode())])
    per_min = 60.0 / epoch_sec
    pad_before = int(round(rng.uniform(pad_min, pad_max) * per_min))
    pad_after = int(round(rng.uniform(pad_min, pad_max) * per_min))
    lo = max(0, int(sleep[0]) - pad_before)
    hi = min(n - 1, int(sleep[-1]) + pad_after)
    if hi - lo + 1 < seq_len:            # too short to sample from once trimmed
        return 0, n - 1
    return lo, hi


class SleepSequenceDataset(Dataset):
    """Flat (rec, start_epoch) index over continuous memmapped recordings."""

    def __init__(self,
                 data_dir: str,
                 entries: List[RecordEntry],
                 seq_len: int,
                 stride: Optional[int] = None,
                 epoch_samples: int = EPOCH_SAMPLES,
                 fs: float = 100.0,
                 c_max: int = 2,
                 normalization: str = "fixed",
                 normalization_cfg: Optional[dict] = None,
                 train: bool = False,
                 augment: Optional[AugmentConfig] = None,
                 tso_jitter: float = 0.025,
                 tso_divisor: float = TSO_DIVISOR,
                 tso_norm: str = "elapsed",
                 min_label_purity: Optional[float] = None,
                 trim_wake: Optional[Tuple[float, float]] = None,
                 seed: int = 0):
        """
        seq_len:  T, epochs per sample (1 for Stage A)
        stride:   index stride in epochs (default: T for train, T for eval;
                  pass 1 for maximally overlapping windows)
        tso_jitter: TSO jitter half-width; U(-j, +j) applied ONLY when
                  ``train=True`` (task doc value 0.025; the old Keras code
                  used 0.03 — the doc wins, see README)
        min_label_purity: epochs with purity below this get label -1
                  (excluded from loss/metrics) — sensitivity analysis switch
        """
        self.data_dir = data_dir
        self.entries = entries
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride else self.seq_len
        self.epoch_samples = int(epoch_samples)
        self.fs = float(fs)
        self.c_max = int(c_max)
        self.normalization = normalization
        self.normalization_cfg = normalization_cfg or {}
        self.train = bool(train)
        self.augment = augment if (train and augment) else None
        self.tso_jitter = float(tso_jitter)
        self.tso_divisor = float(tso_divisor)
        #: "elapsed": t = epochs since the usable start / tso_divisor (the
        #: reported model). "fraction": t = epochs since the usable start /
        #: usable length, so every night spans 0..1 and the harmonics have a
        #: per-night period. This absorbs between-subject differences in sleep
        #: duration and cycle length at the price of making t depend on how
        #: long a recording was left running (see the E13 experiment).
        if tso_norm not in ("elapsed", "fraction"):
            raise ValueError(f"tso_norm must be 'elapsed' or 'fraction', got {tso_norm!r}")
        self.tso_norm = tso_norm
        self.min_label_purity = min_label_purity
        self._base_seed = int(seed)

        # (lo, hi) epoch bounds per entry; the full recording unless trimming
        self.bounds: List[Tuple[int, int]] = []
        self.index: List[Tuple[int, int]] = []
        for ri, e in enumerate(self.entries):
            lo, hi = 0, e.n_epochs - 1
            if trim_wake is not None:
                y = np.load(array_paths(self.data_dir, e.rec_id)["y"])
                lo, hi = wake_trim_bounds(y[:e.n_epochs], self.seq_len,
                                          trim_wake[0], trim_wake[1],
                                          e.rec_id, self._base_seed)
            self.bounds.append((lo, hi))
            last = hi - self.seq_len + 1
            for start in range(lo, last + 1, self.stride):
                self.index.append((ri, start))

        self._cache: Dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.index)

    # -- memmap access ------------------------------------------------------
    def _arrays(self, rec_id: str) -> dict:
        # np.load(mmap_mode='r') keeps OS page cache sharing across workers
        if rec_id not in self._cache:
            self._cache[rec_id] = open_arrays(self.data_dir, rec_id, mmap=True)
        return self._cache[rec_id]

    # -- main ---------------------------------------------------------------
    def __getitem__(self, i: int) -> dict:
        ri, start = self.index[i]
        entry = self.entries[ri]
        arrays = self._arrays(entry.rec_id)
        T, S, C = self.seq_len, self.epoch_samples, entry.n_channels

        s0 = start * S
        seg = np.array(arrays["x"][s0:s0 + T * S], dtype=np.float32)  # copy
        if entry.channel_index is not None:
            # select BEFORE normalising: entry.stats is already subset to these
            # columns, so the two must stay in step
            seg = seg[:, entry.channel_index]
        seg = apply_normalization(seg, self.normalization, entry.stats,
                                  self.normalization_cfg).astype(np.float32)

        y = arrays["y"][start:start + T].astype(np.int64)
        purity = arrays["purity"][start:start + T].astype(np.float32)
        if self.min_label_purity is not None:
            y = np.where(purity < self.min_label_purity, -1, y)

        # Elapsed time since the start of the USABLE span, in units of
        # TSO_DIVISOR epochs. Two things matter here:
        #  * the offset is relative to ``lo``, so the trimmed recording IS the
        #    recording as far as the model is concerned -- TSO 0 means "just
        #    before sleep onset" for every cohort rather than "whenever this lab
        #    started recording", which is what made TSO mean something different
        #    in CFS than in MASS.
        #  * the divisor is CONSTANT, so t is proportional to real elapsed time
        #    instead of being rescaled by each recording's length.
        lo, hi = self.bounds[ri]
        div = float(hi - lo + 1) if self.tso_norm == "fraction" else self.tso_divisor
        tso = (start - lo + np.arange(T, dtype=np.float32)) / div
        rng = None
        if self.train:
            rng = np.random.default_rng(
                (self._base_seed * 1_000_003 + i) % (2 ** 63))
            if self.tso_jitter > 0:
                # one draw per clip, matching the old generator
                tso = tso + rng.uniform(-self.tso_jitter, self.tso_jitter)

        mask = np.zeros(self.c_max, dtype=bool)
        mask[:C] = True

        # augmentations operate on (n_samples, C) before epoch folding
        if self.augment is not None:
            aug = self.augment
            dropped = False
            if C >= 2 and rng.uniform() < aug.channel_dropout_p:
                drop = int(rng.integers(0, C))
                seg[:, drop] = 0.0
                mask[drop] = False
                dropped = True

            # NEVER CORRUPT THE LAST CLEAN CHANNEL. The artifact only fires when
            # at least two channels survive, so a sample always keeps one intact.
            #
            # Previously the two draws were independent and the artifact
            # deliberately targeted a SURVIVING channel, so 0.4 x 0.3 = 12% of
            # C=2 samples had one channel zeroed and the only remaining one
            # corrupted — no clean EEG at all, yet still asked to predict the
            # true stage. The label is not recoverable from such a sample, so it
            # trains the network on noise.
            #
            # The same applied to natively single-channel recordings, where
            # dropout cannot fire but the artifact still hit the sole channel
            # 30% of the time. That is now 148 recordings (WSC 102, DREAMS 46
            # since the dead-channel QC reduced them to C=1), so the case stopped
            # being rare.
            n_clean = int(mask[:C].sum())
            if not dropped and n_clean >= 2 and rng.uniform() < aug.artifact_p:
                candidates = np.flatnonzero(mask[:C])
                ch = int(rng.choice(candidates))
                kind = str(rng.choice(np.asarray(aug.artifact_types)))
                _inject_artifact(seg, ch, self.fs, kind,
                                 aug.artifact_frac, aug.line_noise_hz, rng)

        # fold into epochs and pad channels: (T, C_max, S)
        # (dropped channels were zeroed in ``seg`` above; padding is zero)
        x = np.zeros((T, self.c_max, S), dtype=np.float32)
        x[:, :C, :] = seg.reshape(T, S, C).transpose(0, 2, 1)

        return {
            "x": torch.from_numpy(x),
            "channel_mask": torch.from_numpy(np.repeat(mask[None, :], T, 0)),
            "tso": torch.from_numpy(tso.astype(np.float32)),
            "y": torch.from_numpy(y),
            "purity": torch.from_numpy(purity),
            "rec_index": ri,
            "start_epoch": start,
        }

def sequence_length_for_duration(minutes: float,
                                 epoch_sec: float = 30.0) -> int:
    """Pick T from context DURATION (§3.8): 42.7 min ≈ 85 epochs at 30 s."""
    return max(1, int(round(minutes * 60.0 / epoch_sec)))


def make_loader(dataset: SleepSequenceDataset, batch_size: int,
                shuffle: bool, num_workers: int = 8,
                pin_memory: bool = True) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=shuffle,  # ragged final batch only matters for training
        persistent_workers=num_workers > 0)
