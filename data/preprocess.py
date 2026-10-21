# -*- coding: utf-8 -*-
"""Signal preprocessing and label harmonization.

Pipeline (per recording, applied by scripts/prepare_data.py):

1. resample to 100 Hz with ``scipy.signal.resample_poly`` (clean rational
   factors; 250 Hz device native -> up 2 / down 5)
2. zero-phase IIR band-pass (default 0.2–30 Hz, ``sosfiltfilt``) applied to
   the FULL continuous recording before any epoching — no clip ever sees
   filter edge effects
3. harmonize labels to 30 s epochs by majority vote over temporal overlap,
   with per-epoch ``label_purity``
4. trim the signal to the labeled span; store the continuous filtered array

Normalization (§3.10) is NOT baked into the stored arrays: per-record
median/IQR are stored in the manifest and both modes ('fixed', 'iqr') are
applied at load time, so the normalization ablation does not require
re-running preprocessing. (The filter cutoff, by contrast, IS a preprocessing
parameter; changing it means re-running prepare_data.py — intended.)
"""

import json
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Tuple

import numpy as np
from scipy import signal as sps

TARGET_FS = 100.0
EPOCH_SEC = 30.0
EPOCH_SAMPLES = int(TARGET_FS * EPOCH_SEC)  # 3000


# ---------------------------------------------------------------------------
# Resampling / filtering
# ---------------------------------------------------------------------------

def resample_to_target(x: np.ndarray, fs_in: float,
                       fs_out: float = TARGET_FS,
                       max_denominator: int = 1000) -> np.ndarray:
    """Rational resampling of (n_samples, C) along axis 0.

    ``Fraction(fs_out / fs_in)`` gives exact factors for the common rates
    (250 -> 100 is up 2 / down 5); ``limit_denominator`` absorbs headers that
    store e.g. 256.000001 Hz.
    """
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x)
    frac = Fraction(fs_out / fs_in).limit_denominator(max_denominator)
    return sps.resample_poly(np.asarray(x), up=frac.numerator,
                             down=frac.denominator, axis=0)


def design_bandpass(fs: float = TARGET_FS, low_hz: float = 0.2,
                    high_hz: float = 30.0, order: int = 4) -> np.ndarray:
    """Butterworth band-pass SOS. ``high_hz`` must leave real transition
    margin below Nyquist (30 Hz at fs=100 does; do not push it to 50)."""
    nyq = fs / 2.0
    if not 0 < low_hz < high_hz < nyq:
        raise ValueError(
            f"Band ({low_hz}, {high_hz}) invalid for fs={fs} (Nyquist {nyq})")
    if high_hz > 0.9 * nyq:
        raise ValueError(
            f"Upper cutoff {high_hz} Hz too close to Nyquist {nyq} Hz")
    return sps.butter(order, [low_hz, high_hz], btype="bandpass",
                      fs=fs, output="sos")


def bandpass_filter(x: np.ndarray, fs: float = TARGET_FS, low_hz: float = 0.2,
                    high_hz: float = 30.0, order: int = 4) -> np.ndarray:
    """Zero-phase filter of the full continuous (n_samples, C) recording."""
    sos = design_bandpass(fs, low_hz, high_hz, order)
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    return sps.sosfiltfilt(sos, x, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Label harmonization: anything -> 30 s epochs with purity
# ---------------------------------------------------------------------------

@dataclass
class HarmonizedLabels:
    labels: np.ndarray        # (n_epochs,) int8, common scheme, -1 unknown
    purity: np.ndarray        # (n_epochs,) float32, coverage of winning label
    t0: float                 # start time (sec) of epoch 0 in source signal
    epoch_sec: float = EPOCH_SEC

    @property
    def n_epochs(self) -> int:
        return len(self.labels)


def harmonize_labels(hypnogram, epoch_sec: float = EPOCH_SEC,
                     end_time: Optional[float] = None) -> HarmonizedLabels:
    """Map stage events (onsets/durations/labels) onto a fixed 30 s grid.

    For each 30 s window, each label's temporal overlap with the window is
    accumulated and the label covering the greater share wins; exact ties are
    broken in favour of the EARLIER event (§3.1). ``label_purity`` is the
    fraction of the window covered by the winning label — e.g. 30 s windows
    over a 20 s R&K grid give purity 1.0 or 2/3.

    Unknown (-1) never wins over a real stage; a window wholly uncovered or
    covered only by unknown gets label -1 / purity of that coverage.
    """
    onsets = hypnogram.onsets
    durations = hypnogram.durations
    labels = hypnogram.labels
    if len(onsets) == 0:
        return HarmonizedLabels(np.zeros(0, np.int8), np.zeros(0, np.float32), 0.0)

    t0 = float(onsets[0])
    t_end = float(end_time) if end_time is not None else float(
        np.max(onsets + durations))
    n_epochs = int(np.floor((t_end - t0) / epoch_sec))

    out_labels = np.full(n_epochs, -1, dtype=np.int8)
    out_purity = np.zeros(n_epochs, dtype=np.float32)

    # coverage accumulators; label codes -1..4 -> row 0..5
    coverage = np.zeros((6, n_epochs), dtype=np.float64)
    first_onset = np.full((6, n_epochs), np.inf)  # for the earlier-event tie-break

    for onset, dur, lab in zip(onsets, durations, labels):
        row = int(lab) + 1
        if not 0 <= row <= 5:
            row = 0  # anything exotic counts as unknown
        e_first = max(0, int(np.floor((onset - t0) / epoch_sec)))
        e_last = min(n_epochs - 1, int(np.ceil((onset + dur - t0) / epoch_sec)))
        for e in range(e_first, e_last + 1):
            w_start = t0 + e * epoch_sec
            overlap = min(onset + dur, w_start + epoch_sec) - max(onset, w_start)
            if overlap <= 0:
                continue
            coverage[row, e] += overlap
            first_onset[row, e] = min(first_onset[row, e], onset)

    real = coverage[1:]  # rows for stages 0..4
    for e in range(n_epochs):
        col = real[:, e]
        best = col.max()
        if best <= 0:
            # only unknown coverage (or none)
            out_purity[e] = np.float32(coverage[0, e] / epoch_sec)
            continue
        winners = np.flatnonzero(np.isclose(col, best))
        if len(winners) == 1:
            win = winners[0]
        else:  # tie -> earlier epoch's label
            win = winners[np.argmin(first_onset[1:, e][winners])]
        out_labels[e] = np.int8(win)
        out_purity[e] = np.float32(col[win] / epoch_sec)

    return HarmonizedLabels(out_labels, out_purity, t0, epoch_sec)


def trim_to_labeled(signal: np.ndarray, fs: float,
                    harmonized: HarmonizedLabels
                    ) -> Tuple[np.ndarray, HarmonizedLabels]:
    """Cut the FILTERED continuous signal to [first, last] labeled epoch.

    Filtering already happened on the full recording, so trimming here does
    not introduce edge effects. Epochs whose samples run past the signal end
    are dropped; leading/trailing unknown epochs are kept only between the
    first and last known label (matches the old pipeline's behaviour of
    keeping in-order unknowns inside the scored span).
    """
    labels, purity = harmonized.labels, harmonized.purity
    known = np.flatnonzero(labels != -1)
    if known.size == 0:
        return (np.zeros((0, signal.shape[1]), signal.dtype),
                HarmonizedLabels(np.zeros(0, np.int8),
                                 np.zeros(0, np.float32), harmonized.t0))
    first, last = int(known[0]), int(known[-1])

    epoch_samples = int(round(harmonized.epoch_sec * fs))
    start_sample = int(round((harmonized.t0 + first * harmonized.epoch_sec) * fs))
    # clamp to what the signal actually contains
    max_full = (signal.shape[0] - start_sample) // epoch_samples
    n_epochs = min(last - first + 1, max(0, max_full))
    labels = labels[first:first + n_epochs]
    purity = purity[first:first + n_epochs]
    seg = signal[start_sample:start_sample + n_epochs * epoch_samples]
    new = HarmonizedLabels(labels, purity,
                           harmonized.t0 + first * harmonized.epoch_sec,
                           harmonized.epoch_sec)
    return seg, new


# ---------------------------------------------------------------------------
# Normalization (applied at LOAD time; both modes selectable by config)
# ---------------------------------------------------------------------------

@dataclass
class RecordStats:
    """Per-record, per-channel robust statistics stored in the manifest."""
    median: np.ndarray  # (C,)
    iqr: np.ndarray     # (C,)

    @classmethod
    def from_signal(cls, x: np.ndarray) -> "RecordStats":
        med = np.median(x, axis=0)
        q75, q25 = np.percentile(x, [75, 25], axis=0)
        iqr = np.maximum(q75 - q25, 1e-6)
        return cls(median=med.astype(np.float64), iqr=iqr.astype(np.float64))


def normalize_fixed(x: np.ndarray, scale: float = 50.0,
                    clip: Optional[float] = 5.0) -> np.ndarray:
    """Divide by a constant, PRESERVING absolute amplitude information.

    NOTE the clip trap from the old repo: research ``rescale_array`` clipped
    at ±5 (i.e. ±250 µV), the production copy did not. Clipping is explicit
    and configurable here; ``clip=None`` disables it.
    """
    x = x / float(scale)
    if clip is not None:
        x = np.clip(x, -clip, clip)
    return x


def normalize_iqr(x: np.ndarray, median: np.ndarray, iqr: np.ndarray,
                  clip: Optional[float] = 20.0,
                  scale_match: float = 1.0) -> np.ndarray:
    """Per-record, per-channel zero-median unit-IQR; clip at ``clip``×IQR.

    ``scale_match`` divides the result by a constant so that this mode lands on
    the same amplitude scale as ``normalize_fixed``, which is what makes a
    fixed-vs-IQR comparison a comparison of ADAPTATION rather than of input
    magnitude. See :func:`reference_iqr_scale` for how the constant is derived;
    1.0 leaves the literal U-Sleep unit-IQR convention untouched.

    It is applied AFTER the clip, deliberately. ``clip`` therefore keeps meaning
    exactly "``clip``xIQR" whatever ``scale_match`` is, so the physical
    truncation threshold — and hence the clipped fraction — is identical to the
    unmatched mode. Dividing first would silently move the threshold to
    ``clip/scale_match``, making the clip a second, hidden difference between
    the arms.

    ``clip=20`` is the value §3.10 specifies, and it is U-Sleep's published
    recipe verbatim (Perslev et al., npj Digital Medicine 2021): scale each
    channel per subject to median 0 / IQR 1, then clip anything deviating from
    the median by more than 20×IQR. Do not "tune" it without a reason stronger
    than a preprocessing measurement — matching a published preprocessing
    pipeline is worth more than a marginal gain, because it is what makes
    results comparable to that literature.

    It is a targeted outlier guard, not a routine compressor, and the
    distinction matters. Measured over 357 recordings / 171 M samples: 0.36% of
    values pooled, 0.018% on the median recording, and 96 of 357 recordings
    untouched entirely. What it does act on is degenerate data — the flatlined
    BOAS headband recordings (sub-13 42%, sub-1 25%) whose ~0.04 µV IQR makes any
    IQR-relative threshold enormous. Lowering it to 5 makes it fire on every
    recording (1.87% pooled, 1.27% median) and start eating slow-wave amplitude,
    which AASM defines N3 by.

    For reference, the ``fixed`` default (±250 µV) truncates 0.257% pooled, so
    the two modes are already within 1.4x of each other in aggregate — the
    clipping is not a meaningful confound in a fixed-vs-iqr ablation.
    """
    x = (x - median) / iqr
    if clip is not None:
        x = np.clip(x, -clip, clip)
    if scale_match != 1.0:
        x = x / float(scale_match)
    return x


def reference_iqr_scale(manifest, scale: float = 50.0) -> float:
    """Divisor that puts ``iqr`` mode on the same amplitude scale as ``fixed``.

    The two modes differ in magnitude because they divide by different things:
    ``fixed`` divides every channel by the constant ``scale`` (50 uV), ``iqr`` by
    each channel's own IQR. Matching the MEDIAN PER-CHANNEL IQR AFTER
    normalisation makes them comparable, and the match is exact rather than
    approximate:

        fixed, per channel:  IQR_uV / scale      -> median  median(IQR_uV)/scale
        iqr,   per channel:  1 / k               -> 1/k     (the same for all)

    so setting ``k = scale / median(IQR_uV)`` equates the two by construction,
    with no fitting and no residual.

    What deliberately does NOT get equated is the SPREAD: after ``fixed`` the
    per-channel IQR still varies between recordings, while after ``iqr`` it is
    1/k everywhere. That difference IS the manipulation under test, which is
    exactly why the central value has to be matched — otherwise a magnitude
    offset rides along with it.

    Computed from the pooled per-channel IQRs of the passed manifest (the
    training pool), never hardcoded, and recorded with the run.
    """
    values = []
    for v in manifest["iqr"]:
        if isinstance(v, str):
            v = json.loads(v)
        values.extend(float(t) for t in np.atleast_1d(v))
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        raise ValueError("no usable per-channel IQR values in the manifest")
    return float(scale) / float(np.median(arr))


def apply_normalization(x: np.ndarray, mode: str, stats: RecordStats,
                        cfg: Optional[dict] = None) -> np.ndarray:
    cfg = cfg or {}
    if mode == "fixed":
        return normalize_fixed(x, scale=cfg.get("scale", 50.0),
                               clip=cfg.get("clip", 5.0))
    if mode == "iqr":
        match = cfg.get("scale_match", 1.0)
        if not isinstance(match, (int, float)):
            # 'auto' must be resolved to a number by the caller (run_experiment
            # does this from the pool manifest) and written into the resolved
            # config. Silently treating it as 1.0 would run the arm unmatched.
            raise ValueError(
                f"normalization.scale_match is {match!r}; it must be resolved "
                "to a number before use (see reference_iqr_scale)")
        return normalize_iqr(x, stats.median, stats.iqr,
                             clip=cfg.get("clip", 20.0),
                             scale_match=float(match))
    raise ValueError(f"Unknown normalization mode '{mode}'")
