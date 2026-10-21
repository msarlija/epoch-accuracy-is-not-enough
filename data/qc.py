# -*- coding: utf-8 -*-
"""Spectral quality assessment.

Ported as-is from ``compute_record_quality`` in the Keras repo
(cnn_model_experiment.py): a recording is contaminated when its Welch power
spectrum has a local peak more than ``peak_threshold_db`` (15 dB) above the
local median computed in a ±``fdelta`` (2) Hz window, scanned over
``fmin``–``fmax`` (2–40 Hz in the old pipeline).

Differences from the old code:

* runs per CHANNEL (channels stay separate now); the record-level flag
  aggregates per config ('all' channels must pass [default] or 'any').
* the scan band is a config parameter: with the new 0.2–30 Hz filter the
  old 40 Hz upper edge sits in the filter's stop band, so configs default to
  fmax=30 (see configs/preprocess.yaml). Function defaults keep the ported
  values.
* QC is evaluated on SLEEP epochs only (artifacts during wake are acceptable)
  — same as the old pipeline.
* added :func:`comb_artifact_ratio`, which the ported criterion cannot express
  (see its docstring).
"""

from typing import List, Tuple

import numpy as np
from scipy.signal import welch


def spectral_quality_flag(sig: np.ndarray, fs: float,
                          peak_threshold_db: float = 15.0,
                          fmin: float = 2.0, fmax: float = 40.0,
                          fdelta: float = 2.0) -> int:
    """1 = clean, 0 = contaminated (strong narrowband peak). ``sig`` is 1-D."""
    sig = np.asarray(sig).ravel()
    if sig.size < int(fs) * 8:  # too short for a meaningful Welch estimate
        return 1
    freqs, psd = welch(sig, fs=fs, nperseg=int(fs * 4))
    psd_db = 10.0 * np.log10(psd + 1e-12)

    mask = (freqs >= fmin - fdelta) & (freqs <= fmax)
    freqs, psd_db = freqs[mask], psd_db[mask]

    for i, f in enumerate(freqs):
        if f <= fmin:
            continue
        local = (freqs >= f - fdelta) & (freqs <= f + fdelta)
        baseline = np.median(psd_db[local])
        if psd_db[i] - baseline > peak_threshold_db:
            return 0
    return 1


def comb_artifact_ratio(sig: np.ndarray, fs: float, fmin: float = 2.0,
                        fmax: float = 30.0,
                        spacings: Tuple[float, ...] = (1.0, 2.0, 3.0),
                        min_pairs: int = 5) -> float:
    """Strongest harmonic-comb contamination, as a power ratio (1.0 = none).

    WHY THE PORTED CRITERION CANNOT SEE THIS. ``spectral_quality_flag`` compares
    each bin to the median of a +/-``fdelta`` (2 Hz) window centred on it. A comb
    whose teeth are spaced <= 2*fdelta puts neighbouring teeth INSIDE that
    window, so the local baseline rises with the peak and the contrast cancels.
    STAGES BOGN carries a comb at odd integer Hz (a ~1 Hz pulse train: odd
    harmonics, amplitude ~1/n) that passes the local test at every tooth.

    METHOD. For each candidate tooth spacing ``s``, every bin at ``f`` is paired
    with the bin at ``f + s/2`` — halfway to the next tooth — and the score is
    the median ratio over pairs, in whichever direction is larger.

    Pairing at a FIXED OFFSET is what makes this work, and the two obvious
    alternatives do not:

    * against a band-wide median: EEG is ~1/f, so the statistic measures
      spectral slope, not comb structure (every cohort then scores 5-20).
    * against a local window around each tooth: the window falls on the tooth's
      own shoulders, compressing the contrast until BOGN and clean cohorts
      overlap.

    Paired bins are a fixed distance apart and interleaved across the whole
    band, so the 1/f slope cancels between numerator and denominator, and the
    comparison does not depend on how sharp the teeth are.

    MEASURED on this corpus (sleep epochs, first channel, 30 recordings per
    cohort): thirteen cohorts give median 1.28-1.59 with a 99th percentile of
    1.95, against BOGN median 5.28 (p90 8.44, 25 of 30 above 2.5). The default
    threshold of 2.5 therefore sits above every clean cohort's bulk and well
    below BOGN's centre; it is not finely tuned. One SOF recording scores 6.66
    and is flagged — plausibly a genuine artifact in that night rather than a
    false positive, but it is a single recording and has not been inspected.

    Line noise is not a concern here: 50/60 Hz and their harmonics lie outside
    the 0.2-30 Hz passband.
    """
    sig = np.asarray(sig).ravel()
    if sig.size < int(fs) * 30:
        return 1.0
    freqs, psd = welch(sig, fs=fs, nperseg=int(fs * 10))     # ~0.1 Hz bins
    if freqs.size < 40:
        return 1.0

    worst = 1.0
    for s_hz in spacings:
        half = s_hz / 2.0
        teeth = np.arange(np.ceil(fmin / s_hz) * s_hz, fmax - half, s_hz)
        if teeth.size < min_pairs:
            continue
        ratios = []
        for t in teeth:
            i = int(np.argmin(np.abs(freqs - t)))
            j = int(np.argmin(np.abs(freqs - (t + half))))
            if psd[j] > 0 and psd[i] > 0:
                ratios.append(psd[i] / psd[j])
        if len(ratios) >= min_pairs:
            med = float(np.median(ratios))
            worst = max(worst, med, 1.0 / med)
    return worst


def channel_amplitude_flags(signal: np.ndarray, iqr_min_uv: float = 1.0,
                            iqr_max_uv: float = 150.0,
                            flat_max_frac: float = 0.10) -> List[int]:
    """Per-channel 1/0 on amplitude sanity. ``signal`` is (n_samples, C) in uV.

    The ported spectral criterion is blind to all three of these: a channel
    reading a flat 0.0 uV has no narrowband peak, so it passes. Measured on this
    corpus, 58 QC-PASSING recordings carry a channel below 1 uV IQR (including
    all 46 DREAMS, whose FP2 is dead), 31 have >10% of samples flat, and 5 carry
    a channel above 100 uV IQR (up to 348.9 uV, ~17x normal).

    A failing channel should cost a CHANNEL, not a recording -- see
    ``entries_from_manifest``, which drops flagged columns and reduces C.
    """
    signal = np.atleast_2d(signal)
    flags = []
    for c in range(signal.shape[1]):
        chan = signal[:, c]
        q75, q25 = np.percentile(chan, [75, 25])
        iqr = float(q75 - q25)
        flat = float(np.mean(np.abs(np.diff(chan)) < 1e-6)) if len(chan) > 1 else 0.0
        ok = (iqr >= iqr_min_uv) and (iqr <= iqr_max_uv) and (flat <= flat_max_frac)
        flags.append(int(ok))
    return flags


def hypnogram_is_usable(labels: np.ndarray, min_sleep_frac: float = 0.10,
                        min_stages: int = 3) -> bool:
    """Reject degenerate scorings that make a recording untrainable/unscorable.

    Observed among QC-passing recordings: one MAYO night scored 100% Wake across
    all 508 epochs (kappa is undefined by construction), one 64% N1, and several
    with no N3 at all. Unlike a bad channel this cannot be repaired by dropping
    something, so it is a record-level rejection.
    """
    labels = np.asarray(labels)
    scored = labels[labels >= 0]
    if scored.size == 0:
        return False
    frac = np.array([(scored == s).mean() for s in range(5)])
    if frac[1:].sum() < min_sleep_frac:      # essentially no sleep
        return False
    if int((frac > 0.01).sum()) < min_stages:
        return False
    return True


def quality_kwargs(qc_cfg: dict) -> dict:
    """Filter a ``qc:`` config block to :func:`record_quality`'s parameters.

    Both call sites used to hand-pick keyword arguments, which meant a criterion
    added to the config silently did nothing until every call site was updated
    too. That is not a hypothetical: the amplitude/flatline keys were added and a
    dry run reported "0 recordings changed" because neither caller forwarded
    them, and ``prepare_data`` never forwarded ``comb_max_ratio`` at all.

    An UNKNOWN key raises rather than being ignored — a typo in a QC threshold
    that quietly disables the check is the same failure in a different costume.
    """
    import inspect
    allowed = set(inspect.signature(record_quality).parameters) - {
        "signal", "labels", "fs", "epoch_samples"}
    unknown = set(qc_cfg) - allowed
    if unknown:
        raise ValueError(
            f"unknown qc config key(s): {sorted(unknown)}; "
            f"record_quality accepts {sorted(allowed)}")
    return dict(qc_cfg)


def record_quality(signal: np.ndarray, labels: np.ndarray, fs: float,
                   epoch_samples: int, peak_threshold_db: float = 15.0,
                   fmin: float = 2.0, fmax: float = 40.0, fdelta: float = 2.0,
                   aggregate: str = "all",
                   comb_max_ratio: float = 2.5,
                   iqr_min_uv: float = 0.0, iqr_max_uv: float = np.inf,
                   flat_max_frac: float = 1.0,
                   min_sleep_frac: float = 0.0,
                   min_stages: int = 0) -> Tuple[int, List[int]]:
    """Record- and channel-level QC on the continuous filtered signal.

    ``signal``: (n_samples, C); ``labels``: (n_epochs,) 30 s labels aligned to
    the signal. Sleep epochs (N1/N2/N3/REM) are concatenated per channel and
    passed to the spectral criterion.

    Returns (record_flag, [channel_flags]).
    """
    signal = np.atleast_2d(signal)
    n_epochs = min(len(labels), signal.shape[0] // epoch_samples)
    sleep_epochs = np.flatnonzero(np.isin(labels[:n_epochs], [1, 2, 3, 4]))

    if sleep_epochs.size == 0:
        return 1, [1] * signal.shape[1]

    sample_idx = (sleep_epochs[:, None] * epoch_samples
                  + np.arange(epoch_samples)[None, :]).ravel()
    channel_flags = []
    for c in range(signal.shape[1]):
        chan = signal[sample_idx, c]
        ok = spectral_quality_flag(chan, fs,
                                   peak_threshold_db=peak_threshold_db,
                                   fmin=fmin, fmax=fmax, fdelta=fdelta)
        if ok and comb_max_ratio:
            # 2.5 sits above the clean cohorts' p99 (1.95) and below BOGN's
            # median (5.28) -- see comb_artifact_ratio for the measurements
            if comb_artifact_ratio(chan, fs, fmin=fmin,
                                   fmax=fmax) > comb_max_ratio:
                ok = 0
        channel_flags.append(ok)
    # amplitude / flatline, evaluated on the same sleep epochs
    amp = channel_amplitude_flags(signal[sample_idx], iqr_min_uv=iqr_min_uv,
                                  iqr_max_uv=iqr_max_uv,
                                  flat_max_frac=flat_max_frac)
    channel_flags = [int(a and b) for a, b in zip(channel_flags, amp)]

    if aggregate == "all":
        record_flag = int(all(channel_flags))
    elif aggregate == "any":
        # "any" is the policy that pairs with dropping bad channels: the record
        # survives as long as ONE usable channel remains
        record_flag = int(any(channel_flags))
    else:
        raise ValueError(f"Unknown QC aggregate mode '{aggregate}'")

    if record_flag and (min_sleep_frac > 0 or min_stages > 0):
        if not hypnogram_is_usable(labels[:n_epochs], min_sleep_frac,
                                   min_stages):
            record_flag = 0
    return record_flag, channel_flags
