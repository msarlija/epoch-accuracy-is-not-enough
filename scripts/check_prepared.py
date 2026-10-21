# -*- coding: utf-8 -*-
"""Verify a prepared corpus: run this after every prepare_data.py run.

``prepare_data.py`` catches per-recording failures to keep long runs alive, so
it exits 0 whether it wrote a good corpus, a subtly wrong one, or nothing at
all. These checks are what makes that distinction. They are deliberately
derived from the ARRAYS rather than the config, so a preprocessing parameter
that never took effect is still caught.

What each group is for:

structural
    Shapes, dtypes and the ``n_samples == n_epochs * 3000`` invariant that
    ``trim_to_labeled`` is supposed to guarantee, plus manifest/array agreement.

sample rate and band-pass, from the data
    Nothing in the pipeline records the sample rate actually achieved:
    ``resample_to_target`` derives its ratio from whatever ``rec.fs`` the reader
    reported, and a wrong ``fs`` produces a plausible-looking corpus at the
    wrong rate. Two spectral ratios pin it down. Power above 40 Hz must be far
    below the 1-25 Hz band (the 30 Hz low-pass), and power below 0.1 Hz must be
    far below the 0.5-2 Hz band (the 0.2 Hz high-pass). The second ratio is the
    load-bearing one: EEG is strongly 1/f, so *absolute* low-frequency power
    stays high even after filtering, and only its power RELATIVE to just above
    the corner separates the two cases — measured at 9.3 unfiltered versus
    0.002 filtered on a MrOS recording, so the 0.05 threshold has three orders
    of magnitude of headroom either side.

amplitude
    Per-channel IQR against a physiological window. This is the only check that
    catches a V/mV/µV mix-up, and it needs to exist because such an error is
    invisible everywhere else: band-pass, harmonization, QC and the fixed
    ÷50 normalization all behave identically on a signal scaled a million-fold.
    A real instance was found this way (DREAMS declares its EDF unit as 'UV',
    which mne does not recognise, so it silently returned unscaled microvolts).
    Reported as a warning rather than a failure because a genuinely dead
    electrode lands here too.

labels
    Class balance and purity. A corpus can be structurally perfect and still
    have the hypnogram misparsed, which shows up as an implausible histogram.

Usage:
    python scripts/check_prepared.py --config configs/preprocess.yaml
    python scripts/check_prepared.py --config configs/preprocess.yaml --per-record
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config  # noqa: E402
from data import manifest as mf  # noqa: E402
from data.readers.base import STAGE_NAMES  # noqa: E402

EPOCH_SAMPLES = 3000            # 30 s at 100 Hz
TARGET_FS = 100.0

# Spectral ratio thresholds; see the module docstring for the measured margins.
MIN_HIGH_STOPBAND_RATIO = 300.0   # (1-25 Hz) / (>40 Hz)
MAX_LOW_PASSBAND_RATIO = 0.05     # (<0.1 Hz) / (0.5-2 Hz)

# Prefrontal EEG band-passed to 0.2-30 Hz. Wide on purpose: this is a
# unit-error detector, not a quality metric.
IQR_MIN_UV, IQR_MAX_UV = 0.5, 500.0

# A channel this much weaker than its partner is not carrying comparable signal.
# Relative rather than absolute, because no absolute floor separates "weak lead"
# from "quiet recording": DREAMS' second lead sits at 0.5-0.9 uV IQR against
# 13-32 uV on the first, which straddles any absolute threshold but is a factor
# of 20-30 apart. Worth surfacing because a recording that reports
# n_channels=2 while carrying one usable lead quietly contaminates both the
# 1ch-vs-2ch comparison and the channel-attention analysis.
WEAK_CHANNEL_RATIO = 0.05

# IQR this small relative to the SAME channel's std means the middle of the
# distribution has collapsed onto a constant while large excursions remain —
# a flatlined channel with occasional artifact, not EEG. Needs to be separate
# from WEAK_CHANNEL_RATIO because it is the only thing that catches a recording
# where BOTH leads are flat (there is no healthy partner to compare against).
#
# Do NOT tighten this towards normal-looking values: healthy EEG runs
# iqr/std 0.2-1.0 because artifact excursions inflate std (that is exactly why
# the manifest stores IQR rather than std), so a threshold anywhere near 0.2
# fires on most of the corpus. The flat BOAS recordings sit at 0.0016, roughly
# 30x below the lowest healthy value observed.
COLLAPSED_IQR_RATIO = 0.02


def spectral_ratios(x, fs=TARGET_FS, n=600_000):
    """(in-band / >40 Hz, <0.1 Hz / 0.5-2 Hz) for the first channel."""
    seg = np.asarray(x[:n, 0], dtype=np.float64)
    # nperseg=4000 -> 0.025 Hz bins, which is what it takes to resolve anything
    # below the 0.2 Hz corner at all; a coarser window leaves that band empty.
    freq, psd = welch(seg, fs=fs, nperseg=min(4000, len(seg)))
    in_band = psd[(freq > 1) & (freq < 25)].mean()
    stop_hi = psd[freq > 40].mean()
    low = psd[(freq > 0) & (freq < 0.1)].mean()
    near = psd[(freq >= 0.5) & (freq <= 2.0)].mean()
    return (in_band / stop_hi if stop_hi > 0 else np.inf,
            low / near if near > 0 else np.inf)


def check_record(data_dir, row):
    """(failures, warnings, stats) for one manifest row."""
    fails, warns = [], []
    paths = mf.array_paths(data_dir, row.rec_id)
    x = np.load(paths["x"], mmap_mode="r")
    y = np.load(paths["y"])
    purity = np.load(paths["purity"])

    def hard(cond, msg):
        if not cond:
            fails.append(msg)

    def soft(cond, msg):
        if not cond:
            warns.append(msg)

    hard(x.dtype == np.float32, f"x dtype is {x.dtype}, expected float32")
    hard(y.dtype == np.int8, f"y dtype is {y.dtype}, expected int8")
    hard(purity.dtype == np.float32, f"purity dtype is {purity.dtype}")
    hard(x.ndim == 2 and x.shape[1] in (1, 2), f"x shape {x.shape}")
    hard(x.shape[0] == len(y) * EPOCH_SAMPLES,
         f"signal/label misalignment: {x.shape[0]} samples vs "
         f"{len(y)} epochs x {EPOCH_SAMPLES}")
    hard(len(purity) == len(y), f"purity {len(purity)} != labels {len(y)}")
    hard(int(row.n_samples) == x.shape[0] and int(row.n_epochs) == len(y),
         "manifest n_samples/n_epochs disagree with the arrays")
    hard(set(np.unique(y)) <= {-1, 0, 1, 2, 3, 4},
         f"labels outside the common scheme: {sorted(set(np.unique(y)))}")
    hard(purity.min() >= 0.0 and purity.max() <= 1.0 + 1e-6,
         f"purity outside [0,1]: [{purity.min()}, {purity.max()}]")
    hard(np.isfinite(np.asarray(x[::5000])).all(), "non-finite samples present")
    hard(int(row.n_channels) == x.shape[1],
         f"manifest n_channels={row.n_channels} but array has {x.shape[1]}")

    hi_ratio, lo_ratio = spectral_ratios(x)
    hard(hi_ratio > MIN_HIGH_STOPBAND_RATIO,
         f"weak 30 Hz low-pass: in-band/>40Hz = {hi_ratio:.0f} "
         f"(want > {MIN_HIGH_STOPBAND_RATIO:.0f}); wrong fs or filter not applied")
    hard(lo_ratio < MAX_LOW_PASSBAND_RATIO,
         f"weak 0.2 Hz high-pass: <0.1Hz/0.5-2Hz = {lo_ratio:.4f} "
         f"(want < {MAX_LOW_PASSBAND_RATIO})")

    iqr = np.asarray(json.loads(row.iqr) if isinstance(row.iqr, str) else row.iqr,
                     dtype=float)
    soft(iqr.max() < IQR_MAX_UV,
         f"IQR {np.round(iqr, 3)} uV above {IQR_MAX_UV} — probable unit error")

    # Subsample rather than read every sample; std only needs to be right to
    # within a few percent for these ratios.
    std = np.asarray(x[::7], dtype=np.float64).std(axis=0)
    collapsed = np.flatnonzero(iqr / np.maximum(std, 1e-12) < COLLAPSED_IQR_RATIO)
    soft(not len(collapsed),
         f"channel(s) {collapsed.tolist()} flatlined: IQR {np.round(iqr, 3)} uV "
         f"against std {np.round(std, 1)} uV — the distribution has collapsed "
         f"onto a constant, with excursions on top. Not usable as EEG")

    if len(iqr) > 1 and iqr.max() > 0 and not len(collapsed):
        ratio = iqr.min() / iqr.max()
        soft(ratio >= WEAK_CHANNEL_RATIO,
             f"channel {int(np.argmin(iqr))} is {1 / max(ratio, 1e-9):.0f}x "
             f"weaker than its partner: IQR {np.round(iqr, 3)} uV. Effectively "
             f"a 1-channel recording filed as n_channels={len(iqr)}")

    hours = x.shape[0] / (TARGET_FS * 3600.0)
    soft(3.0 < hours < 16.0, f"implausible duration {hours:.1f} h")

    counts = {STAGE_NAMES[s]: int((y == s).sum()) for s in (0, 1, 2, 3, 4)}
    scored = sum(counts.values())
    soft(scored > 0, "no scored epochs at all")
    if scored:
        top = max(counts, key=counts.get)
        soft(counts[top] / scored < 0.9,
             f"{counts[top] / scored:.0%} of scored epochs are {top} — "
             f"hypnogram probably misparsed")

    return fails, warns, dict(
        rec_id=row.rec_id, cohort=row.cohort, hours=hours,
        channels=x.shape[1], qc_pass=bool(row.qc_pass), iqr=iqr,
        purity=float(row.mean_label_purity), hi_ratio=hi_ratio,
        lo_ratio=lo_ratio, unknown_frac=float((y == -1).mean()), **counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="the config whose data_dir should be checked")
    parser.add_argument("--per-record", action="store_true",
                        help="print a line per recording (default: on for "
                             "small corpora)")
    args = parser.parse_args()

    data_dir = load_config(args.config)["data_dir"]
    man = mf.load_manifest(data_dir)
    if not len(man):
        print(f"NO MANIFEST ROWS in {data_dir} — prepare_data.py wrote nothing")
        return 1
    print(f"{len(man)} recordings in {data_dir}")

    hashes = man.preproc_hash.unique()
    if len(hashes) != 1:
        print(f"\nMIXED PREPROCESSING: {len(hashes)} distinct preproc_hash "
              f"values {list(hashes)}.\nArrays written under different filter "
              f"settings are present; prepare_data.py resumes on rec_id alone "
              f"and cannot detect this. Use a separate data_dir per config.")
        return 1
    print(f"preproc_hash {hashes[0]} (uniform)\n")

    rows, all_fails, all_warns = [], [], []
    for _, row in man.iterrows():
        try:
            fails, warns, stats = check_record(data_dir, row)
        except Exception as err:  # noqa: BLE001
            all_fails.append(f"{row.rec_id}: could not be checked: {err!r}")
            continue
        rows.append(stats)
        all_fails += [f"{row.rec_id}: {f}" for f in fails]
        all_warns += [f"{row.rec_id}: {w}" for w in warns]

    df = pd.DataFrame(rows)
    if args.per_record or len(df) <= 60:
        cols = ("rec_id", "hours", "channels", "qc_pass", "purity",
                "hi_ratio", "lo_ratio", "unknown_frac")
        print(df[list(cols)].to_string(
            index=False, float_format=lambda v: f"{v:.4g}"))
        print()

    print("--- per cohort ---")
    agg = df.groupby("cohort").agg(
        n=("rec_id", "size"), hours=("hours", "mean"),
        qc_pass=("qc_pass", "mean"), purity=("purity", "mean"),
        unknown=("unknown_frac", "mean"),
        **{s: (s, "sum") for s in ("W", "N1", "N2", "N3", "REM")})
    stage_cols = ["W", "N1", "N2", "N3", "REM"]
    total = agg[stage_cols].sum(axis=1)
    for s in stage_cols:
        agg[s] = (agg[s] / total).round(3)
    print(agg.to_string(float_format=lambda v: f"{v:.3g}"))

    print("\n--- corpus stage balance ---")
    tot = df[stage_cols].sum()
    print({s: f"{tot[s] / tot.sum():.1%}" for s in stage_cols})

    if all_warns:
        print(f"\n{len(all_warns)} WARNING(S) — inspect, do not necessarily fix:")
        for w in all_warns:
            print("  -", w)
    if all_fails:
        print(f"\n{len(all_fails)} FAILURE(S):")
        for f in all_fails:
            print("  -", f)
        return 1
    print("\nOK: structural, sample-rate, band-pass and label checks all passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
