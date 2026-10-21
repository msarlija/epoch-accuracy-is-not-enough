# -*- coding: utf-8 -*-
"""Small EDF helpers shared by the readers."""

import logging
import os
from typing import Dict, List, Optional

import numpy as np

from .base import resolve_channels

log = logging.getLogger(__name__)

UV = 1e-6  # mne returns volts; we store microvolts

#: What mne's EDF reader recognises, copied deliberately: the match is EXACT and
#: CASE-SENSITIVE (mne/io/edf/edf.py), and any other string silently falls
#: through to a factor of 1 — i.e. the file's numbers are taken to be volts
#: already. See ``edf_unit_correction`` for why that matters.
_MNE_KNOWN_UNITS = {"uV": 1e-6, "μV": 1e-6, "µV": 1e-6,
                    "\x83\xcaV": 1e-6, "mV": 1e-3}

#: What we are willing to interpret ourselves, case-insensitively.
_UNIT_TO_VOLTS = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "μv": 1e-6,
                  "µv": 1e-6, "nv": 1e-9}


def read_edf(path: str, include: Optional[List[str]] = None,
             preload: Optional[bool] = None):
    """Open an EDF with mne (imported here so that non-EDF cohorts can be
    processed on machines without mne installed).

    ``include`` restricts the read to the named channels, which matters for
    more than speed. A PSG EDF stores EEG at one rate and other modalities at
    others, and mne resolves that by resampling everything to the file's
    highest rate — so a 128 Hz EEG lead is silently delivered at 512 Hz because
    some other channel runs that fast, and with ``preload=False`` mne warns that
    the mixed-rate path leaves edge artifacts at block boundaries. Asking only
    for the two EEG leads removes the rate mismatch entirely: the read returns
    the leads' own sampling rate, and there is nothing to resample.
    """
    import mne
    if preload is None:
        # Cheap once `include` has cut the read to two channels (~80 MB for a
        # 10 h night), and it forecloses the mixed-rate edge-artifact path.
        preload = include is not None
    kwargs = {"preload": preload, "verbose": "ERROR"}
    if include is not None:
        kwargs["include"] = list(include)
    return mne.io.read_raw_edf(path, **kwargs)


def edf_physical_dimensions(path: str) -> Dict[str, str]:
    """``{channel label: physical dimension}`` straight from the EDF header.

    Parsed here rather than taken from mne because mne does not expose the
    per-channel scaling it chose, and we need to know whether it recognised the
    unit at all. The layout is fixed by the EDF spec: a 256-byte fixed header
    (channel count in bytes 252:256), then per-signal blocks of 16-byte labels
    and 80-byte transducer strings, then the 8-byte physical dimensions.
    """
    with open(path, "rb") as fh:
        header = fh.read(256)
        if len(header) < 256:
            raise ValueError(f"{path}: truncated EDF header")
        n_signals = int(header[252:256].decode("ascii", "replace").strip() or 0)
        if n_signals <= 0:
            raise ValueError(f"{path}: EDF header declares {n_signals} signals")
        labels = [fh.read(16).decode("latin-1").strip() for _ in range(n_signals)]
        fh.seek(256 + n_signals * (16 + 80))
        dims = [fh.read(8).decode("latin-1").strip() for _ in range(n_signals)]
    return dict(zip(labels, dims))


def edf_unit_correction(path: str, names: List[str],
                        assume_unit: Optional[str] = None) -> Optional[np.ndarray]:
    """Factor to fix mne's unit handling for ``names``, or None if unavailable.

    mne matches the EDF physical dimension case-sensitively against a short
    list, and *silently* treats anything unmatched as volts. DREAMS declares
    ``'UV'`` (uppercase), so mne returns the file's raw microvolt numbers while
    every caller believes they are volts — a 10**6 error that no downstream
    stage can detect: the band-pass, the label harmonizer and the spectral QC
    all behave identically on a signal scaled a million-fold, and normalization
    happens at load time from statistics computed on the same wrong data.

    Returns ``our_factor / mne_factor`` per channel, so multiplying mne's output
    by it yields true volts regardless of which units the file declares.
    """
    try:
        dims = edf_physical_dimensions(path)
    except Exception:  # noqa: BLE001 — a header we cannot parse just means no fix
        log.warning("%s: could not read EDF physical dimensions; trusting mne's "
                    "unit handling", os.path.basename(path))
        return None

    factors = np.ones(len(names), dtype=float)
    for i, name in enumerate(names):
        if name not in dims:
            log.warning("%s: channel %r not in the EDF header labels; trusting "
                        "mne's unit handling", os.path.basename(path), name)
            continue
        declared = dims[name]
        ours = _UNIT_TO_VOLTS.get(declared.lower())
        if ours is None and assume_unit and not declared:
            # Some exports leave the physical dimension blank (STAGES does).
            # mne then treats the numbers as volts, which is a 10**6 error for a
            # file recorded in µV. A cohort that knows its own units may say so;
            # this only ever applies when the header itself is silent, so it can
            # never override a unit the file actually declares.
            ours = _UNIT_TO_VOLTS.get(assume_unit.lower())
            log.info("%s: channel %r declares no unit; assuming %r per the "
                     "cohort spec", os.path.basename(path), name, assume_unit)
        if ours is None:
            log.warning("%s: channel %r declares unit %r, which is not a "
                        "voltage unit we know; leaving the values as mne "
                        "returned them", os.path.basename(path), name, declared)
            continue
        mne_factor = _MNE_KNOWN_UNITS.get(declared, 1.0)
        if ours != mne_factor:
            factors[i] = ours / mne_factor
            log.info("%s: channel %r declares %r but mne applied %g; "
                     "correcting by %g", os.path.basename(path), name,
                     declared, mne_factor, factors[i])
    return factors


def extract_channels(raw, spec, substring: bool = False,
                     assume_unit: Optional[str] = None):
    """Extract the first matching candidate channel set from an mne Raw.

    Returns (signal (n_samples, C) float32 microvolts with polarity applied,
    resolved_names, fs). Raises LookupError when no candidate matches, with
    the full header list in the message so the miss is diagnosable.

    Prefer :func:`load_edf_channels`, which reads only the channels it needs;
    this remains for callers that already hold an mne ``Raw``.
    """
    resolved = resolve_channels(raw.info["ch_names"],
                               spec.candidate_channel_specs(),
                               substring=substring)
    if resolved is None:
        raise LookupError(
            f"No candidate channel set of cohort {spec.name} found. "
            f"Available channels: {raw.info['ch_names']}")
    path = raw.filenames[0] if getattr(raw, "filenames", None) else None
    return _build_signal(raw, resolved, path, assume_unit), resolved.describe(), \
        float(raw.info["sfreq"])


def _build_signal(raw, resolved, path, assume_unit=None):
    """Read the resolved labels and combine them into (n_samples, C) float32 µV.

    For a plain spec that is one header channel per output channel. For a spec
    with a ``reference`` each output channel is a difference of two header
    channels, computed AFTER unit correction so both sides are in the same units.
    """
    wanted = resolved.all_header_names()
    data = raw.get_data(picks=wanted)              # (n_wanted, n_samples)
    data = _to_microvolts_raw(data, wanted, path, assume_unit)
    by_name = {n: data[i] for i, n in enumerate(wanted)}

    cols = []
    for i, name in enumerate(resolved.names):
        sig = by_name[name]
        if resolved.is_derivation:
            sig = sig - by_name[resolved.reference_names[i]]
        cols.append(sig * float(resolved.polarity[i]))
    return np.stack(cols, axis=1).astype(np.float32)


def _to_microvolts_raw(data, names, path, assume_unit=None):
    """(C, n) volts-per-mne -> (C, n) µV, unit-correcting per channel."""
    correction = edf_unit_correction(path, names, assume_unit) if path else None
    if correction is not None:
        data = data * correction[:, None]
    return data / UV


def load_edf_channels(path: str, spec, substring: bool = False,
                      assume_unit: Optional[str] = None):
    """Open ``path`` reading ONLY the channels ``spec`` asks for.

    Resolving against the raw header first means mne never sees the other
    modalities, so it reports the EEG leads' own sampling rate instead of
    resampling them up to the file's maximum, and the mixed-rate edge-artifact
    path is never taken (see :func:`read_edf`).

    Falls back to reading every channel when the header cannot be parsed, or
    when a wanted label is duplicated in it — mne uniquifies duplicate names,
    so selecting by label would be ambiguous.
    """
    try:
        dims = edf_physical_dimensions(path)
        labels = list(dims)
    except Exception:  # noqa: BLE001 — fall back to letting mne do the parsing
        log.warning("%s: unparseable EDF header, reading all channels",
                    os.path.basename(path))
        return extract_channels(read_edf(path, preload=False), spec,
                                substring=substring, assume_unit=assume_unit)

    resolved = resolve_channels(labels, spec.candidate_channel_specs(),
                               substring=substring)
    if resolved is None:
        raise LookupError(
            f"No candidate channel set of cohort {spec.name} found. "
            f"Available channels: {labels}")
    wanted = resolved.all_header_names()

    if any(labels.count(n) > 1 for n in wanted):
        log.warning("%s: duplicate channel labels in the header, reading all "
                    "channels", os.path.basename(path))
        return extract_channels(read_edf(path, preload=False), spec,
                                substring=substring, assume_unit=assume_unit)

    raw = read_edf(path, include=wanted)
    got = list(raw.info["ch_names"])
    if set(got) != set(wanted):
        raise LookupError(
            f"{os.path.basename(path)}: asked mne for {wanted}, received {got}")
    return _build_signal(raw, resolved, path, assume_unit), resolved.describe(), \
        float(raw.info["sfreq"])
