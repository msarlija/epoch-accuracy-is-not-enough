# -*- coding: utf-8 -*-
"""Common schema all cohort readers normalize to.

Every reader returns a :class:`RawRecording`:

* ``signal``: (n_samples, C) float32/float64 in microvolts, at the source
  sampling rate, channel polarity already applied, channels kept SEPARATE.
* ``hypnogram``: a :class:`Hypnogram` of stage events with onsets/durations in
  seconds relative to signal start, labels already in the common scheme.

Common label scheme (everywhere downstream of the readers):

    -1 = unknown / unscored / movement
     0 = W, 1 = N1, 2 = N2, 3 = N3 (incl. R&K S4), 4 = REM

The old repo used REM=5 in some readers and REM=4 in others, remapped at
training time (``y[y==5]=4``); here readers emit the final scheme directly.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

UNKNOWN, WAKE, N1, N2, N3, REM = -1, 0, 1, 2, 3, 4
STAGE_NAMES = {WAKE: "W", N1: "N1", N2: "N2", N3: "N3", REM: "REM"}
VALID_STAGES = (WAKE, N1, N2, N3, REM)


@dataclass
class Hypnogram:
    """Stage events with explicit onsets and durations (seconds).

    Keeping explicit durations (instead of assuming a fixed grid) lets the
    30 s harmonizer handle 20 s R&K cohorts, DREAMS' 5 s hypnograms and
    variable-duration NSRR XML events with one code path.
    """

    onsets: np.ndarray      # (n_events,) float seconds, relative to signal start
    durations: np.ndarray   # (n_events,) float seconds
    labels: np.ndarray      # (n_events,) int, common scheme

    def __post_init__(self):
        self.onsets = np.asarray(self.onsets, dtype=float)
        self.durations = np.asarray(self.durations, dtype=float)
        self.labels = np.asarray(self.labels, dtype=int)
        if not (len(self.onsets) == len(self.durations) == len(self.labels)):
            raise ValueError("onsets, durations, labels must be equal length")

    @property
    def end_time(self) -> float:
        if len(self.onsets) == 0:
            return 0.0
        return float(self.onsets[-1] + self.durations[-1])

    @classmethod
    def from_grid(cls, labels, epoch_sec: float, t0: float = 0.0) -> "Hypnogram":
        """Build from labels on a fixed grid starting at ``t0``."""
        labels = np.asarray(labels, dtype=int)
        onsets = t0 + np.arange(len(labels), dtype=float) * epoch_sec
        durations = np.full(len(labels), float(epoch_sec))
        return cls(onsets=onsets, durations=durations, labels=labels)


@dataclass
class RawRecording:
    rec_id: str
    cohort: str
    subject_id: str
    montage_family: str
    channel_names: List[str]          # resolved source-header names, in order
    signal: np.ndarray                # (n_samples, C), microvolts, native fs
    fs: float
    hypnogram: Hypnogram
    source_path: str
    age: Optional[float] = None
    sex: Optional[str] = None
    #: e.g. DOD-H per-scorer hypnograms; keys are scorer ids.
    extra_hypnograms: Dict[str, Hypnogram] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        self.signal = np.atleast_2d(np.asarray(self.signal))
        if self.signal.shape[0] < self.signal.shape[1]:
            # accept (C, n_samples) input from readers, store (n_samples, C)
            self.signal = self.signal.T
        if self.signal.shape[1] != len(self.channel_names):
            raise ValueError(
                f"{self.rec_id}: signal has {self.signal.shape[1]} channels "
                f"but {len(self.channel_names)} channel names")

    @property
    def n_channels(self) -> int:
        return self.signal.shape[1]


#: Recordings dropped by readers, per cohort. A reader that skips silently is
#: indistinguishable from a cohort that is simply small, so the count has to
#: leave the generator somehow; ``scripts/prepare_data.py`` reads this for its
#: end-of-run summary. Reset it per cohort there, not here.
SKIPPED: "Counter[str]" = Counter()


def log_skip(cohort: str, rec_id) -> None:
    """Log the in-flight exception, count it, and note the recording is skipped.

    Readers call this from an ``except`` around the per-recording work so that
    one unusable source file costs one recording instead of the whole cohort.
    Reasons seen in practice: a recording listed in an import table but missing
    from a partial download, and an EDF carrying none of the cohort's candidate
    channel sets (50 of the 53 MASS SS1 files have no Fp channel at all).
    """
    SKIPPED[cohort] += 1
    log.exception("%s %s: unusable source, skipping this recording",
                  cohort, rec_id)


def wall_clock_offset(clock_sec: float, origin_sec: float) -> float:
    """Seconds from ``origin_sec`` to ``clock_sec``, resolving the date wrap.

    Annotation files that timestamp with wall clock and no date are ambiguous by
    multiples of a day, and overnight recordings straddle midnight. A plain
    ``(clock - origin) % 86400`` fixes the after-midnight case but breaks the
    other one: an annotation 30 minutes BEFORE the recording start becomes
    +23.5 h, which exceeds the recording length and made ``harmonize_labels``
    compute a negative epoch count (observed on 6 STAGES MAYO recordings).

    Mapping into (-12 h, +12 h] instead handles both, since no PSG session is
    ambiguous at that scale.
    """
    delta = (clock_sec - origin_sec) % 86400.0
    if delta > 43200.0:
        delta -= 86400.0
    return delta


def format_record_id(value, id_format: Optional[str] = None) -> str:
    """Render an import-table id the way it appears in filenames.

    ``DataFrame.iterrows`` collapses each row to a single dtype, so an integer
    id in an otherwise all-numeric table arrives as a float and a plain
    ``str()`` yields ``'1600003.0'`` — a filename that does not exist. Recover
    the integer first; ``id_format`` (e.g. ``'05d'``) then works as intended.
    """
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    return format(value, id_format) if id_format else str(value)


@dataclass(frozen=True)
class ResolvedChannels:
    """Which header labels satisfy a :class:`ChannelSpec`, and how to combine them.

    ``reference_names`` is empty for the usual case of one header channel per
    output channel. When populated it is parallel to ``names``, and output
    channel *i* is ``names[i] - reference_names[i]`` (see
    :attr:`ChannelSpec.reference`).
    """

    indices: List[int]
    names: List[str]
    polarity: List[int]
    reference_names: List[str] = field(default_factory=list)
    reference_indices: List[int] = field(default_factory=list)

    @property
    def is_derivation(self) -> bool:
        return bool(self.reference_names)

    def all_header_names(self) -> List[str]:
        """Every label that must be read, deduplicated, order preserved."""
        seen, out = set(), []
        for n in list(self.names) + list(self.reference_names):
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def describe(self) -> List[str]:
        """Human-readable derivation labels, for the manifest."""
        if not self.is_derivation:
            return list(self.names)
        return [f"{a}-{b}" for a, b in zip(self.names, self.reference_names)]


def _find_label(available: List[str], want: str, taken: set, substring: bool):
    for i, have in enumerate(available):
        match = (want.lower() in have.lower()) if substring \
            else (want.lower() == have.lower())
        if match and i not in taken:
            return i
    return None


def resolve_channels(available: List[str], candidate_specs,
                     substring: bool = False) -> Optional[ResolvedChannels]:
    """First candidate spec fully present in ``available``, or None.

    ``substring=True`` matches each wanted name as a substring of the header
    label (used for MASS, where labels are e.g. 'EEG Fp1-CLE'). Note that
    substring matching is unsafe for headers that repeat a label with a suffix —
    STAGES carries both 'EEG_Fp1-A2' and 'EEG_Fp1-A22' — so cohorts like that
    must resolve by exact name.

    A spec with a ``reference`` also requires its reference labels to be
    present; the returned object then describes a derivation rather than a
    direct channel read.
    """
    for spec in candidate_specs:
        taken: set = set()
        idxs, names = [], []
        for want in spec.names:
            hit = _find_label(available, want, taken, substring)
            if hit is None:
                break
            taken.add(hit)
            idxs.append(hit)
            names.append(available[hit])
        else:
            ref = getattr(spec, "reference", None)
            if not ref:
                return ResolvedChannels(idxs, names, list(spec.polarity))
            # A reference channel may legitimately be shared between output
            # channels, so references are matched without consuming from
            # ``taken``; only the primary channels must be distinct.
            ref_idxs, ref_names = [], []
            for want in ref:
                hit = _find_label(available, want, set(), substring)
                if hit is None:
                    break
                ref_idxs.append(hit)
                ref_names.append(available[hit])
            else:
                return ResolvedChannels(idxs, names, list(spec.polarity),
                                        ref_names, ref_idxs)
    return None
