# -*- coding: utf-8 -*-
"""Per-cohort EDF channel selection and montage metadata.

This module is the single place that encodes which derivations exist in each
dataset and what they are called in the EDF (or h5 / wfdb) headers. It was
ported from the inline conditionals in the Keras repo
(``brain-age/sleep-stage-classification/preprocessing_for_cnn.py``) — expensive
knowledge to rediscover, so keep it explicit and declarative.

Key differences from the old pipeline:

* Channels stay SEPARATE (no Fp1/Fp2 averaging into a virtual channel).
  Each entry lists the derivations in a fixed order; the preprocessing keeps
  them as columns of the output array.
* ``polarity`` is applied per channel so that every stored derivation has the
  orientation of a (pre)frontal-minus-mastoid signal. The old code negated
  mastoid-referenced channels (e.g. CFS ``-(M1+M2)/2``); here each mastoid
  lead is negated individually and kept separate.
* ``montage_family`` is recorded per cohort (needed for stratified reporting
  and it interacts with the normalization / filter-cutoff ablations).

Montage families
----------------
``prefrontal``      Fp1/Fp2-referenced derivations (MASS, DREAMS, ANPHY, DOD-H)
``prefrontal_hb``   headband dry-electrode prefrontal (BOAS)
``mastoid_proxy``   negated mastoid leads used as a prefrontal proxy
                    (CFS, HomePAP, MrOS, SOF)
``frontal``         F3/F4-referenced derivations (YSYW, WSC)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class ChannelSpec:
    """One candidate channel set for a cohort.

    ``names`` are the labels as they appear in the source file header.
    ``polarity`` multiplies each channel after extraction (+1 / -1).
    A cohort may list several candidate specs (``fallbacks`` in
    :class:`CohortSpec`); the first spec whose names are all present is used.
    """

    names: Sequence[str]
    polarity: Sequence[int]
    #: Optional per-channel subtrahend, for cohorts stored as UNIPOLAR channels
    #: that must be re-referenced. ``names=("Fp2","Fpz")`` with
    #: ``reference=("M1","M2")`` yields the derivations Fp2-M1 and Fpz-M2. This
    #: is not expressible with ``polarity``, which only flips one channel's sign.
    #: STAGES MAYO needs it.
    #:
    #: NOTE a true derivation, not the averaging the old pipeline applied
    #: (``((Fp2-M1)+(Fpz-M2))/2``): channels stay separate per §3.4.
    reference: Optional[Sequence[str]] = None

    def __post_init__(self):
        if len(self.names) != len(self.polarity):
            raise ValueError("names and polarity must have equal length")
        if self.reference is not None and len(self.reference) != len(self.names):
            raise ValueError("reference must be parallel to names")


@dataclass(frozen=True)
class CohortSpec:
    name: str
    montage_family: str
    channels: ChannelSpec
    #: Alternative channel name sets tried in order when the primary is absent.
    fallbacks: List[ChannelSpec] = field(default_factory=list)
    #: Native scoring epoch length in seconds (30 = AASM, 20 = R&K 20 s,
    #: 5 = DREAMS hypnogram sampling). None = variable, taken from the file.
    native_epoch_sec: Optional[float] = None
    #: Reader implementation registered in data.readers.
    reader: str = ""
    notes: str = ""

    def candidate_channel_specs(self) -> List[ChannelSpec]:
        return [self.channels] + list(self.fallbacks)


COHORTS = {
    # ------------------------------------------------------------------
    # Existing corpus
    # ------------------------------------------------------------------
    "MASS": CohortSpec(
        name="MASS",
        montage_family="prefrontal",
        # MASS labels vary by subset ("EEG Fp1-CLE", "EEG Fp1-LER", ...);
        # the reader matches any channel containing these substrings.
        channels=ChannelSpec(names=("Fp1", "Fp2"), polarity=(1, 1)),
        native_epoch_sec=None,  # SS2/SS4/SS5 are 20 s (R&K), SS1/SS3 30 s
        reader="mass",
        notes="Substring match on 'Fp'; annotation EDFs ('Base' for SS2, "
              "'Annotations' otherwise) carry true per-event durations.",
    ),
    "CFS": CohortSpec(
        name="CFS",
        montage_family="mastoid_proxy",
        channels=ChannelSpec(names=("M1", "M2"), polarity=(-1, -1)),
        native_epoch_sec=30.0,  # NSRR XML stage events (durations explicit)
        reader="nsrr_xml",
        notes="CFS visit5. Scored on a 20 s R&K grid upstream; the NSRR XML "
              "export carries explicit durations, handled by the harmonizer.",
    ),
    "HPAP": CohortSpec(
        name="HPAP",
        montage_family="prefrontal",
        # The NSRR export stores reference minus electrode (Supplementary
        # Section 1 of the paper). 45 files record FPz against a central
        # reference: stored M1 - stored FPz = FPz - M1 (N3 skew -0.43, standard).
        # The other 98 have no FPz channel because FPz IS the reference: stored
        # M1/M2 = FPz - M1 / FPz - M2. Their stored sign varies from file to file
        # (per-night N3 skew of M1 and of the reference-free F3-M2 agree in sign
        # on 80% of nights), so those nights get a per-night sign in the
        # manifest (scripts/set_orientation.py).
        channels=ChannelSpec(names=("M1", "M2"), polarity=(1, 1), reference=("FPz", "FPz")),
        fallbacks=[ChannelSpec(names=("M1", "M2"), polarity=(1, 1))],
        native_epoch_sec=30.0,
        reader="nsrr_xml",
        notes="HomePAP lab full-PSG arm. Stage 4 mapped into N3. FPz-M1/FPz-M2: "
              "M1/M2 minus FPz where FPz is recorded, else the stored M1/M2 with "
              "a per-night sign (scripts/set_orientation.py).",
    ),
    "MrOS": CohortSpec(
        name="MrOS",
        montage_family="mastoid_proxy",
        channels=ChannelSpec(names=("A1", "A2"), polarity=(-1, -1)),
        native_epoch_sec=30.0,
        reader="nsrr_xml",
        notes="Visit 1. Stage 4 mapped into N3.",
    ),
    "SOF": CohortSpec(
        name="SOF",
        montage_family="mastoid_proxy",
        channels=ChannelSpec(names=("A1", "A2"), polarity=(-1, -1)),
        native_epoch_sec=30.0,
        reader="nsrr_xml",
        notes="Visit 8. Stage 4 mapped into N3.",
    ),
    "BOAS": CohortSpec(
        name="BOAS",
        montage_family="prefrontal_hb",
        channels=ChannelSpec(names=("HB_1", "HB_2"), polarity=(1, 1)),
        native_epoch_sec=30.0,
        reader="boas",
        notes="BitBrain open access sleep. Headband EEG scored against the "
              "simultaneous PSG (events tsv, column 'stage_hum').",
    ),
    "DREAMS": CohortSpec(
        name="DREAMS",
        montage_family="prefrontal",
        channels=ChannelSpec(names=("FP1-A2", "FP2-A1"), polarity=(1, 1)),
        native_epoch_sec=5.0,  # AASM hypnogram txt sampled every 5 s
        reader="dreams",
    ),
    "ANPHY": CohortSpec(
        name="ANPHY",
        montage_family="prefrontal",
        # The released EEG is referenced to the amplifier's central system
        # reference. reextract_anphy.py re-references to the contralateral
        # mastoid side and writes <subject>_Fp-TP_250Hz.edf with exactly these
        # two labels, so there is no fallback: an unconverted folder must fail.
        channels=ChannelSpec(names=("Fp1-TP10", "Fp2-TP9"), polarity=(1, 1)),
        native_epoch_sec=30.0,
        reader="anphy",
        notes="Annotations txt: '<stage> <start_sec> ...' lines; 'L' = unknown. "
              "Signals: Fp1-TP10 and Fp2-TP9 at 250 Hz from reextract_anphy.py.",
    ),
    # ------------------------------------------------------------------
    # Additions
    # ------------------------------------------------------------------
    "YSYW": CohortSpec(
        name="YSYW",
        montage_family="frontal",
        channels=ChannelSpec(names=("F3-M2", "F4-M1"), polarity=(1, 1)),
        native_epoch_sec=30.0,
        reader="ysyw",
        notes="PhysioNet 2018 ('you snooze you win'). wfdb records; stages "
              "embedded in the 'arousal' annotation stream.",
    ),
    "WSC": CohortSpec(
        name="WSC",
        montage_family="frontal",
        # WSC provides ONE frontal lead, not a pair — so this cohort is C=1 by
        # construction, and it is the only cohort that exercises the C=1 path on
        # real data. Candidates are naming variants of that single lead, tried in
        # order; the reader logs every unmatched header so a new variant is
        # visible rather than silently skipped.
        channels=ChannelSpec(names=("F3_M2",), polarity=(1,)),
        fallbacks=[
            ChannelSpec(names=("F3-M2",), polarity=(1,)),
            ChannelSpec(names=("F3_AVG",), polarity=(1,)),
            ChannelSpec(names=("F3",), polarity=(1,)),
            ChannelSpec(names=("F4_M1",), polarity=(1,)),
            ChannelSpec(names=("F4-M1",), polarity=(1,)),
            ChannelSpec(names=("F4_AVG",), polarity=(1,)),
            ChannelSpec(names=("F4",), polarity=(1,)),
            ChannelSpec(names=("Fz_A2",), polarity=(1,)),
        ],
        native_epoch_sec=30.0,
        reader="wsc",
        notes="Streams from S3 (no local copy). Single frontal lead, so C=1. "
              "The Keras repo left this reader a TODO and never read a WSC "
              "header, so the two-channel candidates it guessed were wrong.",
    ),
    "STAGES": CohortSpec(
        name="STAGES",
        montage_family="prefrontal",   # per-site; the reader overrides per record
        # Three site montages, tried in order. Each file carries only its own
        # site's channels, so the fallback chain selects correctly without the
        # reader needing to branch on the site code.
        #   MSMI: bipolar prefrontal, already referenced
        channels=ChannelSpec(names=("EEG_Fp1-A2", "EEG_Fp2-A1"),
                             polarity=(1, 1)),
        fallbacks=[
            #   BOGN: bipolar frontal
            ChannelSpec(names=("F1M2", "F2M1"), polarity=(1, 1)),
            #   Cross-site variant: MSMI-style labels carrying FRONTAL
            #   derivations. Seen at BOGN (13 recordings) AND MSMI (4), so it is
            #   not site-specific -- which is why montage_family is resolved from
            #   the matched channel names per recording (montage_for) rather than
            #   from the site code.
            ChannelSpec(names=("EEG_F3-A2", "EEG_F4-A1"), polarity=(1, 1)),
            #   MAYO: UNIPOLAR, needs re-referencing -> Fp2-M1 and Fpz-M2
            ChannelSpec(names=("Fp2", "Fpz"), polarity=(1, 1),
                        reference=("M1", "M2")),
            #   MAYO variant carrying Fp1 where the above expects Fpz, giving
            #   the conventional prefrontal pair Fp2-M1 / Fp1-M2.
            ChannelSpec(names=("Fp2", "Fp1"), polarity=(1, 1),
                        reference=("M1", "M2")),
        ],
        native_epoch_sec=None,   # event log with explicit per-block durations
        reader="stages",
        notes="Streams from S3. Headers duplicate every label with a '2' suffix "
              "(EEG_Fp1-A2 AND EEG_Fp1-A22), so this cohort must resolve by "
              "EXACT name -- substring matching would take the wrong one. The "
              "EDFs also declare no physical dimension, so the reader passes "
              "assume_unit='uV'; without it values come out 10**6 too large.",
    ),
    "MASS-SS1": CohortSpec(
        name="MASS-SS1",
        montage_family="frontal",
        # SS1 is the one MASS subset with essentially no Fp: 47 of 53 files carry
        # only EEG F3-CLE / F4-CLE and 3 more only F3-LER / F4-LER, which is why
        # the original import table (filtered on Fp availability) contained just
        # 3 SS1 rows and the other 50 were never offered to the reader.
        #
        # Taken as FRONTAL over all 53 recordings, including the 3 that also have
        # Fp -- those carry F3/F4 as well, so the cohort stays homogeneous. CLE
        # and LER are MASS's linked-ear references, making these genuine frontal
        # derivations directly comparable to YSYW's F3-M2 / F4-M1.
        #
        # Held out as EXTERNAL: it is the only frontal, C=2, lab-quality external
        # cohort available (WSC is frontal but single-lead), and SS1 subjects are
        # distinct from SS2/SS3/SS5, so there is no leakage against the MASS
        # recordings in the training pool.
        channels=ChannelSpec(names=("F3", "F4"), polarity=(1, 1)),
        native_epoch_sec=None,
        reader="mass",
        notes="MASS SS1 read as frontal F3/F4. Substring match, as for the "
              "prefrontal MASS cohort; 'F3' cannot collide with F7/F8/Fz.",
    ),
    "DODO": CohortSpec(
        name="DODO",
        # DOD-O is NOT DOD-H with a different population -- it ships a different
        # montage. All 56 records carry C3_M2, C4_M1, F3_F4, F3_M2, F3_O1,
        # F4_O2, O1_M2, O2_M1 and NO prefrontal channels at all, where DOD-H's
        # defining pair is FP1_M2 / FP2_M1. F3_M2 is the only clean
        # frontal-to-contralateral-mastoid derivation present: F4_O2 and F3_O1
        # are frontal-occipital, F3_F4 is bipolar frontal, and there is no
        # F4_M1. So this cohort is C=1 by construction, exactly like WSC, and
        # 'frontal' rather than 'mastoid_proxy' for the same reason WSC is --
        # F3_M2 is a real frontal-mastoid derivation, not an inverted mastoid
        # channel standing in for one.
        montage_family="frontal",
        channels=ChannelSpec(names=("F3_M2",), polarity=(1,)),
        fallbacks=[ChannelSpec(names=("F3-M2",), polarity=(1,))],
        native_epoch_sec=30.0,
        reader="dodh",          # same h5 layout and stage coding as DOD-H
        notes="Dreem open dataset (obstructive sleep apnea). Clinical PSG "
              "montage, single usable frontal lead F3-M2, so C=1. External "
              "evaluation only.",
    ),
    "DODH": CohortSpec(
        name="DODH",
        montage_family="prefrontal",
        channels=ChannelSpec(names=("Fp1_M2", "Fp2_M1"), polarity=(1, 1)),
        fallbacks=[ChannelSpec(names=("Fp1-M2", "Fp2-M1"), polarity=(1, 1))],
        native_epoch_sec=30.0,
        reader="dodh",
        notes="Dreem open dataset (healthy). Read the h5 files directly — "
              "secondary sources disagree on the channel list. Five "
              "independent expert scorings; config selects scorer/consensus.",
    ),
}


def get_cohort(name: str) -> CohortSpec:
    try:
        return COHORTS[name]
    except KeyError:
        raise KeyError(
            f"Unknown cohort '{name}'. Known: {sorted(COHORTS)}") from None
