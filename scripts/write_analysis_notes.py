# -*- coding: utf-8 -*-
"""Write corpus statistics and standing findings to a notes file for the manuscript.

The quantitative half is REGENERATED from the manifest on every run, so the
numbers in the file are never stale relative to the corpus. The qualitative half
is a curated list of things established by investigation that would otherwise
have to be rediscovered -- format quirks, data-quality findings, and decisions
whose rationale is not visible in the code.

    python scripts/write_analysis_notes.py --config configs/preprocess.yaml
"""

import argparse
import collections
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config     # noqa: E402
from data import manifest as mf     # noqa: E402

STAGES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

FINDINGS = """
================================================================================
STANDING FINDINGS  (curated; edit freely, the section above is auto-generated)
================================================================================

STAGES-MAYO ANNOTATION ARTIFACT  (reportable; two independent confirmations)
- A subset of STAGES-MAYO recordings carry a systematic ~2 HOUR discrepancy
  between the EDF header start time and the annotation clock, which misaligns the
  hypnogram against the signal. It is an ANNOTATION/EXPORT artifact of the source
  dataset, not a preprocessing bug on our side: the stage LABELS parse correctly,
  and the stored label sequence matches the raw CSV exactly (agreement 1.000 at
  offset 0 on every recording checked). Only the anchor -- which sample the first
  label is pinned to -- is wrong.
- MAYO is INTERNALLY INCONSISTENT about which timestamp is authoritative. Two
  candidates exist in the files and roughly half the cohort follows each:
      A. the CSV 'Beginning of Recording' / 'Start Recording' event
      B. the EDF header start time PLUS EXACTLY 2 HOURS
  Anchoring everything on A (our original behaviour) leaves ~57/109 misaligned by
  a median 22 epochs (11 min); anchoring everything on B fixes 37 of those but
  breaks 41 that A had right. On the recordings B describes, the residual is a
  median of -0.5 epochs, i.e. correct to ~15 s.
- Nothing in the files distinguishes the two populations. Tested and rejected:
  marker-vs-header skew, biocalibration gap, marker count, marker spread,
  annotation span vs EDF duration, and within-recording drift (0.0 epochs between
  halves -- the offset is constant, not accumulating).
- RESOLVED PER RECORDING with a MODEL-FREE criterion (scripts/diagnose_mayo_alignment.py):
  delta power should rise with scored sleep depth while the zero-crossing rate
  falls; both features are amplitude-invariant. The anchor agreeing better with
  both wins. This deliberately avoids the trained model -- selecting an anchor by
  model agreement would fit the labels to the model being evaluated.
      marker  33 | header+2h  36 | ambiguous 20 | unresolved 19 | no-N3 1
  Correction applied to the header+2h group: median +20 epochs (+10.2 min),
  range +5 to +151; alignment score improves by a median of +0.186.
- INDEPENDENTLY CONFIRMED by the trained model's own best-shift search: the
  model-free verdict agrees on 60/69 resolved recordings (87%) -- 34/36 of the
  header+2h group and 26/33 of the marker group. 17 of the 20 'ambiguous' cases
  have a model shift of <=2 epochs, i.e. the two anchors barely differ there.
- CONSEQUENCE FOR REPORTING: MAYO's uncorrected external kappa (0.49 under the
  best model) is roughly half data defect. Correcting the alignment recovers it
  toward 0.62. Reporting the uncorrected number as a generalisation failure would
  be wrong. Recommend reporting corrected MAYO with the rejection rate stated,
  and raising the offset with the STAGES maintainers.

QC / DATA QUALITY
- Spectral QC rejects a similar share of the consumer-headband cohort and of lab
  PSG: BOAS 4/128 (3.1%) vs HPAP 5/143 (3.5%). Relevant to the criticism that QC
  selectively culls the only wearable dataset -- on the full corpus it does not.
- BOAS sub-1 and sub-13 are FLATLINED: IQR 0.04 uV against std 25 uV, i.e. the
  distribution has collapsed onto a constant with artifact on top. Both pass the
  ported narrowband-peak QC, which structurally cannot see a dead channel.
- DREAMS is effectively SINGLE-CHANNEL. FP2-A1 is ~30x weaker than FP1-A2 in all
  46 recordings (median IQR ratio 0.034 for subjects, 0.035 for patients), and a
  band-power probe on FP2 alone scores kappa 0.079 -- chance -- against 0.573 for
  FP1. Every control cohort's second channel matches its first. Filed as C=2.
- CFS_800312 (94% wake), CFS_800937 (96%), HPAP_1600033 (90%) are genuinely poor
  nights, not parse failures: their NSRR XML contains only Wake/N1/N2 events.
- 13 of 105 WSC recordings are coded unscored (code 7) for EVERY epoch in the
  release, with no .allscore.txt alternative. Not recoverable.
- BOGN runs 2-5x hotter than every other group and clips 4.10% of samples under
  the fixed default, against 0.000-0.045% elsewhere (IQR [40.1, 109.6] uV vs
  MSMI [22.1, 22.2]). Its second lead is real signal -- N3/W delta modulation
  tracks the first lead -- so this is amplitude, not artifact. A cross-montage
  amplitude disparity concentrated in one site of one cohort.

SUBJECT IDENTITY (affects any split)
- BOAS: 128 recordings from 100 people. participants.csv `pid` is the person,
  `participant_id` ('sub-N') is the recording; 20 people appear 2-3 times.
- WSC: repeat visits share a subject id; 94 recordings from 88 subjects.
- These are the only two cohorts where recordings-per-subject exceeds 1.

PREPROCESSING
- The band-pass is applied with sosfiltfilt (zero-phase, so event times are not
  shifted), which SQUARES the magnitude response. The realised -3 dB band is
  therefore 0.223-28.32 Hz, not the nominal 0.2-30, and gain at the nominal
  cutoffs is 0.5 rather than 0.707. Skirts are steeper than 4th-order Butterworth
  implies: -25 dB at 35 Hz, -56 dB at 40 Hz.
- Line noise does not alias into the passband. CFS carries 60 Hz mains below its
  64 Hz Nyquist; notching it out before resampling changes the 0.2-30 Hz PSD by a
  median of 0.08-0.2% across CFS, MrOS and MASS, so resample_poly's anti-alias
  filter removes it before decimation.
- Normalization is applied at LOAD time, not baked into the arrays, so the
  fixed-vs-iqr comparison needs no re-preprocessing.
- iqr clip is 20xIQR: the value the spec states and U-Sleep's published recipe
  (Perslev et al., npj Digital Medicine 2021). It is a targeted outlier guard,
  not a compressor -- 0.36% of samples pooled, 96 of 357 recordings untouched,
  and what it does act on is the degenerate BOAS pair.
- fixed clip is 10 (+/-500 uV), just above the 99.9th percentile of |x/50|, so it
  bounds pathology without touching physiology. Departs from the ported +/-5.

SOURCE FORMAT QUIRKS (each corrupted data silently rather than raising)
- DREAMS EDFs declare the unit as 'UV' (uppercase); mne matches unit strings
  case-sensitively and treats anything unmatched as volts, giving a 10^6 error.
- STAGES EDFs declare NO unit at all -- same 10^6 error, handled by an explicit
  per-cohort assumption that only applies when the header is silent.
- STAGES headers duplicate every label with a '2' suffix (EEG_Fp1-A2 AND
  EEG_Fp1-A22), so that cohort must resolve channels by exact name.
- STAGES annotations use two conventions: MSMI writes variable-length blocks with
  explicit durations, BOGN writes one row per epoch with Duration=0.000.
- WSC annotations split by visit: .stg.txt through visit 2, .allscore.txt from
  visit 3 (stage TRANSITIONS, durations from the gaps).
- WSC montage splits by visit too: visit 1 has NO frontal lead (C3_M2/O1_M2
  only), visit 2 is mixed, visits 3-5 carry F3_M2. Restricted to visits 3-5,
  which biases toward later visits i.e. slightly older subjects.
- STAGES BOGN's F1/F2 labels are PREFRONTAL positions despite the naming; strict
  10-20 reading would say frontal, so it cannot be inferred from the label.
- MASS: 50 of 53 SS1 recordings carry no Fp channel at all; they are correctly
  absent from the import table.
- Wall-clock annotation timestamps are ambiguous by a day. Offsets are mapped to
  (-12 h, +12 h]; a plain modulo turned an annotation 30 min BEFORE the recording
  start into +23.5 h and produced a negative epoch count.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg["data_dir"]
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(data_dir)), "analysis_notes.txt")
    m = mf.load_manifest(data_dir)
    m["hours"] = m.n_epochs * 30 / 3600

    L = []
    w = L.append
    w("=" * 80)
    w("PREPROCESSED CORPUS -- ANALYSIS NOTES")
    w(f"auto-generated {datetime.now():%Y-%m-%d %H:%M} by "
      f"scripts/write_analysis_notes.py")
    w(f"corpus: {data_dir}")
    w("=" * 80)

    f = cfg.get("filter", {})
    w("\nPREPROCESSING PARAMETERS")
    w(f"  target_fs        {cfg.get('target_fs')} Hz (scipy resample_poly)")
    w(f"  epoch_sec        {cfg.get('epoch_sec')} s -> "
      f"{int(cfg.get('target_fs',100)*cfg.get('epoch_sec',30))} samples/epoch")
    w(f"  band-pass        {f.get('low_hz')}-{f.get('high_hz')} Hz, Butterworth "
      f"order {f.get('order')}, sosfiltfilt (zero-phase)")
    w(f"                   realised -3 dB band 0.223-28.32 Hz (filtfilt squares "
      f"the response)")
    q = cfg.get("qc", {})
    w(f"  QC               Welch narrowband peak > {q.get('peak_threshold_db')} dB "
      f"over the +/-{q.get('fdelta')} Hz local median,")
    w(f"                   scanned {q.get('fmin')}-{q.get('fmax')} Hz, "
      f"aggregate='{q.get('aggregate')}' (all channels must pass)")
    w(f"  storage          float32 uV continuous array + int8 labels + float32 "
      f"purity, per recording")
    w(f"  normalization    applied at LOAD time, not stored")

    w("\nCORPUS TOTALS")
    w(f"  recordings       {len(m):,}")
    w(f"  subjects         {m.subject_id.nunique():,}")
    w(f"  cohorts          {m.cohort.nunique()}")
    w(f"  hours            {m.hours.sum():,.0f}")
    w(f"  30 s epochs      {int(m.n_epochs.sum()):,}")
    w(f"  QC pass          {int(m.qc_pass.sum()):,} / {len(m):,} "
      f"({m.qc_pass.mean():.2%})")
    if (m.qc_pass == 0).any():
        w(f"  QC failures      {dict(m[m.qc_pass==0].cohort.value_counts())}")

    w("\nPER COHORT")
    hdr = (f"  {'cohort':8s} {'rec':>5s} {'subj':>5s} {'hours':>8s} "
           f"{'epochs':>9s} {'C':>3s} {'QC':>7s} {'age':>5s}  montage")
    w(hdr); w("  " + "-" * (len(hdr) - 2))
    for c, g in m.groupby("cohort"):
        fam = "/".join(sorted(g.montage_family.unique()))
        age = f"{g.age.mean():.1f}" if g.age.notna().any() else "n/a"
        w(f"  {c:8s} {len(g):5d} {g.subject_id.nunique():5d} {g.hours.sum():8.0f} "
          f"{int(g.n_epochs.sum()):9,d} {int(g.n_channels.median()):3d} "
          f"{g.qc_pass.mean():6.1%} {age:>5s}  {fam}")

    w("\nRECORDINGS PER SUBJECT (>1 means a split must group by subject)")
    rv = m.groupby(["cohort", "subject_id"]).size()
    rv = rv[rv > 1]
    if len(rv):
        for c, n in rv.groupby(level=0).size().items():
            w(f"  {c}: {n} subjects with >1 recording "
              f"({len(m[m.cohort==c])} recordings, "
              f"{m[m.cohort==c].subject_id.nunique()} subjects)")
    else:
        w("  none")

    w("\nMONTAGE FAMILIES (the stratification axis)")
    for fam, g in m.groupby("montage_family"):
        w(f"  {fam:16s} {len(g):5d} rec  {sorted(g.cohort.unique())}")

    w("\nSTAGE BALANCE")
    tot = collections.Counter(); n_all = 0
    for _, r in m.iterrows():
        y = np.load(mf.array_paths(data_dir, r.rec_id)["y"])
        tot.update(y.tolist()); n_all += len(y)
    scored = sum(tot[k] for k in STAGES)
    for k in (0, 1, 2, 3, 4):
        w(f"  {STAGES[k]:3s} {tot[k]:10,d}  {tot[k]/scored:6.2%}")
    w(f"  excluded (-1) {tot[-1]:,} ({tot[-1]/n_all:.3%} of all epochs)")
    w(f"  scored {scored:,} of {n_all:,}")

    w("\nPER-COHORT STAGE PROPORTIONS (of scored epochs)")
    w(f"  {'cohort':8s} " + " ".join(f"{STAGES[k]:>7s}" for k in (0,1,2,3,4)))
    for c, g in m.groupby("cohort"):
        cnt = collections.Counter()
        for _, r in g.iterrows():
            cnt.update(np.load(mf.array_paths(data_dir, r.rec_id)["y"]).tolist())
        sc = sum(cnt[k] for k in STAGES) or 1
        w(f"  {c:8s} " + " ".join(f"{cnt[k]/sc:6.1%} " for k in (0,1,2,3,4)))

    text = "\n".join(L) + "\n" + FINDINGS
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {out}  ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
