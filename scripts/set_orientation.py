# -*- coding: utf-8 -*-
"""Signal orientation of the prepared corpus (Section 2.2, Supplementary Section 1).

Writes the per-recording ``sign`` column of the manifest. Every derivation is
used with the (pre)frontal electrode referenced to the mastoid or ear
electrode. The sign is applied when the arrays are read (data/dataset.py,
run_experiment.predict_recording, domain_adapt.collect_features), so the
prepared arrays themselves are left as they are.

* CFS, MrOS, SOF: the NSRR exports store reference minus electrode, so the
  stored mastoid channels are already Fpz-M1 / Fpz-M2. The cohort specs
  negate them (data/cohorts.py), which inverts them a second time, and
  sign = -1 restores the standard orientation.
* HomePAP nights without a recorded Fpz (stored M1/M2, 95 nights): the stored
  sign varies from night to night. It is set from the slow-oscillation /
  spindle coupling index of the stored signal (orientation_so_spindle.py),
  averaged over the QC-passed channels on up to 300 expert N3 epochs:
  >= +0.1 with every channel agreeing in sign keeps the night (+1),
  <= -0.1 with every channel agreeing flips it (-1), and anything else, or
  fewer than 10 N3 epochs, keeps the stored orientation (+1). The 45 nights
  that record Fpz are formed as Fpz-M1 / Fpz-M2 by the reader and keep +1.
* every other recording: +1.

    python scripts/set_orientation.py [--config configs/preprocess.yaml] [--dry-run]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_config                      # noqa: E402
from orientation_so_spindle import EP, per_channel  # noqa: E402

FLIP = ("CFS", "MrOS", "SOF")
THR = 0.1


def hpap_night_sign(data_dir, row):
    """(sign, index as stored, decision) for one HomePAP night without Fpz"""
    x = np.load(f"{data_dir}/{row.rec_id}_x.npy", mmap_mode="r")
    y = np.load(f"{data_dir}/{row.rec_id}_y.npy").astype(int)
    n3 = np.flatnonzero(y == 3)
    n3 = n3[(n3 + 1) * EP <= x.shape[0]][:300]
    if len(n3) < 10:
        return "1", None, "keep: uncertain or too little N3"
    idx = (n3[:, None] * EP + np.arange(EP)).ravel()
    flags = json.loads(row.qc_channel_flags) if isinstance(row.qc_channel_flags, str) else row.qc_channel_flags
    chans = [i for i, v in enumerate(flags) if v == 1] or [0]
    vals = []
    for c in chans:
        o = per_channel(np.asarray(x[:, c], np.float64), idx)
        if o:
            vals.append(o["index"])
    if not vals:
        return "1", None, "keep: uncertain or too little N3"
    index = float(np.mean(vals))
    agree = bool(np.all(np.sign(vals) == np.sign(vals[0])))
    if agree and index <= -THR:
        return "-1", round(index, 3), "flip: clearly inverted"
    if agree and index >= THR:
        return "1", round(index, 3), "keep: clearly standard"
    return "1", round(index, 3), "keep: uncertain or too little N3"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--dry-run", action="store_true", help="report the signs, do not write the manifest")
    args = ap.parse_args()
    data_dir = load_config(args.config)["data_dir"]
    path = f"{data_dir}/manifest.csv"
    # strings throughout, so the manifest round-trips unchanged apart from the new column
    m = pd.read_csv(path, dtype=str, keep_default_na=False)
    m["sign"] = ["-1" if c in FLIP else "1" for c in m.cohort]
    nofpz = m.cohort.eq("HPAP") & ~m.channel_names.str.contains("FPz")
    decisions = []
    for _, r in m[nofpz].iterrows():
        s, idx, why = hpap_night_sign(data_dir, r)
        decisions.append(dict(rec_id=r.rec_id, sign=s, spindle_index_as_stored=idx, decision=why))
    D = pd.DataFrame(decisions)
    if len(D):
        m.loc[nofpz, "sign"] = m.loc[nofpz, "rec_id"].map(dict(zip(D.rec_id, D.sign)))
        print("HomePAP nights without Fpz:")
        print(D.decision.value_counts().to_string())
    print("sign per cohort:")
    print(m.groupby("cohort").sign.agg(lambda v: " ".join(f"{k}:{n}" for k, n in v.value_counts().items())).to_string())
    if args.dry_run:
        return 0
    if len(D):
        D.to_csv(f"{data_dir}/hpap_spindle_signs.csv", index=False)
    m.to_csv(path, index=False)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
