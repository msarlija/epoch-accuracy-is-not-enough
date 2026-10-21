# -*- coding: utf-8 -*-
"""Table 1 of the paper: the fourteen cohorts as used.

Internal cohorts: every recording in the train, validation and test splits of
the trained fold models (each internal recording is in exactly one test split).
External cohorts: the configured external set after the per-cohort caps.
Per cohort: recordings, subjects, age (mean, SD, range), channels presented to
the model after channel QC, and the stage composition over all scored epochs.

    python scripts/cohort_table.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config                                    # noqa: E402
from data import manifest as mf                                   # noqa: E402
from data.sampling import apply_cohort_caps                       # noqa: E402

ROOT = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging"))
STAGES = ("W", "N1", "N2", "N3", "REM")


def channels(row):
    """channels presented to the model: those that passed channel QC"""
    f = row.get("qc_channel_flags")
    if isinstance(f, str):
        f = json.loads(f)
    return int(sum(f)) if isinstance(f, (list, tuple)) and len(f) else int(row["n_channels"])


def pct(c, total):
    """percentage rounded half up to one decimal, as in the paper"""
    return float(np.floor(1000 * c / total + 0.5) / 10)


def summarise(df, data_dir):
    rows, tot = [], np.zeros(5)
    for coh, g in df.groupby("cohort", sort=False):
        counts = np.zeros(5)
        for rid in g.rec_id:
            y = np.asarray(mf.open_arrays(data_dir, rid, mmap=True)["y"])
            counts += np.bincount(y[y >= 0], minlength=5)[:5]
        tot += counts
        age = pd.to_numeric(g.age, errors="coerce")
        rows.append(dict(cohort=coh, rec=len(g), subj=g.subject_id.nunique(),
                         age=f"{age.mean():.1f} ± {age.std():.1f} ({age.min():.0f}-{age.max():.0f})" if age.notna().any() else "n/a",
                         C=int(g.apply(channels, axis=1).max()),
                         **{s: pct(c, counts.sum()) for s, c in zip(STAGES, counts)}))
    t = pd.DataFrame(rows)
    hours = tot.sum() * 30 / 3600
    total = dict(cohort="total", rec=len(df), subj=df.subject_id.nunique(), age="", C="",
                 **{s: pct(c, tot.sum()) for s, c in zip(STAGES, tot)})
    return pd.concat([t, pd.DataFrame([total])], ignore_index=True), hours, int(tot.sum())


def main():
    cfg = load_config("configs/final_full.yaml")
    man = mf.load_manifest(cfg["data_dir"])
    man = man[man.qc_pass == 1]
    split = f"{ROOT}/runs/final_full_seed0_fold0of10/split_manifest.json"
    s = json.load(open(split))["splits"]
    internal = man[man.rec_id.isin(set(s["train"]) | set(s["val"]) | set(s["test"]))]
    external = apply_cohort_caps(man[man.cohort.isin(cfg["external_cohorts"])], cfg.get("cohort_caps") or {},
                                 seed=cfg.get("seed", 0))
    pd.set_option("display.width", 200)
    for name, df in (("INTERNAL", internal), ("EXTERNAL", external)):
        t, h, n = summarise(df, cfg["data_dir"])
        print(f"\n=== {name}: {len(df)} recordings, {df.subject_id.nunique()} subjects, "
              f"{h:,.0f} h of scored epochs ({n:,} epochs), {df.n_epochs.sum() * 30 / 3600:,.0f} h recorded")
        print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
