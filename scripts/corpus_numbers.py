# -*- coding: utf-8 -*-
"""Two numbers of the text that come from the corpus and the fold predictions
rather than from a results table.

* Introduction: the share of expert-scored epochs that are REM in the first
  15 min after sleep onset in the internal corpus (sleep onset = first sleep
  epoch, as in the macrostructure definitions).
* Methods 2.7 (DA-MC): purity of the N2 cluster, the external epochs the
  unadapted fold models predict as N2 with posterior >= 0.7, by expert label.

    python scripts/corpus_numbers.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config  # noqa: E402

ROOT = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging"))
RUNS = f"{ROOT}/runs"


def main() -> int:
    data_dir = load_config("configs/final_full.yaml")["data_dir"]
    used = set()
    for f in range(10):
        used |= set(pd.read_csv(f"{RUNS}/final_full_seed0_fold{f}of10/stage_b/test/per_recording_metrics.csv").rec_id)
    tot = rem = early = 0
    for rid in sorted(used):
        y = np.load(f"{data_dir}/{rid}_y.npy").astype(int)
        sleep = np.flatnonzero((y >= 1) & (y <= 4))
        if not len(sleep):
            continue
        seg = y[sleep[0]:sleep[0] + 30]
        seg = seg[seg >= 0]
        tot += len(seg); rem += int((seg == 4).sum())
        r = np.flatnonzero(y[sleep[0]:] == 4)
        early += int(len(r) and r[0] <= 30)
    print(f"internal recordings {len(used)}: REM in the first 15 min after sleep onset {100 * rem / tot:.2f}% of "
          f"{tot} scored epochs; recordings with expert ROL <= 15 min: {early}")

    ids = set(pd.read_csv(f"{RUNS}/final_full_seed0_fold0of10/stage_b/external_v2/per_recording_metrics.csv").rec_id)
    tot = n2 = nrem = 0
    for f in range(10):
        for rid in ids:
            z = np.load(f"{RUNS}/final_full_seed0_fold{f}of10/stage_b/predictions_external_v2/{rid}.npz")
            p, y = z["proba"], z["y_true"].astype(int)
            n = min(len(p), len(y))
            p, y = p[:n], y[:n]
            k = (p.argmax(1) == 2) & (p.max(1) >= 0.7) & (y >= 0)
            tot += k.sum(); n2 += (y[k] == 2).sum(); nrem += np.isin(y[k], (1, 2, 3)).sum()
    print(f"N2 cluster on external data: {tot} epochs, N2 by expert {100 * n2 / tot:.1f}%, NREM {100 * nrem / tot:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
