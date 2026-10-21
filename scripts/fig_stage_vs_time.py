# -*- coding: utf-8 -*-
"""Fig. 2: the prior the time branch can learn.

Stacked distribution of expert-scored stages as a function of the model's own
time input over the internal corpus. The x axis is the normalized time feature
t = (epoch - lo) / TSO_DIVISOR, where lo is the trimmed recording start used
in training (scored sleep plus a random 20-40 min wake margin), so the figure
shows the stage composition the branch actually sees at each value of t.

    python scripts/fig_stage_vs_time.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_style  # noqa: E402
from paper_style import STAGES, STAGE_COLOR, despine, grid, plt, save  # noqa: E402
from data.dataset import wake_trim_bounds, TSO_DIVISOR  # noqa: E402

R = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "")
PREP = f"{R}/prepared/fs100_band0.2-30"
OUT = os.environ.get("PAPER_FIG_DIR", "figures/paper") + "/fig_stage_vs_time"
EPOCH_S = 30.0
SEQ_LEN, PAD_MIN, PAD_MAX, SEED = 60, 20.0, 40.0, 0   # final_base.yaml
T_MAX = 1.0                                             # 1000 epochs = 8.33 h
SMOOTH = 20                                             # epochs (10 min)


def main() -> int:
    wide = "--wide" in sys.argv
    COL_W = paper_style.use_wide() if wide else paper_style.COL_W
    out = OUT.replace("figures/paper/", "figures/paper/wide/") if wide else OUT
    internal = set()
    for p in glob.glob(f"{R}/runs/final_full_seed0_fold*of10/split_manifest.json"):
        internal |= set(json.load(open(p))["splits"]["test"])
    n_bins = int(T_MAX * TSO_DIVISOR) + 1
    counts = np.zeros((5, n_bins))
    tot = np.zeros(n_bins)
    for rid in sorted(internal):
        y = np.load(f"{PREP}/{rid}_y.npy")
        lo, hi = wake_trim_bounds(y, SEQ_LEN, PAD_MIN, PAD_MAX, rid, SEED)
        yy = y[lo:hi + 1]
        n = min(len(yy), n_bins)
        yy = yy[:n]
        for s in range(5):
            counts[s, :n] += (yy == s)
        tot[:n] += (yy >= 0)
    valid = tot >= 100
    P = counts / np.maximum(tot, 1)
    Ps = np.vstack([pd.Series(P[s]).rolling(SMOOTH, center=True, min_periods=1)
                    .mean().to_numpy() for s in range(5)])
    Ps = Ps / Ps.sum(0, keepdims=True).clip(min=1e-9)
    t = np.arange(n_bins) / TSO_DIVISOR

    hours = t * TSO_DIVISOR * EPOCH_S / 3600
    fig, ax = plt.subplots(figsize=(COL_W, 1.6 * COL_W / 3.5))
    order = ["W", "N1", "N2", "N3", "REM"]
    ax.stackplot(hours[valid], *[100 * Ps[STAGES.index(s)][valid] for s in order],
                 labels=order, colors=[STAGE_COLOR[s] for s in order], lw=0)
    ax.set_xlim(0, hours[valid].max())
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 9))
    ax.set_xlabel("Elapsed time from recording start (h)")
    ax.set_ylabel("Expert-scored stage (%)")
    grid(ax, "y"); despine(ax)
    ax.legend(loc="upper right", ncol=5, handlelength=1.0, handleheight=0.8,
              columnspacing=0.8, frameon=True, borderaxespad=0.3,
              fontsize=5.8 * COL_W / 3.5)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save(fig, out)
    print(f"wrote {out}.pdf/.png; {len(internal)} recordings; "
          f"valid up to t={t[valid].max():.2f} ({t[valid].max()*TSO_DIVISOR*EPOCH_S/3600:.1f} h)")

    # numbers for the caption
    def frac(stage, a, b):
        s = STAGES.index(stage)
        return 100 * counts[s, a:b].sum() / max(tot[a:b].sum(), 1)
    print(f"REM in first 15 min: {frac('REM', 0, 30):.2f}%   first 30 min: {frac('REM', 0, 60):.2f}%   "
          f"first 60 min: {frac('REM', 0, 120):.2f}%")
    print(f"N3 in first 2 h: {frac('N3', 0, 240):.1f}%   hours 2-4: {frac('N3', 240, 480):.1f}%   "
          f"after 6 h: {frac('N3', 720, n_bins):.1f}%")
    print(f"REM after 6 h: {frac('REM', 720, n_bins):.1f}%   W in first 30 min: {frac('W', 0, 60):.1f}%")
    for h in (0.25, 0.5, 1, 2, 4, 6, 8):
        i = int(h * 120)
        if i < n_bins and tot[i] >= 100:
            print(f"  {h:>4} h  " + "  ".join(f"{s} {100*P[STAGES.index(s), i]:.1f}" for s in STAGES)
                  + f"  n={int(tot[i])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
