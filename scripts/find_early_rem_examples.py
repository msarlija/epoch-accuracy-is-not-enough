# -*- coding: utf-8 -*-
"""Find illustrative recordings for the false-early-REM figure.

Selects recordings where the time-naive arm places REM onset implausibly early
and the time-aware arm does not, WHILE the two agree closely on epoch-level
accuracy -- the point of the figure being that a metric which summarises the
night epoch by epoch cannot see the difference that matters clinically.

Candidates are ranked by a transparent score rather than cherry-picked:
  * the control must produce a false early onset (pred <= --early, true above);
  * the time-aware arm must land within --tolerance of the true latency;
  * the two arms' epoch accuracy must differ by no more than --acc-delta;
  * among survivors, prefer SMALL accuracy difference (the paradox is starkest)
    and LARGE latency difference (the figure is legible).

Every candidate is printed with the numbers a reader would need to check it, and
--plot writes one hypnogram panel per candidate: truth, time-aware, time-naive.

    python scripts/find_early_rem_examples.py --arm-b none --plot
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNS = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/runs")
NAMES = ["W", "N1", "N2", "N3", "REM"]
# plotting order puts REM adjacent to wake, the conventional hypnogram layout
ROW = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
ROWNAME = ["N3", "N2", "N1", "REM", "W"]


def rec_table(prefix, arm, split, folds):
    fr = []
    for f in range(folds):
        p = (f"{RUNS}/{prefix}_{arm}_seed0_fold{f}of10/stage_b/{split}/"
             f"per_recording_metrics.csv")
        d = pd.read_csv(p)[["rec_id", "cohort", "accuracy", "kappa",
                            "rol_true_min", "rol_pred_min", "rol_ae_min"]]
        d["fold"] = f
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


def load_pred(prefix, arm, split, fold, rec_id):
    p = (f"{RUNS}/{prefix}_{arm}_seed0_fold{fold}of10/stage_b/"
         f"predictions_{split}/{rec_id}.npz")
    z = np.load(p)
    return z["y_true"], z["y_pred"]


def plot(cands, prefix, arm_a, arm_b, split, outdir, epoch_s=30,
         paper=False, wide=False):
    """One panel per arm: expert, time-aware, time-naive.

    REM onset latency is defined from each hypnogram's OWN sleep onset (the
    first sleep epoch), not from the start of the recording, so both are marked
    -- a figure showing only the first REM epoch would disagree with the ROL in
    the caption and invite exactly the confusion it is meant to resolve.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import paper_style
    from paper_style import INK, INK2, MUTED, SERIES, POS, despine, grid, plt, save
    COL_W = paper_style.use_wide() if wide else paper_style.COL_W
    SLEEP = (1, 2, 3, 4)
    os.makedirs(outdir, exist_ok=True)
    for _, r in cands.iterrows():
        yt, pa = load_pred(prefix, arm_a, split, int(r.fold), r.rec_id)
        _, pb = load_pred(prefix, arm_b, split, int(r.fold), r.rec_id)
        n = len(yt)
        t = np.arange(n) * epoch_s / 3600.0
        size = (COL_W, 2.4 * COL_W / 3.5) if paper else (11, 6.4)
        fig, axes = plt.subplots(3, 1, figsize=size, sharex=True)
        for ax, y, lab in zip(axes, (yt, pa, pb),
                              ("Expert", "Time-aware", "Time-na\u00efve")):
            y = np.asarray(y)
            step = np.array([ROW.get(int(x), np.nan) if x >= 0 else np.nan
                             for x in y], dtype=float)
            ax.step(t, step, where="post", lw=0.6, color=INK)
            ax.set_yticks(range(5)); ax.set_yticklabels(ROWNAME)
            ax.set_ylim(-0.6, 4.9); ax.set_ylabel(lab)
            ax.set_xlim(0, t[-1])
            grid(ax, "x"); despine(ax)
            on = np.flatnonzero(np.isin(y, SLEEP))
            rem = np.flatnonzero(y == 4)
            if on.size:
                ax.axvline(on[0] * epoch_s / 3600.0, color=SERIES[0],
                           lw=0.9, ls=":")
            if rem.size:
                x = rem[0] * epoch_s / 3600.0
                ax.axvline(x, color=POS, lw=1.0, ls="--")
                rol = ((rem[0] - on[0]) * epoch_s / 60.0) if on.size else float("nan")
                ax.text(x + 0.08, 4.6, f"ROL {rol:.0f} min", color=POS,
                        fontsize=max(6 * COL_W / 3.5, float(os.environ.get("PAPER_FONT_MIN", 0))), va="top")
        axes[0].plot([], [], color=SERIES[0], lw=0.9, ls=":", label="sleep onset")
        axes[0].plot([], [], color=POS, lw=1.0, ls="--", label="first REM epoch")
        fig.legend(loc="outside upper right", ncol=2, handlelength=1.8,
                   borderaxespad=0.0)
        axes[-1].set_xlabel("Time from start of recording (h)")
        if not paper:
            axes[0].set_title(
                f"{r.rec_id}  ({r.cohort}, fold {int(r.fold)})\n"
                f"REM onset latency: expert {r.rol_true_min:.0f} min | "
                f"time-aware {r.rol_pred_min_a:.0f} min | "
                f"time-naive {r.rol_pred_min_b:.0f} min        "
                f"epoch accuracy {100*r.accuracy_a:.1f}% vs "
                f"{100*r.accuracy_b:.1f}%", fontsize=9)
        if paper:
            base = os.path.join(outdir, f"fig_early_rem_{r.cohort}")
            save(fig, base)
            print(f"  wrote {base}.pdf and .png")
        else:
            out = os.path.join(
                outdir, f"early_rem_{r.cohort}_{r.rec_id}_f{int(r.fold)}.png")
            fig.savefig(out, dpi=160)
            print(f"  wrote {out}")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", default="final")
    ap.add_argument("--arm-a", default="full", help="time-aware")
    ap.add_argument("--arm-b", default="zero", help="time-naive control")
    ap.add_argument("--split", default="external_v2")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--early", type=float, default=15.0,
                    help="control predicts ROL <= this while truth is above it")
    ap.add_argument("--naive-rol-min", type=float, default=0.0,
                    help="require the control's predicted ROL to be at least "
                         "this, to find illustrations that are wrong without "
                         "being degenerate. Note the false-early-REM endpoint "
                         "is defined at --early, so a value above it selects "
                         "recordings that are NOT false early onsets, only "
                         "large latency errors.")
    ap.add_argument("--tolerance", type=float, default=15.0,
                    help="time-aware arm must land within this of the truth")
    ap.add_argument("--acc-delta", type=float, default=0.02,
                    help="max allowed epoch-accuracy difference between arms")
    ap.add_argument("--min-accuracy", type=float, default=0.70,
                    help="exclude recordings where EITHER arm scores below "
                         "this; a poorly-scored night is a bad illustration "
                         "because the reader cannot tell the contrast from "
                         "general failure")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--only", default="",
                    help="plot just this rec_id (paper figure)")
    ap.add_argument("--paper", action="store_true",
                    help="publication form: no title, vector PDF alongside the "
                         "PNG, cohort label instead of the raw recording id")
    ap.add_argument("--outdir", default="figures/early_rem")
    ap.add_argument("--wide", action="store_true",
                    help="full-width manuscript version (5.4 in, 9 pt)")
    args = ap.parse_args()

    A = rec_table(args.prefix, args.arm_a, args.split, args.folds)
    B = rec_table(args.prefix, args.arm_b, args.split, args.folds)
    m = A.merge(B, on=["rec_id", "fold", "cohort"], suffixes=("_a", "_b"))
    m = m.dropna(subset=["rol_true_min_a", "rol_pred_min_a", "rol_pred_min_b"])

    sel = m[(m.rol_pred_min_b <= args.early) &
            (m.rol_pred_min_b >= args.naive_rol_min) &
            (m.rol_true_min_a > args.early) &
            (m.rol_ae_min_a <= args.tolerance) &
            ((m.accuracy_a - m.accuracy_b).abs() <= args.acc_delta) &
            (m.accuracy_a >= args.min_accuracy) &
            (m.accuracy_b >= args.min_accuracy)].copy()
    sel["acc_gap"] = (sel.accuracy_a - sel.accuracy_b).abs()
    sel["rol_gap"] = (sel.rol_pred_min_a - sel.rol_pred_min_b).abs()
    sel["rol_true_min"] = sel.rol_true_min_a
    # small accuracy gap first, then large latency gap
    # prefer a control prediction that is clearly early but not degenerate,
    # a time-aware arm close to truth, and near-identical epoch accuracy
    sel["score"] = (sel.rol_pred_min_b.clip(lower=0.5)
                    / (1.0 + sel.rol_ae_min_a) / (1.0 + 100 * sel.acc_gap))
    sel = sel.sort_values("score", ascending=False)

    print(f"{len(m)} paired evaluations; {len(sel)} satisfy every criterion "
          f"({sel.rec_id.nunique()} distinct recordings)\n")
    if sel.empty:
        print("no candidates -- relax --tolerance or --acc-delta")
        return 1

    cols = ["rec_id", "cohort", "fold", "rol_true_min", "rol_pred_min_a",
            "rol_pred_min_b", "accuracy_a", "accuracy_b", "kappa_a", "kappa_b"]
    if args.only:
        sel = sel[sel.rec_id == args.only]
        if sel.empty:
            raise SystemExit(f"{args.only} does not satisfy the criteria")
    top = sel.drop_duplicates("rec_id").head(args.top)
    print(f"  {'rec_id':<34}{'cohort':<12}{'fold':>5}{'trueROL':>9}"
          f"{'aware':>7}{'naive':>7}{'acc_aw':>8}{'acc_nv':>8}{'d_acc':>9}")
    for _, r in top.iterrows():
        rid = r.rec_id if len(r.rec_id) <= 33 else r.rec_id[:15] + ".." + r.rec_id[-16:]
        print(f"  {rid:<34}{r.cohort:<12}{int(r.fold):>5}"
              f"{r.rol_true_min:>9.0f}{r.rol_pred_min_a:>7.0f}"
              f"{r.rol_pred_min_b:>7.0f}{100*r.accuracy_a:>7.1f}%"
              f"{100*r.accuracy_b:>7.1f}%{100*(r.accuracy_a-r.accuracy_b):>+7.2f}pp")

    out = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/analysis/early_rem_examples.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sel[cols + ["acc_gap", "rol_gap"]].to_csv(out, index=False)
    print(f"\nwrote {out}  (all {len(sel)} candidates)")

    if args.plot:
        print(f"\nplotting top {len(top)}:")
        plot(top, args.prefix, args.arm_a, args.arm_b, args.split, args.outdir,
             paper=args.paper, wide=args.wide)
    return 0


if __name__ == "__main__":
    sys.exit(main())
