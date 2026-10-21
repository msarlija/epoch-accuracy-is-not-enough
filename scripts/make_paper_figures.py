# -*- coding: utf-8 -*-
"""Manuscript figures. One subcommand each, written to figures/paper/.

Colour follows the job, not taste:
  * the confusion difference is POLARITY (full sends more / fewer epochs here),
    so it gets a diverging blue-red ramp with a neutral grey midpoint -- never a
    rainbow, and never a hue at the midpoint, which would read as a value;
  * the K-curve carries IDENTITY (three label-free methods plus the
    supervised reference DA-O), so it gets categorical hues assigned in fixed
    order, plus direct labels so identity never rests on colour alone;
  * the forest plot is a single series and needs no hue at all.

Palette slots are taken in order from a pre-validated categorical set rather
than invented per figure.

    python scripts/make_paper_figures.py confusion
    python scripts/make_paper_figures.py kcurve
    python scripts/make_paper_figures.py forest
    python scripts/make_paper_figures.py all
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_style  # noqa: E402
from paper_style import (COL_2, INK, INK2, MUTED, GRID, SERIES,  # noqa: E402
                         NEG, POS, STAGES, DIVERGING, despine, grid, plt, save)
COL_W = paper_style.COL_W   # overridden by --wide
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from matplotlib.colors import TwoSlopeNorm          # noqa: E402

from config import external_cohorts                 # noqa: E402
from eval.stats import paired_bootstrap             # noqa: E402

RUNS = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/runs")
PAPER = os.path.expanduser(os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/analysis/paper")
OUT = os.environ.get("PAPER_FIG_DIR", "figures/paper")   # --wide appends /wide


# DA-3 at 1 and 2 reference subjects, measured with the anchor floor at 50
# epochs (sec 7.6). At the default 200-epoch floor its clusters are not all
# estimable and coverage collapses to 0.4% / 32%, so those runs are not
# comparable and are not plotted. Source: analysis/paper/da_runs/nostages_minep.csv.
MINEP_K1, MINEP_K2 = 0.0076, 0.0078   # seven cohorts, da_paper_numbers.py (DA-3, 50-epoch floor); --minep

# DA-O, the supervised reference: the classifier head fine-tuned on the expert
# labels of K target subjects (methods.md sec 7.10, branch
# worktree-da4-self-training). Its last point is half of each cohort, not
# "every other subject", because the halves protocol is inductive.
HEAD_CSV = os.path.expanduser(
    os.environ.get("SLEEP_DATA_ROOT", "~/data/sleep_staging") + "/analysis/da4/head_kcurve10_pooled.csv")



def _recs(arm, split, folds=10, stage="stage_b"):
    fr = []
    for f in range(folds):
        p = (f"{RUNS}/final_{arm}_seed0_fold{f}of10/{stage}/{split}/"
             "per_recording_metrics.csv")
        d = pd.read_csv(p)
        if "external" in split:
            d = d[d.cohort.isin(external_cohorts())]
        d["key"] = d.rec_id.astype(str) + f"|f{f}"
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


def _finish(fig, name):
    os.makedirs(OUT, exist_ok=True)
    save(fig, f"{OUT}/{name}")
    print(f"  wrote {OUT}/{name}.png and .pdf")


# ---------------------------------------------------------------- confusion
def fig_confusion(args):
    """Where the epoch-level cost is paid.

    A single 5x5 difference matrix conflates two different quantities on one
    colour scale. The diagonal is a change in RECALL -- a scorecard, where
    negative reads as "worse" -- while the off-diagonal is a REDISTRIBUTION of
    misclassified epochs, where sign only means direction of flow. Printed
    together, the eye reads the whole grid as a scorecard and concludes the
    time-aware model is simply worse, which is not what the matrix shows.

    So they are split. Left: the recall change per stage, stated plainly,
    including the four stages that lose. Right: where the reallocated epochs
    go, diagonal masked. The claim the figure supports is not "better" but
    "the cost is structured" -- almost every reallocated epoch lands in N2.
    """
    def conf(arm):
        c = np.zeros((5, 5), dtype=np.int64)
        for f in range(args.folds):
            d = (f"{RUNS}/final_{arm}_seed0_fold{f}of10/stage_b/"
                 "predictions_external_v2")
            # the directory still holds predictions for cohorts that were
            # later dropped from the external set (STAGES-MSMI); keep only the
            # recordings of config.external_cohorts(), as every table does
            m = pd.read_csv(f"{RUNS}/final_{arm}_seed0_fold{f}of10/"
                            "stage_b/external_v2/per_recording_metrics.csv")
            keep = set(m[m.cohort.isin(external_cohorts())].rec_id.astype(str))
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".npz") or fn[:-4] not in keep:
                    continue
                z = np.load(os.path.join(d, fn))
                t, p = z["y_true"], z["y_pred"]
                m = (t >= 0) & (p >= 0)
                np.add.at(c, (t[m].astype(int), p[m].astype(int)), 1)
        return c

    ca, cb = conf("full"), conf(args.arm_b)
    pa = 100 * ca / ca.sum(1, keepdims=True).clip(min=1)
    pb = 100 * cb / cb.sum(1, keepdims=True).clip(min=1)
    d = pa - pb
    recall = np.diag(d)
    off = d.copy()
    np.fill_diagonal(off, np.nan)

    fig, (axb, axm) = plt.subplots(
        1, 2, figsize=(COL_W, 1.7 * COL_W / 3.5), gridspec_kw={"width_ratios": [1.0, 2.0]})

    # ---- left: the recall change, not hidden
    yy = np.arange(5)
    cols = [POS if v > 0 else NEG for v in recall]
    axb.barh(yy, recall, color=cols, height=0.62)
    axb.axvline(0, color=MUTED, lw=0.8)
    for i, v in enumerate(recall):
        if abs(v) >= 1.3:          # label fits inside the bar
            axb.text(v / 2, i, f"{v:+.2f}", va="center", ha="center",
                     fontsize=max(5.5 * COL_W / 3.5, float(os.environ.get("PAPER_FONT_MIN", 0))), color="white")
        else:
            axb.text(v + (0.08 if v > 0 else -0.08), i, f"{v:+.2f}",
                     va="center", ha="left" if v > 0 else "right",
                     fontsize=max(5.5 * COL_W / 3.5, float(os.environ.get("PAPER_FONT_MIN", 0))), color=INK)
    axb.set_yticks(yy); axb.set_yticklabels(STAGES)
    axb.invert_yaxis()
    axb.set_xlim(-2.8, 2.0)
    axb.set_xticks([-2, -1, 0, 1])
    axb.set_xlabel("Change in recall (pp)")
    grid(axb, "x"); despine(axb, left=False)

    # ---- right: where the reallocated epochs land
    lim = np.nanmax(np.abs(off))
    cmap = DIVERGING.copy(); cmap.set_bad("#efefec")
    im = axm.imshow(np.ma.masked_invalid(off), cmap=cmap,
                    norm=TwoSlopeNorm(0, -lim, lim))
    for i in range(5):
        for j in range(5):
            if i == j:
                axm.text(j, i, "—", ha="center", va="center", fontsize=7,
                         color=MUTED)
                continue
            v = off[i, j]
            axm.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=max(5.3 * COL_W / 3.5, float(os.environ.get("PAPER_FONT_MIN", 0))),
                     color=INK if abs(v) < 0.62 * lim else "white")
    axm.set_xticks(range(5)); axm.set_xticklabels(STAGES)
    axm.set_yticks(range(5)); axm.set_yticklabels(STAGES)
    axm.set_xlabel("Predicted stage"); axm.set_ylabel("Expert stage")
    axm.set_xticks(np.arange(-.5, 5, 1), minor=True)
    axm.set_yticks(np.arange(-.5, 5, 1), minor=True)
    axm.grid(which="minor", color="white", linewidth=1.4)
    axm.tick_params(which="minor", length=0)
    for sp in axm.spines.values():
        sp.set_visible(False)
    cb_ = fig.colorbar(im, ax=axm, fraction=0.046, pad=0.04)
    _fm = float(os.environ.get("PAPER_FONT_MIN", 0))
    cb_.outline.set_visible(False); cb_.ax.tick_params(length=2, labelsize=max(6 * COL_W / 3.5, _fm))
    cb_.set_label("Epochs sent here (pp)", color=INK2, fontsize=max(6.5 * COL_W / 3.5, _fm))
    _finish(fig, "fig_confusion_difference")


# ------------------------------------------------------------------ kcurve
def fig_kcurve(args):
    """Pooled change in kappa against the number of reference subjects.

    Values come from analysis/paper/da_runs/current_kcurve.csv, written by
    scripts/da_paper_numbers.py on the current config.external_cohorts(), so
    the figure and the manuscript text are derived from one computation.
    """
    src = args.loso_csv or f"{PAPER}/da_runs/current_kcurve.csv"
    if not os.path.exists(src):
        raise SystemExit(f"missing {src}; run scripts/da_paper_numbers.py first")
    df = pd.read_csv(src)
    df["K"] = df["K"].astype(str)
    # the largest finite n_ref is whatever the table carries (15 or 20)
    kmax = "15" if (df.K == "15").any() else "20"
    order = ["1", "2", "5", "10", kmax, "all"]
    xs = [1, 2, 5, 10, int(kmax), 40]

    def series(name, ks=order):
        g = df[df.series == name].set_index("K").dk
        return np.array([g.get(k, np.nan) for k in ks], dtype=float)

    ys = {k: series(k) for k in ("DA-1", "DA-2", "DA-3", "DA-O")}
    # older tables carry a "DA-3 subset" series: DA-3 at 1-2 subjects from the
    # 50-epoch-floor run on the recordings with all three clusters, drawn with
    # open markers. With the missing-cluster fallback DA-3 covers every
    # recording and is one solid curve.
    has_subset = (df.series == "DA-3 subset").any()
    if has_subset:
        sub = df[df.series == "DA-3 subset"].set_index("K")
        ys["DA-3"][0] = sub.dk.get("1", np.nan); ys["DA-3"][1] = sub.dk.get("2", np.nan)
    ymax = 0.010 * np.ceil(np.nanmax(ys["DA-O"]) / 0.010)
    ymin = 0.010 * np.floor(min(np.nanmin(v) for v in ys.values()) / 0.010)

    fig, ax = plt.subplots(figsize=(COL_W, 1.5 * COL_W / 3.5))
    ax.axhspan(ymin, 0, color="#f7eeee", zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    style = {"DA-1": ("o", SERIES[0], "-", "DA-1 global"),
             "DA-2": ("^", SERIES[1], "-", "DA-2 N2 anchor"),
             "DA-3": ("s", SERIES[2], "-", "DA-3 multi-cluster"),
             "DA-O": ("D", SERIES[3], "--", "DA-O supervised")}
    for k, (mk, col, ls, lab) in style.items():
        # DA-3 at 1-2 subjects comes from a different (50-epoch) run on a
        # subset of recordings: join it to the full-set curve with a dotted line
        x0 = 2 if (k == "DA-3" and has_subset) else 0
        ax.plot(xs[x0:], ys[k][x0:], ls, marker=mk, color=col, lw=1.5 if ls == "-" else 1.3,
                ms=3.8, zorder=3, markeredgecolor="white", markeredgewidth=0.6, label=lab)
    if has_subset:
        ax.plot(xs[:3], ys["DA-3"][:3], ":", color=SERIES[2], lw=1.3, zorder=2)
        ax.plot(xs[:2], ys["DA-3"][:2], "s", mfc="white", mec=SERIES[2], mew=1.3, ms=4.6,
                zorder=4, label="DA-3 subset")
    # right half of the shaded band, clear of the x-axis and of DA-1, which is
    # below zero only at one and two reference subjects
    ax.text(30, ymin / 2, "adaptation harms", fontsize=max(5.8 * COL_W / 3.5, float(os.environ.get("PAPER_FONT_MIN", 0))), color="#9c3a3a",
            ha="center", va="center", style="italic")

    ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(order)
    ax.set_xlim(0.88, 52)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(ymin, ymax + 1e-9, 0.01))
    ax.set_xlabel("Reference subjects from the target cohort")
    ax.set_ylabel("Change in Cohen's κ")
    grid(ax, "y"); despine(ax)
    ax.minorticks_off()
    # above the axes: every region inside is crossed by one of the five series
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
              fontsize=5.8 * COL_W / 3.5, handlelength=1.4, labelspacing=0.2,
              ncol=3, columnspacing=1.0, borderaxespad=0.2)
    _finish(fig, "fig_data_efficiency")


# ------------------------------------------------------------------ forest
def fig_forest(args):
    """Per-cohort within-10-minute agreement gain, time-aware - time-naive.

    The endpoint is the proportion of evaluations whose predicted REM onset
    latency lands within 10 min of the expert value -- an accuracy, not a mean
    error, because half the recordings sit near zero error and a mean is
    governed by a handful of catastrophic ones. Positive is better, which keeps
    the reading direction the same as every other panel.
    """
    A = _recs("full", args.split)
    B = _recs(args.arm_b, args.split)
    m = A[["key", "rec_id", "cohort", "subject_id", "rol_ae_min"]].merge(
        B[["key", "rol_ae_min"]], on="key", suffixes=("_a", "_b")).dropna()
    # rec_id repeats once per fold; pair on the fold-qualified key, cluster on
    # the real subject so the interval counts people, not evaluations
    m = m.rename(columns={"rec_id": "recording"})
    m["rec_id"] = m["key"]
    T = args.within

    def _bs(g):
        a = g[["rec_id", "cohort", "subject_id"]].copy()
        a["v"] = (g.rol_ae_min_a <= T).astype(float)
        b = g[["rec_id"]].copy()
        b["v"] = (g.rol_ae_min_b <= T).astype(float)
        return paired_bootstrap(a, b, "v", n_boot=10000, seed=0)

    rows = []
    for coh, g in m.groupby("cohort"):
        r = _bs(g)
        rows.append((coh, g.recording.nunique(), 100 * r["mean_diff"],
                     100 * r["ci_low"], 100 * r["ci_high"]))
    rp = _bs(m)

    d = pd.DataFrame(rows, columns=["cohort", "n", "diff", "lo", "hi"])
    d = d.sort_values("diff", ascending=False)
    d["cohort"] = d.cohort.replace({"DODH": "DOD-H", "DODO": "DOD-O", "MASS-SS1": "MASS SS1"})
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(COL_W, 1.7 * COL_W / 3.5))
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    for i, r in enumerate(d.itertuples()):
        col = SERIES[0] if r.diff > 0 else POS
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.3, alpha=.6,
                solid_capstyle="round", zorder=2)
        ax.plot(r.diff, i, "o", color=col, ms=4.5, zorder=3,
                markeredgecolor="white", markeredgewidth=0.7)
    ax.plot(100 * rp["mean_diff"], len(d), "D", color=INK, ms=5, zorder=3)
    ax.plot([100 * rp["ci_low"], 100 * rp["ci_high"]], [len(d), len(d)],
            color=INK, lw=1.5, solid_capstyle="round", zorder=2)
    labels = [f"{r.cohort} (n = {r.n})" for r in d.itertuples()] + ["Pooled"]
    ax.set_yticks(list(y) + [len(d)]); ax.set_yticklabels(labels)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_ylim(-0.7, len(d) + 0.7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Change in REM-onset agreement within {T:g} min (pp)")
    grid(ax, "x"); despine(ax, left=False)
    _finish(fig, f"fig_rol_within{T:g}_by_cohort")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", choices=("confusion", "kcurve", "forest", "all"))
    ap.add_argument("--arm-b", default="zero")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--within", type=float, default=10.0,
                    help="agreement tolerance in minutes (the paper endpoint)")
    ap.add_argument("--split", default="external_v2")
    ap.add_argument("--loso-csv", default="",
                    help="pooled DA summary to draw the K-curve from "
                         "(default analysis/paper/da_runs/loso.csv)")
    ap.add_argument("--coverage", nargs=2, type=float, default=(66, 95),
                    help="DA-3 coverage at K=1 and K=2 for the annotation")
    ap.add_argument("--wide", action="store_true",
                    help="full-width manuscript versions (5.4 in, 9 pt) in "
                         "figures/paper/wide; the column versions are kept")
    ap.add_argument("--head-csv", default="",
                    help="pooled supervised-reference (DA-O) K-curve "
                         f"(default {HEAD_CSV})")
    ap.add_argument("--minep", nargs=2, type=float, default=None,
                    help="DA-3 delta-kappa at K=1 and K=2 from the 50-epoch "
                         "anchor-floor run, overriding the constants")
    args = ap.parse_args()
    global COL_W, OUT
    if args.wide:
        COL_W = paper_style.use_wide()
        OUT = os.path.join(OUT, "wide")
    jobs = {"confusion": fig_confusion, "kcurve": fig_kcurve,
            "forest": fig_forest}
    for k, fn in jobs.items():
        if args.which in (k, "all"):
            print(f"[{k}]")
            fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
