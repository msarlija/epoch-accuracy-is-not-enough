# -*- coding: utf-8 -*-
"""One visual style for every manuscript figure.

Import this before creating any axes:

    from paper_style import *          # colours, sizes, finish()

Rules the figures follow (so that they read as one set):
  * sans-serif text, 7 pt labels / 6.5 pt ticks, below the caption size;
  * every figure is SAVED at its final printed width (COL_W or COL_2) and
    included at that width, so text is never rescaled -- that, not the font
    setting, is what makes sizes identical across figures;
  * sentence-case axis labels with the unit in parentheses;
  * top and right spines removed, light horizontal or vertical grid only
    along the axis that carries the reading;
  * one fixed colour per sleep stage (STAGE_COLOR) and one fixed categorical
    palette for anything that is not a stage (SERIES); diverging maps only
    for signed quantities;
  * single-column figures are 3.45 in wide, double-column 7.1 in.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

COL_W = 3.5           # IEEE single column, inches (saved at this exact width)
COL_2 = 7.10          # IEEE double column, inches
WIDE = 5.4            # Elsevier single-column manuscript (elsarticle preprint, 390 pt)

INK, INK2, MUTED = "#111111", "#444444", "#8a8a85"
GRID = "#e3e2de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd1"]
NEG, POS = "#2a78d6", "#d03b3b"
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_COLOR = {"W": "#b9b8b3", "N1": "#8fd0b4", "N2": "#5b9bdf",
               "N3": "#1f4e86", "REM": "#d9534f"}
DIVERGING = LinearSegmentedColormap.from_list(
    "bl_gy_rd", ["#2a78d6", "#f2f1ee", "#d03b3b"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 6.5,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.labelpad": 3, "legend.handlelength": 1.4, "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25, "legend.columnspacing": 1.0,
    "legend.framealpha": 1.0, "legend.edgecolor": "none", "legend.facecolor": "white",
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.hspace": 0.04, "figure.constrained_layout.wspace": 0.04,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.labelcolor": INK, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "xtick.direction": "out",
    "ytick.direction": "out", "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
})


import os as _os
if _os.environ.get("PAPER_FONT_MIN"):
    # venue minimum for text inside figures (TNSRE: 8 pt); applied to column figures
    _m = float(_os.environ["PAPER_FONT_MIN"])
    plt.rcParams.update({"font.size": _m, "axes.labelsize": _m, "axes.titlesize": _m,
                         "xtick.labelsize": _m, "ytick.labelsize": _m, "legend.fontsize": _m})


def use_wide():
    """Switch to the full-width manuscript style: same rules, larger type so
    that a figure printed at 5.4 in carries 9 pt labels. Returns the width."""
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.7, "lines.linewidth": 1.6,
    })
    return WIDE


def despine(ax, left=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, path_base, dpi=300):
    r"""Write <base>.pdf and <base>.png at the figure's nominal size.

    No tight cropping: the saved width is exactly COL_W (or COL_2), so when
    LaTeX scales the file to \columnwidth the scale factor is 1 and every
    figure's 7 pt text is 7 pt on the page. Constrained layout keeps labels
    inside the canvas instead."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_base}.{ext}", dpi=dpi)
    plt.close(fig)
