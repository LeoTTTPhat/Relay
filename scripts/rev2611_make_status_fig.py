"""Revision 2026-11: single-column status figure (replaces fig_protocol.pdf).

Panel (b) of ``make_protocol_fig.py`` redrawn for one column, with every label
at 7.5 pt or larger at its printed size and semantic colours only (blue: the
write-ahead record Relay adds; green: a resolved outcome; red: a duplicate
risk; grey: undetermined or secondary).  Panel (a) of the old figure is not
redrawn because Fig. 1 already shows where the check runs.  It plots no data.

Output: ``paper/figures/fig_status.pdf`` (printed at \\columnwidth = 3.5 in).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch     # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "figures", "fig_status.pdf")

BLUE = "#2E6FD8"
GREEN = "#2E9E4F"
RED = "#C62828"
GREY = "#4C5566"
LGREY = "#9AA3B2"
FS = 7.6

plt.rcParams.update({"font.size": FS, "pdf.fonttype": 42,
                     "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, text, ec, fc="white", weight="normal", tc="black",
        ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                linewidth=1.0, facecolor=fc, edgecolor=ec,
                                linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=FS, fontweight=weight, color=tc, linespacing=1.15)


def arrow(ax, p, q, color=GREY, ls="-", rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8,
                                 linewidth=1.0, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def main():
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.80
    box(ax, 0.012, y - 0.07, 0.18, 0.14, "intended", LGREY)
    box(ax, 0.25, y - 0.09, 0.25, 0.18, "prepared\n(written first)", BLUE,
        fc="#E3ECFA", weight="bold")
    box(ax, 0.60, y + 0.04, 0.22, 0.13, "committed", GREEN, fc="#E3F3E8")
    box(ax, 0.60, y - 0.17, 0.22, 0.13, "failed", RED, fc="#FBE5E5")
    box(ax, 0.60, y - 0.40, 0.19, 0.13, "unknown", GREY, ls="--")
    box(ax, 0.825, y - 0.40, 0.17, 0.13, "reconcile", GREY)
    arrow(ax, (0.192, y), (0.25, y))
    arrow(ax, (0.50, y + 0.03), (0.60, y + 0.10), GREEN, rad=-0.15)
    arrow(ax, (0.50, y - 0.03), (0.60, y - 0.10), RED, rad=0.15)
    arrow(ax, (0.375, y - 0.09), (0.375, y - 0.335), GREY, ls="--",
          style="-")
    arrow(ax, (0.375, y - 0.335), (0.60, y - 0.335), GREY, ls="--")
    ax.text(0.36, y - 0.215, "crash, timeout\nor lost reply", ha="right",
            va="center", fontsize=FS, color=GREY, linespacing=1.1)
    arrow(ax, (0.79, y - 0.335), (0.825, y - 0.335), GREY)

    ax.text(0.00, 0.31, "Crash point", fontsize=FS, fontweight="bold",
            color=GREY)
    ax.text(0.40, 0.31, "What the crash leaves", fontsize=FS,
            fontweight="bold", color=GREY)
    rows = [("c0 before prepared", "nothing happened", GREEN),
            ("c1 after prepared", "nothing happened, record kept", GREEN),
            ("c2 request in flight", "outcome unknown: reconcile", GREY),
            ("c3 after the reply", "effect done, not yet recorded", RED),
            ("c4 after the record", "effect done and recorded", GREEN)]
    for i, (a, b, c) in enumerate(rows):
        yy = 0.245 - i * 0.058
        ax.text(0.00, yy, a, fontsize=FS, color=c)
        ax.text(0.40, yy, b, fontsize=FS, color=GREY)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.0)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
