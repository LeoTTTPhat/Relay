"""Protocol figure: where the duplicate check happens, and what a crash leaves.

Two panels, because the safety design has two independent parts and a
reader needs to see that they are independent:

  (a) the resumed agent proposes a typed call and a *trusted dispatcher*,
      not the model, decides -- the architectural choice that takes the
      model out of the trusted computing base for this property;
  (b) the action status machine with the write-ahead intent, and the crash
      points injected around it, so the reader can see which window each
      design leaves open.

Drawn with vector primitives only: no external assets, small embedded PDF.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

C_MODEL = "#0072B2"
C_SYS = "#D55E00"
C_OK = "#009E73"
C_BAD = "#CC3311"
C_MUT = "#666666"

plt.rcParams.update({"font.size": 7.2, "pdf.fonttype": 42})


def _box(ax, x, y, w, h, text, fc, ec=None, fs=7.2, weight="normal",
         tc="black", r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.004,rounding_size={r}",
                                linewidth=0.9, facecolor=fc,
                                edgecolor=ec or fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tc, linespacing=1.25)


def _arrow(ax, p, q, color=C_MUT, style="-|>", lw=0.9, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


def panel_a(ax):
    ax.set_title("(a) the check runs below the model", fontsize=7.8, pad=3)
    _box(ax, .02, .70, .30, .20,
         "resumed agent\n(any model)", "#DCE9F5", C_MODEL)
    ax.text(.17, .655, "proposes a typed call", ha="center", fontsize=6.6,
            color=C_MUT, style="italic")
    _arrow(ax, (.17, .70), (.17, .56), C_MODEL)

    _box(ax, .01, .34, .32, .21,
         "trusted dispatcher\ncanonicalise $\\to$ $fp$", "#FBE3D2", C_SYS,
         fs=6.9, weight="bold")
    _arrow(ax, (.33, .445), (.44, .445), C_SYS)
    _box(ax, .44, .34, .22, .21, "ledger\n$(fp,\\sigma,\\kappa)$", "#DFF1E8",
         C_OK)

    verdicts = [("execute", C_OK, .90), ("skip duplicate", C_OK, .74),
                ("reconcile", "#E69F00", .58), ("retry w/ key", C_OK, .42),
                ("block: stale epoch", C_BAD, .26)]
    for lbl, c, y in verdicts:
        _box(ax, .70, y - .055, .28, .11, lbl, "white", c, fs=6.7, tc=c)
        _arrow(ax, (.66, .445), (.70, y), c, lw=0.8, rad=0.12)

    ax.text(.17, .27, "the model never gets a vote:\na call the ledger says "
                      "happened\nnever reaches the tool",
            ha="center", va="top", fontsize=6.6, color=C_MUT)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(.12, 1)


def panel_b(ax):
    ax.set_title("(b) status, and what each crash leaves behind",
                 fontsize=7.8, pad=3)
    y = .70
    nodes = [(".02", "intended"), (".26", "prepared"), (".545", "committed")]
    _box(ax, .02, y, .19, .13, "intended", "white", C_MUT, fs=6.9)
    _box(ax, .26, y, .21, .13, "prepared\n(write-ahead)", "#FBE3D2", C_SYS,
         fs=6.9, weight="bold")
    _box(ax, .545, y + .10, .20, .12, "committed", "#DFF1E8", C_OK, fs=6.9)
    _box(ax, .545, y - .14, .20, .12, "failed", "#F6E3E3", C_BAD, fs=6.9)
    _box(ax, .545, y - .30, .20, .12, "unknown", "#FDF0D5", "#E69F00", fs=6.9)
    _box(ax, .80, y - .30, .18, .12, "reconcile", "white", "#E69F00", fs=6.9)

    _arrow(ax, (.21, y + .065), (.26, y + .065))
    _arrow(ax, (.47, y + .065), (.545, y + .16), C_OK, rad=-0.15)
    _arrow(ax, (.47, y + .065), (.545, y - .08), C_BAD, rad=0.15)
    _arrow(ax, (.365, y), (.365, y - .24), "#E69F00", ls="--")
    ax.text(.355, y - .075, "crash $\\to$ recover", fontsize=6.2,
            color="#E69F00", va="center", ha="right")
    # D2: an unobserved outcome (timeout, lost response) is not a failure
    ax.text(.355, y - .165, "timeout / lost\nresponse (D2)", fontsize=6.2,
            color="#E69F00", va="center", ha="right", linespacing=1.1)
    _arrow(ax, (.545, y - .24), (.365, y - .24), "#E69F00", style="-")
    _arrow(ax, (.745, y - .24), (.80, y - .24), "#E69F00")

    ax.text(.02, .30, "crash points injected:", fontsize=6.9,
            fontweight="bold", color=C_MUT)
    rows = [("c0 before prepare", "world unchanged", C_OK),
            ("c1 after prepare", "world unchanged, intent durable", C_OK),
            ("c2 in flight", "undetermined → must reconcile", "#E69F00"),
            ("c3 after response", "world changed, not yet recorded", C_BAD),
            ("c4 after record", "world changed, recorded", C_OK)]
    for i, (a, bl, c) in enumerate(rows):
        yy = .235 - i * .052
        ax.text(.04, yy, a, fontsize=6.4, color=c, family="monospace")
        ax.text(.34, yy, bl, fontsize=6.4, color=C_MUT)
    ax.text(.02, -.055,
            "a status-only ledger has no record at c2/c3: the action is "
            "invisible, not undetermined",
            fontsize=6.4, color=C_BAD, style="italic")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-.10, 1)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.32))
    panel_a(axes[0])
    panel_b(axes[1])
    fig.subplots_adjust(left=.005, right=.995, top=.90, bottom=.02, wspace=.06)
    out = os.path.join(FIG, "fig_protocol.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print("wrote", out)


if __name__ == "__main__":
    main()
