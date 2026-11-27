"""Framework / study-design figure (the paper's Fig. 1).

One float that answers three questions a reader has on page 1: what object
moves, what \sys{} does to it, and how the claims about it are measured.
Device icons are drawn as vector primitives so the figure stays a small,
font-embedded PDF with no external assets.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

C_SRC = "#0072B2"      # source device / raw trajectory
C_SYS = "#D55E00"      # Relay
C_LED = "#009E73"      # ledger / safety
C_EVA = "#8C6D31"      # evaluation
C_MUT = "#666666"
C_BG = "#F4F4F4"

plt.rcParams.update({"font.size": 6.2, "pdf.fonttype": 42,
                     "font.family": "sans-serif"})


def _device(ax, x, y, kind, label, color=C_MUT, scale=1.0):
    """Minimal vector device icon: watch, speaker, phone or laptop."""
    s = scale
    if kind == "watch":
        ax.add_patch(Rectangle((x - .10 * s, y - .13 * s), .20 * s, .26 * s,
                               fc="white", ec=color, lw=1.0, zorder=3))
        ax.add_patch(Rectangle((x - .055 * s, y + .13 * s), .11 * s, .05 * s,
                               fc=color, ec=color, lw=0, zorder=3))
        ax.add_patch(Rectangle((x - .055 * s, y - .18 * s), .11 * s, .05 * s,
                               fc=color, ec=color, lw=0, zorder=3))
    elif kind == "speaker":
        ax.add_patch(FancyBboxPatch((x - .11 * s, y - .16 * s), .22 * s,
                                    .32 * s, boxstyle="round,pad=0.012",
                                    fc="white", ec=color, lw=1.0, zorder=3))
        ax.add_patch(Circle((x, y + .02 * s), .055 * s, fc="none", ec=color,
                            lw=0.8, zorder=4))
        ax.add_patch(Circle((x, y + .02 * s), .022 * s, fc=color, ec="none",
                            zorder=4))
    elif kind == "phone":
        ax.add_patch(FancyBboxPatch((x - .095 * s, y - .17 * s), .19 * s,
                                    .34 * s, boxstyle="round,pad=0.012",
                                    fc="white", ec=color, lw=1.0, zorder=3))
        ax.plot([x - .03 * s, x + .03 * s], [y - .13 * s] * 2, color=color,
                lw=0.9, zorder=4)
    else:  # laptop
        ax.add_patch(Rectangle((x - .15 * s, y - .02 * s), .30 * s, .21 * s,
                               fc="white", ec=color, lw=1.0, zorder=3))
        ax.add_patch(Rectangle((x - .20 * s, y - .06 * s), .40 * s, .045 * s,
                               fc=color, ec=color, lw=0, zorder=3))
    if label:
        ax.text(x, y - .26 * s, label, ha="center", va="top", fontsize=5.6,
                color="#222222")


def _box(ax, x, y, w, h, title, body, color, fs=5.5):
    """Titled box: title on the top rule, body centred in the remainder."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc="white", ec=color, lw=1.0, zorder=3))
    head = 0.17
    ax.text(x + w / 2, y + h - head / 2, title, ha="center", va="center",
            fontsize=fs + 0.6, color=color, fontweight="bold", zorder=4)
    ax.plot([x + 0.05, x + w - 0.05], [y + h - head] * 2, color=color,
            lw=0.5, alpha=0.45, zorder=4)
    ax.text(x + w / 2, y + (h - head) / 2, body, ha="center", va="center",
            fontsize=fs, color="#222222", zorder=4, linespacing=1.45)


def _arrow(ax, p, q, color=C_MUT, ls="-", lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=5))


def main():
    fig, ax = plt.subplots(figsize=(7.1, 2.62))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.65)
    ax.axis("off")

    # ---------------- (a) the problem ----------------
    ax.add_patch(Rectangle((0.05, 0.05), 2.62, 3.5, fc=C_BG, ec="none",
                           zorder=0))
    ax.text(1.36, 3.44, "(a) the object that must move", ha="center",
            fontsize=6.4, fontweight="bold", color="#222222")
    _device(ax, 0.62, 2.70, "laptop", "", C_SRC, scale=1.9)
    ax.text(0.62, 3.12, "laptop", ha="center", fontsize=5.6, color="#222222")
    ax.text(1.55, 2.86, "live trajectory", fontsize=5.8, color=C_SRC,
            fontweight="bold")
    ax.text(1.55, 2.62, "goal · tool results\nplan · dialogue", fontsize=5.2,
            color="#222222", linespacing=1.3, va="top")
    ax.add_patch(FancyBboxPatch((1.50, 1.93), 1.02, 0.30,
                                boxstyle="round,pad=0.012", fc="white",
                                ec=C_LED, lw=1.0, zorder=3))
    ax.text(2.01, 2.08, "actions already\ntaken in the world", ha="center",
            va="center", fontsize=5.0, color=C_LED, linespacing=1.25,
            zorder=4)
    _arrow(ax, (0.62, 2.20), (0.62, 1.58), C_SRC, lw=1.2)
    ax.text(0.78, 1.88, "user moves", fontsize=5.2, color=C_SRC,
            va="center")
    _device(ax, 0.62, 1.10, "watch", "watch", C_SRC, scale=1.7)
    ax.text(1.42, 1.42, "ship everything", fontsize=5.4, color="#B22222")
    ax.text(1.42, 1.24, "window truncates", fontsize=5.0, color="#555555")
    ax.text(1.42, 0.96, "restart", fontsize=5.4, color="#B22222")
    ax.text(1.42, 0.78, "work lost, actions\nre-executed", fontsize=5.0,
            color="#555555", va="top", linespacing=1.25)

    # ---------------- (b) Relay ----------------
    ax.text(5.30, 3.44, "(b) Relay: distil to the target's capability class",
            ha="center", fontsize=6.4, fontweight="bold", color="#222222")
    _box(ax, 2.90, 2.20, 1.62, 0.96, "DISTILLER",
         "value-density fill\nunder budget $B$ =\ntarget usable window", C_SYS)
    _box(ax, 4.72, 2.20, 1.72, 0.96, "STATE BLOCK $\\Sigma_k$",
         "GOAL · CONSTRAINTS\nENTITIES · DIGEST\nPLAN · TAIL", C_SYS)
    ax.add_patch(FancyBboxPatch((4.62, 1.50), 1.92, 0.56,
                                boxstyle="round,pad=0.012", fc="white",
                                ec=C_LED, lw=1.2, zorder=3))
    ax.text(5.58, 1.94, "SIDE-EFFECT LEDGER", ha="center", va="center",
            fontsize=5.6, color=C_LED, fontweight="bold", zorder=4)
    ax.plot([4.70, 6.46], [1.84, 1.84], color=C_LED, lw=0.5, alpha=0.45,
            zorder=4)
    ax.text(5.58, 1.69, "always shipped; action fingerprint\n"
                        "+ idempotency key", ha="center", va="center",
            fontsize=5.0, color="#444444", zorder=4, linespacing=1.45)
    _arrow(ax, (2.72, 2.68), (2.88, 2.68), C_SYS, lw=1.2)
    _arrow(ax, (4.54, 2.68), (4.70, 2.68), C_SYS, lw=1.2)
    _arrow(ax, (6.46, 2.42), (7.12, 2.42), C_SYS, lw=1.2)
    ax.text(6.79, 2.52, "gzip JSON\nBLE/WiFi/LTE", ha="center", fontsize=5.0,
            color="#444444", linespacing=1.25)
    _device(ax, 7.55, 2.42, "watch", "", C_SYS, scale=1.6)
    _device(ax, 8.25, 2.42, "speaker", "", C_SYS, scale=1.4)
    _device(ax, 8.95, 2.42, "phone", "", C_SYS, scale=1.4)
    ax.text(8.25, 1.98, "target class sets the budget;\n"
                        "dispatcher checks the ledger below the model",
            ha="center", va="top", fontsize=5.2, color="#222222",
            linespacing=1.3)

    # ---------------- (c) how it is measured ----------------
    ax.add_patch(Rectangle((2.78, 0.06), 7.17, 1.12, fc=C_BG, ec="none",
                           zorder=0))
    ax.text(2.92, 1.04, "(c) evidence bases, and what each supports",
            fontsize=6.4, fontweight="bold", color="#222222", va="center")
    _box(ax, 2.92, 0.16, 1.55, 0.72, "REAL TRACES",
         "184 trajectories\nrecall · ledger size", C_EVA, fs=5.2)
    _box(ax, 4.62, 0.16, 1.72, 0.72, "SYNTHETIC EPISODES",
         "exact ground truth\nidentity · continuation", C_EVA, fs=5.2)
    _box(ax, 6.47, 0.16, 1.66, 0.72, "PROCESSES + CODE",
         "SIGKILL · prototype · delegation\nbounded exhaustive (≤5 events)", C_LED, fs=4.9)
    _box(ax, 8.26, 0.16, 1.62, 0.72, "PROBES",
         "translated traces\nrewritten corrections", C_EVA, fs=5.2)

    fig.tight_layout(pad=0.15)
    out = os.path.join(FIG, "fig_framework.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("fig_framework done")


if __name__ == "__main__":
    main()
