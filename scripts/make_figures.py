"""Generate all paper figures (vector PDF, colorblind-safe Okabe-Ito subset,
validated: CVD deltaE pass; identity always double-encoded with markers)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# fixed policy -> (color, marker, label); never re-ordered or cycled
STYLE = {
    "full":       ("#0072B2", "o", "Full context"),
    "trunc25":    ("#E69F00", "s", "Last-25%"),
    "recency":    ("#009E73", "^", "Recency+goal"),
    "typed-recency": ("#117733", "<", "Typed-recency"),
    "extractive": ("#8C6D31", "P", "TextRank"),
    "selective":  ("#7B3294", "*", "Selective-Ctx"),
    "llmlingua":  ("#AA4499", "1", "LongLLMLingua*"),
    "llm-summary": ("#44AA99", "2", "LLM summary"),
    "relay":      ("#D55E00", "D", "Relay"),
    "relay-lite": ("#D55E00", "d", "Relay-lite"),
    "restart":    ("#56B4E9", "v", "Restart"),
    "randdrop":   ("#CC79A7", "x", "Random-drop"),
}
ORDER = ["full", "trunc25", "recency", "typed-recency", "extractive",
         "relay", "relay-lite", "restart", "randdrop"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "pdf.fonttype": 42})


def _load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def _prefill_tps(dst):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from devices.constants import DEVICES
    return DEVICES[dst]["prefill_tps"]


def _usable(dst):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from devices.cost import usable_ctx
    return usable_ctx(dst)


def fig_pareto():
    e1 = _load("e1_pareto.json")
    e3 = _load("e3_costs.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.28))
    ax = axes[0]
    # Relay budget-sweep curve
    sweep = sorted(((int(b), v) for b, v in e1["sweep"].items()),
                   key=lambda x: x[0])
    xs = [v["bytes_mean"] / 1024 for _, v in sweep]
    ys = [v["recall"][0] for _, v in sweep]
    ylo = [v["recall"][1] for _, v in sweep]
    yhi = [v["recall"][2] for _, v in sweep]
    c, m, lab = STYLE["relay"]
    ax.plot(xs, ys, "-", color=c, marker=m, ms=3.5, lw=1.3, zorder=5,
            label="Relay (budget sweep)")
    ax.fill_between(xs, ylo, yhi, color=c, alpha=0.15, lw=0)
    for pol in ORDER:
        if pol == "relay":
            continue
        c, m, lab = STYLE[pol]
        # one point per device pair; annotate the go-run (watch) point
        pts = e1["pairs"][pol]
        bx = [v["bytes_mean"] / 1024 for v in pts.values()]
        by = [v["recall"][0] for v in pts.values()]
        ax.scatter(bx, by, color=c, marker=m, s=16, label=lab, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("shipped bytes (KB, gzip, log)")
    ax.set_ylabel("future-relevance recall")
    ax.set_title("(a) fidelity vs. shipped bytes")
    ax.grid(True, axis="y")
    ax.set_ylim(0.25, 1.06)

    # ---- (b) fidelity vs. user-perceived stall on the watch ----
    ax = axes[1]
    tps = _prefill_tps("watch")
    sx = [min(v["tokens_mean"], _usable("watch")) / tps for _, v in sweep]
    ax.plot(sx, ys, "-", color=STYLE["relay"][0], marker=STYLE["relay"][1],
            ms=4, lw=1.4, zorder=5, label="Relay (budget sweep)")
    ax.fill_between(sx, ylo, yhi, color=STYLE["relay"][0], alpha=0.15, lw=0)
    for pol in ORDER:
        if pol == "relay":
            continue
        c, m, lab = STYLE[pol]
        v = e1["pairs"][pol]["go-run"]
        x = min(v["tokens_mean"], _usable("watch")) / tps
        ax.scatter([x], [v["recall"][0]], color=c, marker=m, s=18, zorder=4)
        if pol in ("recency", "full", "relay-lite", "trunc25", "restart"):
            dx, dy = {"full": (-30, -8), "recency": (-34, 5),
                      "relay-lite": (-30, -9), "trunc25": (3, -3),
                      "restart": (3, -2)}[pol]
            ax.annotate(lab, (x, v["recall"][0]), fontsize=5.0,
                        xytext=(dx, dy), textcoords="offset points")
    ax.set_xlabel("target re-prefill stall (s)")
    ax.set_title("(b) fidelity vs. handoff stall")
    ax.set_ylim(0.25, 1.06)
    ax.grid(True, axis="y")

    ax = axes[2]
    xs = np.arange(len(ORDER))
    for i, pol in enumerate(ORDER):
        c, m, lab = STYLE[pol]
        v = e1["synth"][pol]["probe_recall"]
        ax.bar(i, v[0], color=c, width=0.68, zorder=3)
        ax.errorbar(i, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]],
                    fmt="none", ecolor="#333333", lw=0.8, capsize=1.5,
                    zorder=4)
    SHORT = {"full": "full ctx", "trunc25": "last-25%",
             "recency": "recency", "typed-recency": "typed-rec",
             "extractive": "TextRank",
             "relay": "Relay", "relay-lite": "Relay-lite",
             "restart": "restart", "randdrop": "rand-drop"}
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[p] for p in ORDER], fontsize=5.2,
                       rotation=40, ha="right")
    ax.set_ylabel("probe recall (exact GT)", fontsize=7)
    ax.set_title("(c) synthetic probe recall")
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis="y")

    handles = [Line2D([], [], color=STYLE["relay"][0],
                      marker=STYLE["relay"][1], ms=3.5, lw=1.3,
                      label="Relay (budget sweep, 128\u20138k tok)")]
    handles += [Line2D([], [], color=STYLE[p][0], marker=STYLE[p][1],
                       ms=3.5, lw=0, label=STYLE[p][2])
                for p in ORDER if p != "relay"]
    fig.subplots_adjust(left=0.062, right=0.995, top=0.90, bottom=0.31,
                        wspace=0.30)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=5.2, handlelength=1.4, handletextpad=0.4,
               columnspacing=1.2, labelspacing=0.25,
               bbox_to_anchor=(0.5, 0.005))
    fig.savefig(os.path.join(FIG, "fig_pareto.pdf"))
    plt.close(fig)


def fig_position():
    """E2 (position) and E7 (scaling) share one single-column float."""
    e2 = _load("e2_position.json")
    e7 = _load("e7_scaling.json")
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.12))
    ax = axes[0]
    for pol in ORDER:
        c, m, lab = STYLE[pol]
        bins = sorted(((float(b), v) for b, v in e2[pol].items()),
                      key=lambda x: x[0])
        xs = [b for b, _ in bins]
        ys = [v["recall"][0] for _, v in bins]
        lo = [v["recall"][1] for _, v in bins]
        hi = [v["recall"][2] for _, v in bins]
        ax.plot(xs, ys, "-", color=c, marker=m, ms=3.5, lw=1.2, label=lab)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.12, lw=0)
    ax.set_xlabel("handoff position $k/N$", fontsize=6.5)
    ax.set_ylabel("recall", fontsize=7)
    ax.set_xticks([0.125, 0.375, 0.625, 0.875])
    ax.set_xticklabels(["0–.25", ".25–.5", ".5–.75", ".75–1"],
                       fontsize=5.4, rotation=40, ha="right")
    ax.set_title("(a) recall vs. position", fontsize=7.5)
    ax.grid(True, axis="y")

    ax = axes[1]
    pts = e7["points"]
    x = np.array([p["prefix_tokens"] for p in pts], float)
    for key, col, ls, lab in (("full_bytes", "#0072B2", "-", "full ctx"),
                              ("relay_bytes", "#8C6D31", "--",
                               "Relay unbudgeted"),
                              ("relay_watch_bytes", "#D55E00", "-",
                               "Relay @ watch budget")):
        y = np.array([p[key] for p in pts], float) / 1024.0
        ax.scatter(x, y, s=1.6, color=col, alpha=0.28, lw=0)
        o = np.argsort(x)
        k, b = np.polyfit(np.log(x), np.log(y), 1)
        ax.plot(x[o], np.exp(b) * x[o] ** k, color=col, lw=1.3, ls=ls,
                label=f"{lab} ({k:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("prefix tokens", fontsize=6.5)
    ax.set_ylabel("KB", fontsize=7)
    ax.set_title("(b) size vs. prefix length", fontsize=7.5)
    ax.grid(True, which="major")
    hb, lb = ax.get_legend_handles_labels()
    ha = [Line2D([], [], color=STYLE[p][0], marker=STYLE[p][1], ms=3.0,
                 lw=1.0, label=STYLE[p][2]) for p in ORDER]
    fig.subplots_adjust(left=0.115, right=0.985, top=0.90, bottom=0.40,
                        wspace=0.42)
    fig.legend(handles=ha + hb, loc="lower center", ncol=3, frameon=False,
               fontsize=4.8, handlelength=1.3, handletextpad=0.35,
               columnspacing=0.9, labelspacing=0.22,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(FIG, "fig_position.pdf"))
    plt.close(fig)


def fig_schema():
    """Fig. 2: the typed state block schema (drawn, not data-driven)."""
    fig, ax = plt.subplots(figsize=(3.4, 1.95))
    ax.axis("off")
    rows = [
        ("GOAL", "task + extracted constraints", "#0072B2", "always shipped"),
        ("SIDE\\_FX", "ordered ledger: fingerprint = h(tool, args),\n"
         "status, idempotency key", "#D55E00", "always shipped"),
        ("TOOL\\_DIGEST", "per call: args-hash, summary $\\leq$ s chars,\n"
         "pointer to full result", "#009E73", "budget-filled"),
        ("ENTITIES", "anchor $\\to$ context snippet,\n"
         "salience = tier(tool) $\\times$ recency $\\times$ freq",
         "#E69F00", "budget-filled"),
        ("PLAN", "most recent explicit next-step text", "#56B4E9",
         "shrinks under pressure"),
        ("TAIL", "last $m$ raw steps (bounded)", "#CC79A7",
         "shrinks under pressure"),
    ]
    y = 1.0
    h = 0.155
    for name, desc, color, note in rows:
        ax.add_patch(plt.Rectangle((0.0, y - h), 0.985, h - 0.015,
                                   fc=color, alpha=0.14, ec=color, lw=1.2))
        ax.text(0.015, y - h / 2 + 0.012, name.replace("\\_", "_"),
                fontsize=7.5, fontweight="bold", va="center")
        ax.text(0.015, y - h / 2 - 0.038, note, fontsize=5.4,
                va="center", style="italic", color="#555555")
        ax.text(0.315, y - h / 2 - 0.007,
                desc.replace("\\leq", "≤").replace("$", "")
                    .replace("\\to", "→").replace("\\times", "×"),
                fontsize=6.2, va="center")
        y -= h
    ax.set_xlim(0, 1)
    ax.set_ylim(y - 0.01, 1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_schema.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_safety():
    """E4 + E8: the two error directions of a side-effect ledger."""
    e4 = _load("e4_safety.json")
    e8 = _load("e8_ledger.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45),
                             gridspec_kw={"width_ratios": [1.5, 0.95, 1.15]})

    ax = axes[0]
    pols = [p for p in ORDER if p in e4["real"]]
    xs = np.arange(len(pols))
    for i, p in enumerate(pols):
        c = STYLE[p][0]
        r = e4["real"][p]["rate"]
        ax.bar(i - 0.19, r[0], width=0.36, color=c, zorder=3)
        ax.errorbar(i - 0.19, r[0], yerr=[[r[0] - r[1]], [r[2] - r[0]]],
                    fmt="none", ecolor="#333333", lw=0.7, capsize=1.3,
                    zorder=4)
        ax.bar(i + 0.19, e4["synth"][p]["rate"][0], width=0.36, color=c,
               alpha=0.4, hatch="///", edgecolor=c, lw=0, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([STYLE[p][2] for p in pols], rotation=38, ha="right",
                       fontsize=5.0)
    ax.set_ylabel("exposure rate", fontsize=6.5)
    ax.set_title("(a) double execution the ledger removes", fontsize=6.8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y")

    ax = axes[1]
    fp = e8["E8a_fingerprint_quality"]
    fs = e8["E8b_false_suppression"]
    keys = ["v1-head", "v2-action"]
    labs = ["head hash\n(v1)", "action sig.\n(v2, ours)"]
    series = [("#B22222", lambda k: fp[k]["collision_rate"]),
              ("#E69F00", lambda k: fs[k]["rate"]),
              ("#0072B2", lambda k: fp[k]["divergence_rate"])]
    w = 0.26
    for si, (col, get) in enumerate(series):
        for j, k in enumerate(keys):
            v = get(k)
            ax.bar(j + (si - 1) * w, v, width=w, color=col, zorder=3)
            ax.text(j + (si - 1) * w, v + 0.014, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=4.4, color="#333333", zorder=4)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, fontsize=5.2)
    ax.set_ylabel("rate", fontsize=6.5)
    ax.set_title("(b) the hazard a ledger creates", fontsize=6.8)
    ax.set_ylim(0, 0.62)
    ax.grid(True, axis="y")

    ax = axes[2]
    sat = e8["E8c_saturation"]
    xs = np.arange(len(sat))
    ax.bar(xs - 0.19, [r["ledger_frac_budget"] for r in sat], width=0.36,
           color="#009E73", zorder=3)
    ax.bar(xs + 0.19, [r["recall"] for r in sat], width=0.36, color="#D55E00",
           zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["sfx_bucket"] for r in sat], fontsize=5.2,
                       rotation=38, ha="right")
    ax.set_xlabel("recorded side-effects at the handoff", fontsize=5.8,
                  labelpad=1)
    ax.set_title("(c) what the mandatory ledger costs", fontsize=6.8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y")

    handles = [Patch(fc="#666666", label="(a) real traces"),
               Patch(fc="#666666", alpha=0.4, hatch="///",
                     label="(a) synthetic, exact GT"),
               Patch(fc="#B22222", label="(b) collision"),
               Patch(fc="#E69F00", label="(b) false suppression"),
               Patch(fc="#0072B2", label="(b) divergence"),
               Patch(fc="#009E73", label="(c) ledger share of budget"),
               Patch(fc="#D55E00", label="(c) Relay recall")]
    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.34,
                        wspace=0.34)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=5.2, handlelength=1.1, handletextpad=0.4,
               columnspacing=1.1, labelspacing=0.25,
               bbox_to_anchor=(0.5, -0.01))
    fig.savefig(os.path.join(FIG, "fig_safety.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["pareto", "position", "safety"]
    for w in which:
        try:
            globals()[f"fig_{w}"]()
            print(f"fig_{w} done")
        except (FileNotFoundError, KeyError) as e:
            print(f"fig_{w} skipped ({e})")

