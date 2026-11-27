"""Diagnostic: per-case behaviour of the learned-summariser baseline.

E17 reports a mean, and a mean is not enough to tell a weak baseline from a
broken one. Spot checks on short prefixes produced well-formed 200--290 token
summaries scoring 0.73--1.00, which does not obviously reconcile with the
aggregate. This dumps every case so the distribution can be inspected rather
than guessed at: output length, recall, anchor count and prefix length, plus
the first line of each summary.

Written to ``results/diag_llm_summary.json``; not part of the paper, kept in
the artifact because the conclusion it supports (whether the E17 number is a
property of the baseline or of the harness) is one a reader may want to check.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, sample_positions, real_fidelity, RES   # noqa
from policies.baselines import p11_llm_summary                      # noqa
from policies.policies import relay_budget                          # noqa

OUT = os.path.join(RES, "diag_llm_summary.json")
N = int(os.environ.get("RELAY_E17_N", "72"))
DST = "watch"


def run():
    t0 = time.time()
    trajs, idx, eps, synth, idx_s = load_all()
    b = relay_budget(DST)
    cases = []
    for traj in trajs:
        for k in sample_positions(traj.n, traj.tid, 0, 1):
            cases.append((traj, k))
    cases.sort(key=lambda c: (c[0].tid, c[1]))
    stride = max(1, len(cases) // N)
    cases = cases[::stride][:N]

    import run_e10_llm as E10
    lm = E10.LM("Qwen/Qwen2.5-7B-Instruct")

    rows = []
    for i, (traj, k) in enumerate(cases):
        A = idx.future_relevant(traj, k)
        o = p11_llm_summary(traj, k, DST, lm, budget=b)
        r, rs, _ = real_fidelity(idx, traj, k, o.resume_text, anchors=A)
        rows.append({
            "tid": traj.tid, "k": k,
            "prefix_tokens": traj.prefix_tokens(k),
            "out_tokens": o.tokens_wire,
            "recall": round(r, 4), "n_anchors": len(A),
            "head": o.wire_text[:120].replace("\n", " "),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(cases)} mean_out={np.mean([x['out_tokens'] for x in rows]):.0f} "
                  f"mean_recall={np.mean([x['recall'] for x in rows]):.4f}", flush=True)

    out_tok = np.array([r["out_tokens"] for r in rows], float)
    rec = np.array([r["recall"] for r in rows], float)
    pre = np.array([r["prefix_tokens"] for r in rows], float)
    summary = {
        "n": len(rows),
        "out_tokens": {"mean": float(out_tok.mean()),
                       "median": float(np.median(out_tok)),
                       "min": float(out_tok.min()), "max": float(out_tok.max()),
                       "n_under_20": int((out_tok < 20).sum())},
        "recall": {"mean": float(rec.mean()), "median": float(np.median(rec))},
        "recall_by_prefix_quartile": [],
        "corr_prefix_vs_out_tokens": float(np.corrcoef(pre, out_tok)[0, 1]),
        "runtime_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    qs = np.quantile(pre, [0.25, 0.5, 0.75])
    edges = [pre.min() - 1] + list(qs) + [pre.max() + 1]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (pre > lo) & (pre <= hi)
        if sel.sum():
            summary["recall_by_prefix_quartile"].append({
                "prefix_tokens": f"{lo:.0f}-{hi:.0f}", "n": int(sel.sum()),
                "recall": round(float(rec[sel].mean()), 4),
                "out_tokens": round(float(out_tok[sel].mean()), 1)})
    json.dump(summary, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    run()
