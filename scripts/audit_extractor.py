"""30-trace manual precision audit of the future-relevance anchor extractor.

Protocol (reported verbatim in the paper as author annotation):
  1. Sample 30 trajectories uniformly (seed 3), one handoff position k each,
     drawn uniformly from steps 2..N-2 (seed 7) -- the same seeds and ranges
     the experiments use.
  2. For each (trajectory, k), list every extracted future-relevant anchor
     with (a) a prefix context window around its first occurrence and (b) a
     suffix context window around its re-reference. Cap at 8 anchors per
     trajectory by deterministic stride subsampling to keep the audit
     tractable; record the cap in the CSV.
  3. The author inspects each anchor-in-context and judges it TP or FP:
       TP = the string denotes a genuine task-level entity, quantity,
            artifact, or answer, AND the suffix occurrence re-references the
            same referent (the future actually used it again).
       FP = scaffold/chrome vocabulary, extraction artifact (truncated or
            mis-segmented span), or an accidental string collision (the
            suffix occurrence is a different referent).
  4. Precision = TP / (TP + FP), reported overall and per anchor type,
     together with this protocol, in the paper. Judgments are stored in
     results/anchor_audit.csv (columns: tid, k, anchor, type, prefix_ctx,
     suffix_ctx, judgment, note).

Usage:
  python3 scripts/audit_extractor.py dump   -> writes results/anchor_audit_todo.csv
                                              (judgment column empty) and a
                                              readable .txt for inspection
  python3 scripts/audit_extractor.py score  -> reads results/anchor_audit.csv
                                              (judged) and prints precision
"""
import csv
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from traces.parser import load_trajectories            # noqa: E402
from traces.anchors import AnchorIndex, norm_text, _boundary_pattern  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
CAP_PER_TRACE = 8
CTX = 90


def ctx_window(norm_hay: str, anchor: str, width: int = CTX) -> str:
    m = _boundary_pattern(anchor).search(norm_hay)
    if not m:
        return "(no match?)"
    a, b = max(0, m.start() - width), min(len(norm_hay), m.end() + width)
    return ("..." + norm_hay[a:b] + "...").replace("\n", " ")


def dump(holdout: bool = False):
    """holdout=True: final-report sample -- seed 11, drawn from trajectories
    NOT in the seed-3 tuning sample (filters were iterated on the tuning
    sample; reported precision comes from this disjoint sample)."""
    trajs = load_trajectories()
    idx = AnchorIndex(trajs)
    tuning = {t.tid for t in random.Random(3).sample(trajs, 30)}
    if holdout:
        pool = [t for t in trajs if t.tid not in tuning]
        sample = random.Random(11).sample(pool, 30)
        rng = random.Random(13)
    else:
        sample = random.Random(3).sample(trajs, 30)
        rng = random.Random(7)
    rows = []
    for t in sample:
        k = rng.randrange(2, max(3, t.n - 2))
        A = idx.future_relevant(t, k)
        items = sorted(A.items())
        if len(items) > CAP_PER_TRACE:
            stride = len(items) / CAP_PER_TRACE
            items = [items[int(i * stride)] for i in range(CAP_PER_TRACE)]
        pre_n = norm_text(t.prefix_text(k))
        suf_n = norm_text(t.suffix_text(k))
        for a, ty in items:
            rows.append({
                "tid": t.tid, "k": k, "n_steps": t.n, "anchor": a, "type": ty,
                "prefix_ctx": ctx_window(pre_n, a),
                "suffix_ctx": ctx_window(suf_n, a),
                "judgment": "", "note": ""})
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "anchor_audit_todo.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RES, "anchor_audit_readable.txt"), "w") as f:
        cur = None
        for i, r in enumerate(rows):
            if r["tid"] != cur:
                cur = r["tid"]
                f.write(f"\n===== {r['tid']} (k={r['k']}/{r['n_steps']}) =====\n")
            f.write(f"[{i}] ({r['type']}) `{r['anchor']}`\n")
            f.write(f"    PRE: {r['prefix_ctx']}\n")
            f.write(f"    SUF: {r['suffix_ctx']}\n")
    print(f"dumped {len(rows)} anchor judgments to audit "
          f"({len(sample)} traces); inspect anchor_audit_readable.txt, "
          f"fill judgment column, save as anchor_audit.csv")


def score():
    path = os.path.join(RES, "anchor_audit.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    from collections import Counter
    tot, tp = Counter(), Counter()
    for r in rows:
        j = r["judgment"].strip().upper()
        if j not in ("TP", "FP"):
            raise SystemExit(f"unjudged row: {r['tid']} {r['anchor']}")
        tot[r["type"]] += 1
        tot["__all__"] += 1
        if j == "TP":
            tp[r["type"]] += 1
            tp["__all__"] += 1
    out = {ty: {"n": tot[ty], "tp": tp[ty],
                "precision": round(tp[ty] / tot[ty], 4)}
           for ty in tot}
    import json
    with open(os.path.join(RES, "anchor_audit_summary.json"), "w") as f:
        json.dump({"n_traces": len({r['tid'] for r in rows}),
                   "cap_per_trace": CAP_PER_TRACE,
                   "per_type": out}, f, indent=1)
    for ty, d in sorted(out.items()):
        print(f"{ty:10s} n={d['n']:4d} precision={d['precision']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        score()
    else:
        dump(holdout="holdout" in sys.argv)
