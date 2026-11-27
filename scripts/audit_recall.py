"""Extractor RECALL audit (companion to the precision audit).

The precision audit asks: of the anchors the extractor emits, how many are
genuine future-relevant references?  It cannot answer the complementary
question: how many genuine references does the extractor miss?  A metric
with high precision and low recall would still bias the policy comparison,
because it would score policies only on the anchor types the extractor
happens to catch.

Protocol (author annotation, same style as the precision audit):
  1. sample ``n_traces`` trajectories and one handoff position each
     (fixed seed);
  2. generate candidates with a deliberately *over-generating* extractor --
     no document-frequency cut, no stoplist, no sub-span de-duplication,
     every alphanumeric token, number, quoted span, and capitalised n-gram
     that occurs in the prefix and again in the suffix;
  3. subsample the candidates by deterministic stride to a tractable number
     per trajectory and dump them with prefix/suffix context;
  4. the annotating author marks each candidate ``genuine`` iff it denotes a
     task-level entity, quantity, artifact, or answer whose suffix
     occurrence re-references the same referent -- the *same* criterion the
     precision audit uses;
  5. recall = |genuine and recovered| / |genuine|, where a genuine candidate
     counts as *recovered* if the shipped extractor emits it or emits an
     anchor that contains it -- a candidate ``nature`` is not missed when
     the extractor emitted ``human nature``, because the information the
     future needs is present either way.

``dump`` writes the unjudged CSV, ``score`` reads the judged one.
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, sample_positions, RES
from traces.anchors import (extract_anchors, norm_text, norm_anchor,
                            contains_anchor)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}|\d[\d,\.]*\d|\d")
QUOTE_RE = re.compile(r"[\"'“”']([^\"'“”'\n]{4,60})[\"'“”']")
CAP_RE = re.compile(r"\b([A-Z][\w&\-]+(?:\s+[A-Z][\w&\-]+){0,3})\b")
URLISH_RE = re.compile(r"https?://\S+|\b[\w\-.]+\.(?:com|org|net|gov|edu|io)"
                       r"\b|\b[\w\-]+\.(?:pdf|csv|json|xlsx?|py|txt|zip|md)\b",
                       re.I)


def over_generate(prefix: str, suffix_norm: str, cap: int = 400):
    """High-recall candidate set: everything that recurs, filtered by nothing."""
    cands = {}
    for rx, ty in ((URLISH_RE, "urlish"), (QUOTE_RE, "quoted"),
                   (CAP_RE, "capseq"), (TOKEN_RE, "token")):
        for m in rx.finditer(prefix):
            a = norm_anchor(m.group(1) if rx in (QUOTE_RE, CAP_RE)
                            else m.group(0))
            if not a or len(a) < 2 or a in cands:
                continue
            cands[a] = ty
            if len(cands) >= cap:
                break
    return {a: t for a, t in cands.items() if contains_anchor(suffix_norm, a)}


def dump(n_traces=20, per_trace=12, seed=23,
         path=None):
    trajs, idx, eps, synth, idx_s = load_all()
    import random
    rng = random.Random(seed)
    picks = rng.sample(range(len(trajs)), n_traces)
    rows = []
    for ti in sorted(picks):
        traj = trajs[ti]
        k = sample_positions(traj.n, traj.tid, seed, 1)[0]
        prefix = traj.prefix_text(k)
        pn, sn = norm_text(prefix), norm_text(traj.suffix_text(k))
        cands = over_generate(prefix, sn)
        emitted = {a for a in extract_anchors(prefix) if idx.keep(a)}
        gt = norm_anchor(traj.ground_truth)
        if gt:
            emitted.add(gt)

        def recovered(a):
            return any(a == e or a in e for e in emitted)
        keys = sorted(cands)
        stride = max(1, len(keys) // per_trace)
        for a in keys[::stride][:per_trace]:
            i = pn.find(a)
            j = sn.find(a)
            rows.append({
                "trace": traj.tid, "k": k, "candidate": a,
                "cand_type": cands[a],
                "emitted_by_extractor": int(recovered(a)),
                "prefix_ctx": pn[max(0, i - 70):i + len(a) + 70],
                "suffix_ctx": sn[max(0, j - 70):j + len(a) + 70],
                "judgment": "", "note": ""})
    path = path or os.path.join(RES, "anchor_recall_todo.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"dumped {len(rows)} candidates from {n_traces} trajectories -> {path}")


def score(path=None):
    path = path or os.path.join(RES, "anchor_recall_judged.csv")
    rows = list(csv.DictReader(open(path)))
    judged = [r for r in rows if r["judgment"].strip()]
    gen = [r for r in judged if r["judgment"].strip() == "genuine"]
    hit = [r for r in gen if r["emitted_by_extractor"] == "1"]
    by_type = {}
    for r in gen:
        t = r["cand_type"]
        d = by_type.setdefault(t, [0, 0])
        d[1] += 1
        d[0] += r["emitted_by_extractor"] == "1"
    res = {"n_candidates": len(rows), "n_judged": len(judged),
           "n_genuine": len(gen), "n_recovered": len(hit),
           "recall": round(len(hit) / max(len(gen), 1), 4),
           "per_type": {t: round(v[0] / v[1], 4) for t, v in by_type.items()},
           "per_type_n": {t: v[1] for t, v in by_type.items()}}
    print(res)
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        import json
        r = score()
        with open(os.path.join(RES, "anchor_recall_summary.json"), "w") as f:
            json.dump(r, f, indent=1)
    else:
        dump()
