"""E14b/c: validating the fidelity metric against its sharpest objection.

The objection is precise and correct as stated: \\sys{}'s allocator ranks
candidate items by how many salient anchors they carry, and the headline
metric counts how many salient anchors survive.  The allocator sees only the
prefix, so this is not oracle access to the future -- but method and metric
share a lexical ontology, and a shared prior can flatter a method just as an
oracle can.  Arguing about it is not evidence.  This experiment tries to break
the result three ways.

  **E14c-1 metric decoupling.**  Partition the anchor types into two disjoint
  halves.  Let the allocator count only half $A$; score recall only on half
  $B$.  The allocator is now optimising a quantity that shares no anchor type
  with the metric.  Two strengths are run: ``value-blind`` restricts only the
  value function, ``fully-blind`` also restricts which anchors may be selected
  as entities at all -- the harsher test, since it stops \\sys{} from shipping
  $B$-type anchors deliberately and leaves only what its structure carries
  incidentally.  Both partitions are run in both directions.

  **E14c-2 empty anchor sets.**  Recall is defined as 1.0 when the future
  referenced nothing from the prefix.  If such handoffs are common they inflate
  every mean equally but compress the differences.  We report the fraction and
  re-report every policy conditional on a non-empty set.

  **E14c-3 per-type recall.**  The claim that extractor errors "affect all
  policies alike" is testable: if one policy's advantage lived in a single
  anchor type -- say, page chrome retained by raw windows -- it would show up
  as a type whose ordering differs from the aggregate.

  **E14b out-of-schema state.**  A second synthetic corpus (``synth/oos.py``)
  whose ground truth is deliberately *not* the schema's field list: revised
  goals, contradicted constraints, superseded tool values and explicitly
  deferred questions.  Two numbers are reported, and the second matters more:
  recall of the state that is now *live*, and the rate at which a policy ships
  the *superseded* state without its replacement -- an error that a recall
  metric alone scores as success.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from explib import (load_all, sample_positions, real_fidelity, boot_ci,   # noqa
                    policy_budget, RES)
from stats import compare                                                # noqa
from state.schema import Distiller                                       # noqa
from synth.oos import generate_oos                                       # noqa
from traces.anchors import AnchorIndex, contains_anchor, norm_text       # noqa
from policies.policies import POLICIES, relay_budget                     # noqa

OUT = os.path.join(RES, "e14_independence.json")
SEEDS = [0, 1, 2, 3, 4]
POLS = ["full", "trunc25", "recency", "typed-recency", "relay", "relay-lite",
        "restart", "randdrop"]
DST = "watch"

#: Disjoint partitions of the anchor ontology.  ``answer`` rides with the
#: half it is scored in; it is a single string per trajectory and never drives
#: allocation.
SPLITS = {
    "A=url,file | B=number,entity,answer":
        ({"url", "file"}, {"number", "entity", "answer"}),
    "A=number,entity | B=url,file,answer":
        ({"number", "entity"}, {"url", "file", "answer"}),
}


def _recall_on_types(resume_text, anchors, types):
    """Recall restricted to a subset of anchor types."""
    sel = {a: t for a, t in anchors.items() if t in types}
    if not sel:
        return None
    h = norm_text(resume_text)
    return sum(1 for a in sel if contains_anchor(h, a)) / len(sel)


def e14c1_decoupling(trajs, idx):
    """Relay's allocator optimises A; the metric scores only B."""
    budget = relay_budget(DST)
    out = {}
    for label, (A, B) in SPLITS.items():
        dists = {
            "deployed": Distiller(idx),
            "value-blind": Distiller(idx, value_types=A),
            "fully-blind": Distiller(idx, value_types=A, select_types=A),
        }
        vals = defaultdict(list)
        for t in trajs:
            for seed in SEEDS:
                for k in sample_positions(len(t.steps), t.tid, seed):
                    if k >= len(t.steps):
                        continue
                    anchors = idx.future_relevant(t, k)
                    if not any(ty in B for ty in anchors.values()):
                        continue
                    for name in ("recency", "typed-recency", "full",
                                 "randdrop"):
                        o = POLICIES[name](t, k, DST, seed=seed, budget=budget)
                        r = _recall_on_types(o.resume_text, anchors, B)
                        if r is not None:
                            vals[name].append(r)
                    for dname, dist in dists.items():
                        o = POLICIES["relay"](t, k, DST, distiller=dist,
                                              budget=budget)
                        r = _recall_on_types(o.resume_text, anchors, B)
                        if r is not None:
                            vals[f"relay/{dname}"].append(r)
        res = {}
        for kk, v in vals.items():
            m, lo, hi = boot_ci(v)
            res[kk] = {"mean": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
                       "n": len(v)}
        # does the blinded allocator still beat the strongest baseline?
        for variant in ("deployed", "value-blind", "fully-blind"):
            a = vals.get(f"relay/{variant}")
            if not a:
                continue
            res[f"paired:{variant}"] = compare(
                a, {base: vals[base] for base in ("recency", "typed-recency")
                    if vals.get(base)})
        out[label] = res
    return out


def e14c2_empty_sets(trajs, idx, dist):
    """How many handoffs have an empty future-relevant set, and what changes?"""
    budget = relay_budget(DST)
    empty = 0
    tot = 0
    all_v, nonempty_v = defaultdict(list), defaultdict(list)
    sizes = []
    for t in trajs:
        for seed in SEEDS:
            for k in sample_positions(len(t.steps), t.tid, seed):
                if k >= len(t.steps):
                    continue
                A = idx.future_relevant(t, k)
                tot += 1
                sizes.append(len(A))
                if not A:
                    empty += 1
                for p in POLS:
                    kw = {"distiller": dist} if p.startswith("relay") else {}
                    o = POLICIES[p](t, k, DST, seed=seed,
                                    budget=policy_budget(p, DST), **kw)
                    r, _, _ = real_fidelity(idx, t, k, o.resume_text, anchors=A)
                    all_v[p].append(r)
                    if A:
                        nonempty_v[p].append(r)
    res = {"n_handoffs": tot, "n_empty": empty,
           "empty_fraction": round(empty / tot, 4) if tot else 0.0,
           "mean_anchors": round(float(np.mean(sizes)), 2),
           "median_anchors": float(np.median(sizes))}
    res["per_policy"] = {
        p: {"all": round(float(np.mean(all_v[p])), 4),
            "nonempty_only": round(float(np.mean(nonempty_v[p])), 4)
            if nonempty_v[p] else None,
            "ci_nonempty": [round(x, 4) for x in boot_ci(nonempty_v[p])]
            if nonempty_v[p] else None}
        for p in POLS}
    return res


def e14c3_per_type(trajs, idx, dist):
    """Recall by anchor type: does any policy's edge live in one type?"""
    budget = relay_budget(DST)
    per = {p: defaultdict(list) for p in POLS}
    for t in trajs:
        for seed in SEEDS:
            for k in sample_positions(len(t.steps), t.tid, seed):
                if k >= len(t.steps):
                    continue
                A = idx.future_relevant(t, k)
                if not A:
                    continue
                types = set(A.values())
                for p in POLS:
                    kw = {"distiller": dist} if p.startswith("relay") else {}
                    o = POLICIES[p](t, k, DST, seed=seed,
                                    budget=policy_budget(p, DST), **kw)
                    for ty in types:
                        r = _recall_on_types(o.resume_text, A, {ty})
                        if r is not None:
                            per[p][ty].append(r)
    return {p: {ty: {"mean": round(float(np.mean(v)), 4), "n": len(v)}
                for ty, v in sorted(d.items())}
            for p, d in per.items()}


def e14b_out_of_schema(seed=7, eps=None):
    """Live-state recall and superseded-state error on the OOS corpus.

    ``eps`` lets E25 score a re-worded copy of the same corpus with exactly
    this code; ``None`` generates the template corpus as before.
    """
    if eps is None:
        eps = generate_oos(120, seed=seed)
    trajs = [e.to_trajectory() for e in eps]
    idx = AnchorIndex.__new__(AnchorIndex)
    idx.df, idx.n_trajs, idx.df_cut = {}, len(trajs), len(trajs)
    dist = Distiller(idx)
    budget = relay_budget(DST)

    live = defaultdict(list)
    per_field = {p: defaultdict(list) for p in POLS}
    stale_only = defaultdict(list)      # superseded shipped, replacement absent
    for e, traj in zip(eps, trajs):
        for s in SEEDS:
            for k in sample_positions(e.n, e.eid, s):
                if k >= e.n:
                    continue
                probes, negs = e.oos_probes(k), e.oos_negatives(k)
                if not probes:
                    continue
                for p in POLS:
                    kw = {"distiller": dist} if p.startswith("relay") else {}
                    o = POLICIES[p](traj, k, DST, seed=s,
                                    budget=policy_budget(p, DST), **kw)
                    h = norm_text(o.resume_text)
                    hits = tot = 0
                    present = {}
                    for f, targets in probes.items():
                        ok = all(norm_text(str(x)) in h for x in targets if x)
                        present[f] = ok
                        per_field[p][f].append(float(ok))
                        hits += ok
                        tot += 1
                    live[p].append(hits / tot if tot else 1.0)
                    # superseded state present while its replacement is not
                    pairs = [("stale_goal", "revised_goal"),
                             ("stale_constraint", "active_constraint"),
                             ("stale_value", "corrected_value")]
                    bad = n = 0
                    for sname, lname in pairs:
                        if sname not in negs or lname not in probes:
                            continue
                        st = all(norm_text(str(x)) in h
                                 for x in negs[sname] if x)
                        n += 1
                        bad += int(st and not present.get(lname, False))
                    if n:
                        stale_only[p].append(bad / n)
    out = {"n_episodes": len(eps),
           "live_state_recall": {
               p: {"mean": round(float(np.mean(live[p])), 4),
                   "ci": [round(x, 4) for x in boot_ci(live[p])],
                   "n": len(live[p])} for p in POLS},
           "superseded_shipped_alone": {
               p: round(float(np.mean(stale_only[p])), 4) if stale_only[p]
               else None for p in POLS},
           "per_field": {p: {f: round(float(np.mean(v)), 4)
                             for f, v in sorted(d.items())}
                         for p, d in per_field.items()}}
    out["paired_vs_relay"] = compare(
        live["relay"], {b: live[b] for b in ("recency", "typed-recency", "full")})
    return out


def run():
    import time
    t0 = time.time()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    dist = Distiller(idx)
    out = {}
    out["e14b_out_of_schema"] = e14b_out_of_schema()
    print("E14b live recall:",
          {p: v["mean"] for p, v in out["e14b_out_of_schema"]["live_state_recall"].items()},
          flush=True)
    print("E14b superseded-alone:",
          out["e14b_out_of_schema"]["superseded_shipped_alone"], flush=True)
    out["e14c2_empty_sets"] = e14c2_empty_sets(trajs, idx, dist)
    print("E14c2:", {k: v for k, v in out["e14c2_empty_sets"].items()
                     if k != "per_policy"}, flush=True)
    out["e14c1_decoupling"] = e14c1_decoupling(trajs, idx)
    print("E14c1 done", flush=True)
    out["e14c3_per_type"] = e14c3_per_type(trajs, idx, dist)
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)

    for label, res in out["e14c1_decoupling"].items():
        print(f"\n{label}")
        for k, v in res.items():
            if k.startswith("paired"):
                for r in v:
                    print(f"   {k} vs {r['vs']:14s} delta={r['delta']:+.4f} "
                          f"p={r['p_holm']:.2g} cliff={r['cliffs_delta']:+.3f} "
                          f"({r['effect']})")
            else:
                print(f"   {k:24s} {v['mean']:.4f}  n={v['n']}")
    print("\nper-type recall (mean):")
    for p, d in out["e14c3_per_type"].items():
        print(f"  {p:14s} " + "  ".join(f"{ty}={v['mean']:.3f}"
                                        for ty, v in d.items()))
    print("->", OUT)


if __name__ == "__main__":
    run()
