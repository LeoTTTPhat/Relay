"""E4b: record loss measured on identity, not on prose.

E4 counts a record as lost when the normalised *head* of the action's step is
absent from what the target holds. That was the only option before the ledger
had a typed identity, and it is the wrong probe now, for a reason the
budget-saturation path makes obvious: when the mandatory ledger has to shrink,
the distiller drops head prose and keeps the fingerprint and the idempotency
key, because those are what a dispatcher matches on. A head-text probe scores
that as a lost record. A dispatcher would not.

So we report both, on the same handoffs:

  ``head``      the original text-containment probe (E4's definition),
                comparable with the numbers in the previous version;
  ``identity``  is the action's idempotency key present in what the target
                holds -- the property enforcement actually depends on.

For window policies the two are nearly the same, because a raw window either
carries the whole step or none of it. For \\sys{} and \\sys{}-lite they come
apart exactly where the ledger saturates, and the gap is the measure's, not
the mechanism's.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import (load_all, sample_positions, hazards_real, boot_ci,  # noqa
                    policy_budget, RES)
from state.schema import Distiller, build_side_fx                       # noqa
from policies.policies import POLICIES                                  # noqa
from policies.baselines import p7_extractive                            # noqa
from traces.anchors import norm_text                                    # noqa

OUT = os.path.join(RES, "e4b_identity_exposure.json")
SEEDS = [0, 1, 2, 3, 4]
DST = "watch"
POLS = ["full", "trunc25", "recency", "typed-recency", "relay", "relay-lite",
        "restart", "randdrop", "extractive"]


def run():
    import time
    t0 = time.time()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    dist = Distiller(idx)
    agg = {p: {"head_haz": 0, "id_haz": 0, "tot": 0} for p in POLS}
    per_traj = {p: {"head": [], "id": []} for p in POLS}

    for t in trajs:
        seen = {p: [0, 0, 0] for p in POLS}       # head_haz, id_haz, tot
        for seed in SEEDS:
            for k in sample_positions(len(t.steps), t.tid, seed):
                if k >= len(t.steps):
                    continue
                entries = build_side_fx(t, k)
                if not entries:
                    continue
                for p in POLS:
                    if p == "extractive":
                        o = p7_extractive(t, k, DST)
                    else:
                        kw = {"distiller": dist} if p.startswith("relay") else {}
                        o = POLICIES[p](t, k, DST, seed=seed,
                                        budget=policy_budget(p, DST), **kw)
                    h = norm_text(o.resume_text)
                    hh, tot = hazards_real(t, k, o.resume_text)
                    ih = sum(1 for e in entries
                             if e.idem_key.lower() not in h)
                    agg[p]["head_haz"] += hh
                    agg[p]["id_haz"] += ih
                    agg[p]["tot"] += tot
                    seen[p][0] += hh
                    seen[p][1] += ih
                    seen[p][2] += tot
        for p in POLS:
            if seen[p][2]:
                per_traj[p]["head"].append(seen[p][0] / seen[p][2])
                per_traj[p]["id"].append(seen[p][1] / seen[p][2])

    out = {"target": DST, "n_labelled": agg["relay"]["tot"], "per_policy": {}}
    for p in POLS:
        a = agg[p]
        out["per_policy"][p] = {
            "head_rate": [round(v, 4) for v in boot_ci(per_traj[p]["head"])],
            "identity_rate": [round(v, 4) for v in boot_ci(per_traj[p]["id"])],
            "head_haz": a["head_haz"], "id_haz": a["id_haz"], "tot": a["tot"],
        }
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)

    print(f"{'policy':15s} {'head probe':>11s} {'identity':>10s}   (n="
          f"{out['n_labelled']} labelled)")
    for p in POLS:
        d = out["per_policy"][p]
        print(f"{p:15s} {d['head_rate'][0]:11.4f} {d['identity_rate'][0]:10.4f}")
    print("->", OUT)


if __name__ == "__main__":
    run()
