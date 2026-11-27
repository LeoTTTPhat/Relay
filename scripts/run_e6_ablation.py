"""E6: leave-one-out ablation of Relay schema components.

Two kinds of ablation are reported together, because they answer different
questions.  *Component* ablations remove one schema section at a time
({entities, tool_digest, plan, tail, side_fx, constraints}) and ask what
that section carries.  *Mechanism* ablations keep the whole schema but
disable one design decision -- step-grouped entity encoding, and
value-density budget allocation -- and ask what the distiller's two
non-obvious choices are worth.  Each is scored on delta future-relevance
recall (real, all + strict anchors), delta probe recall (synthetic, exact),
and delta double-execution hazards (synthetic exact + real
side-effect-like).
"""
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import (load_all, sample_positions, real_fidelity, probe_recall,
                    hazards_real, hazards_synth, boot_ci, RES)
from state.schema import Distiller
from policies.policies import p4_relay

SEEDS = [0, 1, 2, 3, 4]
COMPONENTS = ["none", "entities", "tool_digest", "plan", "tail", "side_fx",
              "constraints"]
MECHANISMS = ["flat_entities", "fixed_order"]
ALL = COMPONENTS + MECHANISMS
DST = "watch"


def _dists(idx, idx_s):
    """One distiller per configuration under test."""
    return ({"default": Distiller(idx),
             "flat_entities": Distiller(idx, group_entities=False),
             "fixed_order": Distiller(idx, allocator="fixed")},
            {"default": Distiller(idx_s),
             "flat_entities": Distiller(idx_s, group_entities=False),
             "fixed_order": Distiller(idx_s, allocator="fixed")})


def _cfg(c):
    """(distiller key, dropped components) for an ablation label."""
    if c in MECHANISMS:
        return c, frozenset()
    return "default", (frozenset() if c == "none" else frozenset([c]))


def run():
    t0 = time.time()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    DR, DS = _dists(idx, idx_s)
    out = {"dst": DST, "components": {}}
    acc = {c: {"recall": [], "strict": [], "probe": [],
               "haz_r": [0, 0], "haz_s": [0, 0]} for c in ALL}

    for traj in trajs:
        per = defaultdict(list)
        per_s = defaultdict(list)
        for seed in SEEDS:
            for k in sample_positions(traj.n, traj.tid, seed, 1):
                A = idx.future_relevant(traj, k)
                for c in ALL:
                    dk, drop = _cfg(c)
                    o = p4_relay(traj, k, DST, distiller=DR[dk], drop=drop)
                    r, rs, _ = real_fidelity(idx, traj, k, o.resume_text,
                                             anchors=A)
                    per[c].append(r)
                    per_s[c].append(rs)
                    h, tot = hazards_real(traj, k, o.resume_text)
                    acc[c]["haz_r"][0] += h
                    acc[c]["haz_r"][1] += tot
        for c in ALL:
            acc[c]["recall"].append(sum(per[c]) / len(per[c]))
            acc[c]["strict"].append(sum(per_s[c]) / len(per_s[c]))

    for ei, ep in enumerate(eps):
        st = synth_trajs[ei]
        per = defaultdict(list)
        for seed in SEEDS:
            for k in sample_positions(ep.n, ep.eid, seed, 1):
                for c in ALL:
                    dk, drop = _cfg(c)
                    o = p4_relay(st, k, DST, distiller=DS[dk], drop=drop)
                    pr, _ = probe_recall(ep, k, o.resume_text)
                    per[c].append(pr)
                    h, tot = hazards_synth(ep, k, o.resume_text)
                    acc[c]["haz_s"][0] += h
                    acc[c]["haz_s"][1] += tot
        for c in ALL:
            acc[c]["probe"].append(sum(per[c]) / len(per[c]))

    base = {}
    for c in ALL:
        m, lo, hi = boot_ci(acc[c]["recall"])
        ms, _, _ = boot_ci(acc[c]["strict"])
        mp, plo, phi = boot_ci(acc[c]["probe"])
        hr = acc[c]["haz_r"]
        hs = acc[c]["haz_s"]
        entry = {"recall": [m, lo, hi], "recall_strict": ms,
                 "probe": [mp, plo, phi],
                 "haz_real_rate": hr[0] / max(1, hr[1]),
                 "haz_synth_rate": hs[0] / max(1, hs[1])}
        if c == "none":
            base = entry
        entry["delta_recall"] = round(m - base["recall"][0], 4)
        entry["delta_probe"] = round(mp - base["probe"][0], 4)
        entry["delta_haz_synth"] = round(
            entry["haz_synth_rate"] - base["haz_synth_rate"], 4)
        out["components"][c] = entry
    json.dump(out, open(os.path.join(RES, "e6_ablation.json"), "w"), indent=1)
    rt_p = os.path.join(RES, "runtimes.json")
    rt = json.load(open(rt_p)) if os.path.exists(rt_p) else {}
    rt["run_e6"] = round(time.time() - t0, 1)
    json.dump(rt, open(rt_p, "w"), indent=1)
    print(f"e6 done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
