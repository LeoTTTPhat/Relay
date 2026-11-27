"""E8: the ledger, measured in both directions.

E4 measures the hazard the ledger is designed to remove (double execution).
A ledger also *creates* a hazard: if two genuinely different actions share a
fingerprint, the resumed agent consults the ledger, finds a hit, and skips an
action it should have performed -- *false suppression*.  A safety mechanism
evaluated only in the direction it was built to win has not been evaluated,
so E8 reports:

  E8a  fingerprint quality with exact ground truth (synthetic episodes, whose
       ``action_id`` field states which ledger entries denote the same world
       action): collision rate (-> false suppression) and divergence rate
       (-> missed suppression), for the head-hash (v1) and the action-
       signature (v2) fingerprints;
  E8b  end-to-end false-suppression rate under the resumption contract: for
       every action the real suffix goes on to execute, does it hit a ledger
       entry that denotes a *different* action?
  E8c  ledger saturation: how much of a wearable budget the mandatory ledger
       consumes as side-effect counts grow, when it starts crowding out the
       rest of the block, and what that costs in fidelity;
  E8d  a real-trace collision sample dumped for author annotation
       (``results/e8_collisions_todo.csv``), since real traces carry no
       action-identity ground truth.
"""
import csv
import json
import os
import time
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, sample_positions, real_fidelity, boot_ci, RES
from state.fingerprint import fp_action, fp_head
from state.schema import Distiller, build_side_fx
from policies.policies import POLICIES, relay_budget, usable
from traces.parser import exact_tokens_cached
from traces.anchors import norm_text

FP = {"v1-head": fp_head, "v2-action": fp_action}


def e8a_fingerprint_quality(eps):
    """Exact pairwise scoring on synthetic episodes."""
    out = {}
    for name, fn in FP.items():
        coll = coll_tot = div = div_tot = 0
        for e in eps:
            ses = [(s, st) for s, st in zip(e.steps, range(e.n))
                   if s.side_effect_like]
            final = e.states[-1].side_effects
            ids = {se["key"]: se["action_id"] for se in final}
            recs = []
            for s in e.steps:
                if not s.side_effect_like:
                    continue
                key = None
                for se in final:
                    if se["key"] in s.content:
                        key = se["key"]
                        break
                if key is None:
                    continue
                recs.append((ids[key], fn(s.agent, s.content)))
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    same_id = recs[i][0] == recs[j][0]
                    same_fp = recs[i][1] == recs[j][1]
                    if same_id:
                        div_tot += 1
                        div += (not same_fp)
                    else:
                        coll_tot += 1
                        coll += same_fp
        out[name] = {
            "collision_rate": coll / max(coll_tot, 1), "collision_n": coll,
            "collision_pairs": coll_tot,
            "divergence_rate": div / max(div_tot, 1), "divergence_n": div,
            "divergence_pairs": div_tot}
    return out


def e8b_false_suppression(eps, dist, seeds=(0, 1, 2, 3, 4)):
    """Resumption contract: does a *future* action hit a ledger entry that
    denotes a different action?"""
    out = {}
    for name, fn in FP.items():
        fs = tot = 0
        per_ep = []
        for e in eps:
            traj = e.to_trajectory()
            final = e.states[-1].side_effects
            ep_fs = ep_tot = 0
            for seed in seeds:
                for k in sample_positions(traj.n, traj.tid, seed, 2):
                    ledger = build_side_fx(traj, k)
                    ids_in_prefix = set()
                    for s in traj.steps[:k + 1]:
                        for se in final:
                            if se["key"] in s.content:
                                ids_in_prefix.add(se["action_id"])
                    fps = {}
                    for s in traj.steps[:k + 1]:
                        if not s.side_effect_like:
                            continue
                        aid = next((se["action_id"] for se in final
                                    if se["key"] in s.content), None)
                        if aid is not None:
                            fps.setdefault(fn(s.agent, s.content), aid)
                    for s in traj.steps[k + 1:]:
                        if not s.side_effect_like:
                            continue
                        aid = next((se["action_id"] for se in final
                                    if se["key"] in s.content), None)
                        if aid is None or aid in ids_in_prefix:
                            continue     # a genuine repeat: suppression is
                        ep_tot += 1      # correct there, not a false one
                        f = fn(s.agent, s.content)
                        if f in fps and fps[f] != aid:
                            ep_fs += 1
            fs += ep_fs
            tot += ep_tot
            if ep_tot:
                per_ep.append(ep_fs / ep_tot)
        m, lo, hi = boot_ci(per_ep)
        out[name] = {"rate": fs / max(tot, 1), "n": fs, "tot": tot,
                     "ci": [m, lo, hi]}
    return out


def e8c_saturation(trajs, eps, idx, idx_s, dist_r, dist_s):
    """How much of the budget the mandatory ledger takes, and what it costs."""
    rows = []
    per_bucket = defaultdict(lambda: {"frac": [], "recall": [], "n": 0,
                                      "clipped": 0, "over": 0})
    for traj, dist, index, kind in ((t, dist_r, idx, "real") for t in trajs):
        for k in sample_positions(traj.n, traj.tid, 0, 2):
            led = build_side_fx(traj, k)
            if not led:
                continue
            budget = relay_budget("watch")
            ledger_txt = "\n".join(
                f"[{e.idem_key}|{e.fingerprint}|{e.status}] step {e.step} "
                f"{e.agent}: {e.head}" for e in led)
            ltok = exact_tokens_cached(ledger_txt)
            nb = min(5, len(led) // 5)
            b = per_bucket[nb]
            b["n"] += 1
            b["frac"].append(min(1.0, ltok / budget))
            b["clipped"] += ltok > budget * 0.5
            b["over"] += ltok > budget
            A = index.future_relevant(traj, k)
            o = POLICIES["relay"](traj, k, "watch", distiller=dist)
            r, _, _ = real_fidelity(index, traj, k, o.resume_text, anchors=A)
            b["recall"].append(r)
    for nb in sorted(per_bucket):
        b = per_bucket[nb]
        rows.append({
            "sfx_bucket": f"{nb*5}-{nb*5+4}" if nb < 5 else "25+",
            "n": b["n"],
            "ledger_frac_budget": sum(b["frac"]) / len(b["frac"]),
            "frac_over_half": b["clipped"] / b["n"],
            "frac_over_budget": b["over"] / b["n"],
            "recall": sum(b["recall"]) / len(b["recall"])})
    return rows


def e8d_collision_sample(trajs, limit=60, seed=11):
    """Dump real-trace fingerprint collisions for author annotation."""
    import random
    rng = random.Random(seed)
    cand = []
    for t in trajs:
        for name, fn in FP.items():
            seen = {}
            for s in t.steps:
                if not s.side_effect_like:
                    continue
                f = fn(s.agent, s.content)
                nb = norm_text(s.content)
                if f in seen and seen[f][0] != nb:
                    cand.append({"fp_version": name, "trace": t.tid,
                                 "agent": s.agent,
                                 "step_a": seen[f][1], "step_b": s.idx,
                                 "text_a": seen[f][2][:400].replace("\n", " "),
                                 "text_b": s.content[:400].replace("\n", " "),
                                 "judgment": "", "note": ""})
                seen.setdefault(f, (nb, s.idx, s.content))
    rng.shuffle(cand)
    keep = []
    for v in FP:
        keep += [c for c in cand if c["fp_version"] == v][:limit // 2]
    path = os.path.join(RES, "e8_collisions_todo.csv")
    if keep:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(keep[0]))
            w.writeheader()
            w.writerows(keep)
    return {"n_dumped": len(keep), "path": os.path.basename(path),
            "total_collisions": {v: sum(1 for c in cand
                                        if c["fp_version"] == v) for v in FP}}


def run():
    _t0 = time.time()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    dist_r, dist_s = Distiller(idx), Distiller(idx_s)
    res = {
        "E8a_fingerprint_quality": e8a_fingerprint_quality(eps),
        "E8b_false_suppression": e8b_false_suppression(eps, dist_s),
        "E8c_saturation": e8c_saturation(trajs, eps, idx, idx_s,
                                         dist_r, dist_s),
        "E8d_real_collision_sample": e8d_collision_sample(trajs),
    }
    with open(os.path.join(RES, "e8_ledger.json"), "w") as f:
        json.dump(res, f, indent=1)
    _rt_p = os.path.join(RES, "runtimes.json")
    _rt = json.load(open(_rt_p)) if os.path.exists(_rt_p) else {}
    _rt["run_e8"] = round(time.time() - _t0, 1)
    json.dump(_rt, open(_rt_p, "w"), indent=1)
    print(json.dumps(res["E8a_fingerprint_quality"], indent=1))
    print(json.dumps(res["E8b_false_suppression"], indent=1))
    for r in res["E8c_saturation"]:
        print(r)
    print(res["E8d_real_collision_sample"])


if __name__ == "__main__":
    run()
