"""Core experiment pass: E1 (Pareto), E2 (fidelity vs position), E3
(latency/energy), E4 (side-effect safety). One pass over (trajectory,
position, policy, device pair) shared by all four; results written to
results/e1_pareto.json, e2_position.json, e3_costs.json, e4_safety.json.

Seeds: 5 (positions and the randdrop policy are the stochastic elements);
bootstrap CIs over trajectories/episodes.
"""
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import (load_all, sample_positions, real_fidelity, probe_recall,
                    hazards_real, hazards_synth, boot_ci, RES, STRICT_TYPES)
from traces.parser import est_tokens
from traces.anchors import norm_text
from state.schema import Distiller
from policies.policies import POLICIES, POLICY_ORDER, relay_budget
from policies.baselines import p7_extractive
from stats import compare
from devices.constants import DEVICES, LINKS, SCENARIOS
from devices.cost import handoff_cost

SEEDS = [0, 1, 2, 3, 4]
POS_PER_SEED = 2
PAIRS = [(v["src"], v["dst"], v["link"], name)
         for name, v in SCENARIOS.items()]
RELAY_BUDGETS = [128, 256, 512, 1024, 1792, 2048, 4096, 8192]
ALL_POLICIES = POLICY_ORDER + ["extractive"]
SWEEP_DST = "phone"   # budget sweep uses phone ctx (8192) so no cap interferes


def run():
    t_start = time.time()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    dist_r = Distiller(idx)
    dist_s = Distiller(idx_s)

    # accumulators
    fid = defaultdict(lambda: defaultdict(list))       # [policy][pair] -> per-traj recalls
    fid_sub = defaultdict(lambda: defaultdict(list))   # [policy][subcorpus]
    fid_strict = defaultdict(lambda: defaultdict(list))
    fid_synth = defaultdict(list)                      # [policy] -> per-episode probe recall
    fid_synth_fields = defaultdict(lambda: defaultdict(list))
    bytes_by = defaultdict(lambda: defaultdict(list))
    tokens_by = defaultdict(lambda: defaultdict(list))
    ser_by = defaultdict(list)
    sweep = defaultdict(lambda: {"recall": [], "strict": [], "bytes": [],
                                 "tokens": []})
    pos_bins = defaultdict(lambda: defaultdict(list))  # [policy][bin] -> recalls
    haz = defaultdict(lambda: {"haz": 0, "tot": 0, "units": []})
    haz_s = defaultdict(lambda: {"haz": 0, "tot": 0, "units": []})
    trunc_cases = defaultdict(lambda: defaultdict(int))
    restart_recompute = []
    anchor_counts = []

    # -------------------- real traces --------------------
    for ti, traj in enumerate(trajs):
        per_traj = defaultdict(lambda: defaultdict(list))
        per_traj_s = defaultdict(lambda: defaultdict(list))
        per_traj_sweep = defaultdict(lambda: defaultdict(list))
        per_traj_haz = defaultdict(lambda: [0, 0])
        for seed in SEEDS:
            for k in sample_positions(traj.n, traj.tid, seed, POS_PER_SEED):
                A = idx.future_relevant(traj, k)
                anchor_counts.append(len(A))
                kbin = round(min(0.99, k / traj.n) // 0.25 * 0.25 + 0.125, 3)
                prefix_tok = traj.prefix_tokens(k)
                restart_recompute.append(prefix_tok)
                fid_memo = {}
                for src, dst, link, pname in PAIRS:
                    for pol in ALL_POLICIES:
                        kw = {}
                        if pol.startswith("relay"):
                            kw["distiller"] = dist_r
                        if pol == "randdrop":
                            kw["seed"] = seed
                        if pol == "extractive":
                            out = p7_extractive(traj, k, dst)
                        else:
                            out = POLICIES[pol](traj, k, dst, **kw)
                        if pol == "full" and not out.truncated_at_target:
                            r_all, r_s = 1.0, 1.0   # resume is a superset
                        else:
                            mk = (pol, hashlib.md5(
                                out.resume_text.encode()).hexdigest())
                            if mk in fid_memo:
                                r_all, r_s = fid_memo[mk]
                            else:
                                r_all, r_s, _ = real_fidelity(
                                    idx, traj, k, out.resume_text, anchors=A)
                                fid_memo[mk] = (r_all, r_s)
                        per_traj[pol][pname].append(r_all)
                        per_traj_s[pol][pname].append(r_s)
                        bytes_by[pol][pname].append(out.bytes_wire)
                        tokens_by[pol][pname].append(out.tokens_wire)
                        ser_by[pol].append(out.serialize_cpu_s)
                        if out.truncated_at_target:
                            trunc_cases[pol][pname] += 1
                        if pname == "go-run":   # phone->watch: position study
                            pos_bins[pol][kbin].append(r_all)
                        if pname == "go-run":
                            h, tot = hazards_real(traj, k, out.resume_text)
                            per_traj_haz[pol][0] += h
                            per_traj_haz[pol][1] += tot
                # Relay budget sweep (device-neutral, dst=phone)
                for b in RELAY_BUDGETS:
                    out = POLICIES["relay"](traj, k, SWEEP_DST,
                                            distiller=dist_r, budget=b)
                    r_all, r_s, _ = real_fidelity(idx, traj, k,
                                                  out.resume_text, anchors=A)
                    per_traj_sweep[b]["recall"].append(r_all)
                    per_traj_sweep[b]["strict"].append(r_s)
                    per_traj_sweep[b]["bytes"].append(out.bytes_wire)
                    per_traj_sweep[b]["tokens"].append(out.tokens_wire)
        # aggregate per trajectory (unit for bootstrap)
        for pol in ALL_POLICIES:
            for _, _, _, pname in PAIRS:
                if per_traj[pol][pname]:
                    fid[pol][pname].append(
                        sum(per_traj[pol][pname]) / len(per_traj[pol][pname]))
                    if pname == "go-run":
                        fid_sub[pol][traj.source].append(fid[pol][pname][-1])
                    fid_strict[pol][pname].append(
                        sum(per_traj_s[pol][pname]) / len(per_traj_s[pol][pname]))
            h, tot = per_traj_haz[pol]
            haz[pol]["haz"] += h
            haz[pol]["tot"] += tot
            if tot:
                haz[pol]["units"].append(h / tot)
        for b in RELAY_BUDGETS:
            d = per_traj_sweep[b]
            if d["recall"]:
                sweep[b]["recall"].append(sum(d["recall"]) / len(d["recall"]))
                sweep[b]["strict"].append(sum(d["strict"]) / len(d["strict"]))
                sweep[b]["bytes"].append(sum(d["bytes"]) / len(d["bytes"]))
                sweep[b]["tokens"].append(sum(d["tokens"]) / len(d["tokens"]))
        if (ti + 1) % 50 == 0:
            print(f"  real {ti+1}/{len(trajs)} ({time.time()-t_start:.0f}s)",
                  flush=True)

    # -------------------- synthetic episodes --------------------
    for ei, ep in enumerate(eps):
        st = synth_trajs[ei]
        per_ep = defaultdict(list)
        per_ep_haz = defaultdict(lambda: [0, 0])
        for seed in SEEDS:
            for k in sample_positions(ep.n, ep.eid, seed, 1):
                for pol in ALL_POLICIES:
                    kw = {}
                    if pol.startswith("relay"):
                        kw["distiller"] = dist_s
                    if pol == "randdrop":
                        kw["seed"] = seed
                    out = (p7_extractive(st, k, "watch") if pol == "extractive"
                           else POLICIES[pol](st, k, "watch", **kw))
                    pr, fields = probe_recall(ep, k, out.resume_text)
                    per_ep[pol].append(pr)
                    for f, v in fields.items():
                        fid_synth_fields[pol][f].append(v)
                    h, tot = hazards_synth(ep, k, out.resume_text)
                    per_ep_haz[pol][0] += h
                    per_ep_haz[pol][1] += tot
        for pol in ALL_POLICIES:
            fid_synth[pol].append(sum(per_ep[pol]) / len(per_ep[pol]))
            h, tot = per_ep_haz[pol]
            haz_s[pol]["haz"] += h
            haz_s[pol]["tot"] += tot
            if tot:
                haz_s[pol]["units"].append(h / tot)

    # -------------------- write E1 --------------------
    e1 = {"pairs": {}, "sweep": {}, "synth": {}, "notes": {
        "anchor_count_mean": sum(anchor_counts) / len(anchor_counts),
        "n_real_evals": len(anchor_counts), "seeds": SEEDS,
        "pos_per_seed": POS_PER_SEED, "strict_types": sorted(STRICT_TYPES)}}
    for pol in ALL_POLICIES:
        e1["pairs"][pol] = {}
        for _, _, _, pname in PAIRS:
            m, lo, hi = boot_ci(fid[pol][pname])
            ms, los, his = boot_ci(fid_strict[pol][pname])
            e1["pairs"][pol][pname] = {
                "recall": [m, lo, hi], "recall_strict": [ms, los, his],
                "bytes_mean": sum(bytes_by[pol][pname]) / max(1, len(bytes_by[pol][pname])),
                "tokens_mean": sum(tokens_by[pol][pname]) / max(1, len(tokens_by[pol][pname]))}
        m, lo, hi = boot_ci(fid_synth[pol]) if fid_synth[pol] else (0, 0, 0)
        e1["synth"][pol] = {"probe_recall": [m, lo, hi],
                            "fields": {f: sum(v) / len(v) for f, v in
                                       fid_synth_fields[pol].items()}}
    for b in RELAY_BUDGETS:
        m, lo, hi = boot_ci(sweep[b]["recall"])
        ms, _, _ = boot_ci(sweep[b]["strict"])
        e1["sweep"][b] = {"recall": [m, lo, hi], "recall_strict": ms,
                          "bytes_mean": sum(sweep[b]["bytes"]) / len(sweep[b]["bytes"]),
                          "tokens_mean": sum(sweep[b]["tokens"]) / len(sweep[b]["tokens"])}
    # paired significance vs Relay (same trajectory, same positions)
    e1["paired_vs_relay"] = {}
    for pname in [p[3] for p in PAIRS]:
        base = fid["relay"][pname]
        others = {p: fid[p][pname] for p in ALL_POLICIES if p != "relay"}
        e1["paired_vs_relay"][pname] = compare(base, others)
        bs = fid_strict["relay"][pname]
        e1["paired_vs_relay"][pname + "|strict"] = compare(
            bs, {p: fid_strict[p][pname] for p in ALL_POLICIES if p != "relay"})
    # generalisation across the two source frameworks
    e1["by_subcorpus"] = {
        pol: {sub: [round(v, 4) for v in boot_ci(vals)]
              for sub, vals in fid_sub[pol].items()}
        for pol in ALL_POLICIES}
    e1["by_subcorpus_n"] = {sub: len(fid_sub["relay"][sub])
                            for sub in fid_sub["relay"]}
    json.dump(e1, open(os.path.join(RES, "e1_pareto.json"), "w"), indent=1)

    # -------------------- write E2 --------------------
    e2 = {}
    for pol in ALL_POLICIES:
        e2[pol] = {}
        for kbin, vals in sorted(pos_bins[pol].items()):
            m, lo, hi = boot_ci(vals)
            e2[pol][str(kbin)] = {"recall": [m, lo, hi], "n": len(vals)}
    json.dump(e2, open(os.path.join(RES, "e2_position.json"), "w"), indent=1)

    # -------------------- write E3 --------------------
    e3 = {"cells": [], "trunc_fraction": {}}
    n_evals = len(anchor_counts)
    for pol in ALL_POLICIES:
        ser_mean = sum(ser_by[pol]) / max(1, len(ser_by[pol]))
        for src, dst, link0, pname in PAIRS:
            by = sum(bytes_by[pol][pname]) / max(1, len(bytes_by[pol][pname]))
            tk = sum(tokens_by[pol][pname]) / max(1, len(tokens_by[pol][pname]))
            for link in LINKS:
                c = handoff_cost(int(by), int(tk), src, dst, link, ser_mean)
                e3["cells"].append({
                    "policy": pol, "pair": pname, "src": src, "dst": dst,
                    "link": link, "native_link": link == link0,
                    "bytes": int(by), "tokens": int(tk),
                    "latency_s": round(c.latency_s, 3),
                    "serialize_s": round(c.serialize_s, 4),
                    "transfer_s": round(c.transfer_s, 3),
                    "prefill_s": round(c.prefill_s, 3),
                    "energy_j": round(c.energy_j, 4),
                    "truncated": c.truncated_at_target})
        e3["trunc_fraction"][pol] = {
            pname: trunc_cases[pol][pname] / (n_evals / 1)
            for _, _, _, pname in PAIRS}
    # note: trunc_fraction denominators = n (traj,k) evals per pair
    for pol in e3["trunc_fraction"]:
        for pname in e3["trunc_fraction"][pol]:
            e3["trunc_fraction"][pol][pname] = round(
                trunc_cases[pol][pname] / n_evals, 4)
    json.dump(e3, open(os.path.join(RES, "e3_costs.json"), "w"), indent=1)

    # -------------------- write E4 --------------------
    e4 = {"real": {}, "synth": {}, "restart_recompute_tokens":
          {"mean": sum(restart_recompute) / len(restart_recompute),
           "max": max(restart_recompute)}}
    for pol in ALL_POLICIES:
        m, lo, hi = boot_ci(haz[pol]["units"])
        e4["real"][pol] = {"hazards": haz[pol]["haz"], "labeled": haz[pol]["tot"],
                           "rate": [m, lo, hi]}
        m, lo, hi = boot_ci(haz_s[pol]["units"])
        e4["synth"][pol] = {"hazards": haz_s[pol]["haz"],
                            "executed": haz_s[pol]["tot"], "rate": [m, lo, hi]}
    json.dump(e4, open(os.path.join(RES, "e4_safety.json"), "w"), indent=1)

    wall = time.time() - t_start
    rt = {}
    p = os.path.join(RES, "runtimes.json")
    if os.path.exists(p):
        rt = json.load(open(p))
    rt["run_core_E1-E4"] = round(wall, 1)
    json.dump(rt, open(p, "w"), indent=1)
    print(f"core done in {wall:.0f}s")


if __name__ == "__main__":
    run()
