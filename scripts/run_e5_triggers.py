"""E5: handoff trigger policies -- stall time vs wasted pre-ship bytes.

Modeled, per SPEC (compact): the user switches devices at step k of a real
trajectory (u.a.r. positions, 5 seeds x 1 position). Trigger policies:

  immediate   ship the Relay block distilled at k the moment the switch
              happens. Stall = serialize + transfer + prefill.
  defer       wait for the current step to finish (consistent boundary):
              delta = 0.5 x (tokens of step k) / source decode rate, then
              ship the same block. Stall = delta + handoff latency.
  predictive  pre-ship the block distilled at k-2 while still on the source
              device; at the switch, ship only a delta (the raw text of
              steps k-1..k, gzipped, capped at the Relay budget) and prefill
              only the delta. Stall = delta transfer + delta prefill.
              Waste = P(wrong prediction) x pre-shipped bytes; predictor
              accuracy is a modeled parameter, reported at 0.9/0.7/0.5.

All quantities from the same cost model as E3 (modeled links, measured
serialize CPU). No live deployment claimed.
"""
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, sample_positions, boot_ci, RES
from state.schema import Distiller
from scenarios.triggers import (SCENARIOS, immediate_stall, defer_stall,
                                predictive_stall)

SEEDS = [0, 1, 2, 3, 4]
P_WRONG = [0.1, 0.3, 0.5]


def run():
    t0 = time.time()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    dist = Distiller(idx)
    out = {"scenarios": {}, "p_wrong": P_WRONG}
    for sname, sc in SCENARIOS.items():
        src, dst, link = sc["src"], sc["dst"], sc["link"]
        stall = defaultdict(list)
        waste = defaultdict(list)
        for traj in trajs:
            per = defaultdict(list)
            per_w = defaultdict(list)
            for seed in SEEDS:
                for k in sample_positions(traj.n, traj.tid, seed, 1):
                    for trig, fn in (("immediate", immediate_stall),
                                     ("defer", defer_stall),
                                     ("predictive", predictive_stall)):
                        st_s, waste_b, _ = fn(traj, k, src, dst, link, dist)
                        per[trig].append(st_s)
                        per_w[trig].append(waste_b)
            for trig in ("immediate", "defer", "predictive"):
                stall[trig].append(sum(per[trig]) / len(per[trig]))
                waste[trig].append(sum(per_w[trig]) / len(per_w[trig]))
        out["scenarios"][sname] = {}
        for trig in ("immediate", "defer", "predictive"):
            m, lo, hi = boot_ci(stall[trig])
            wb = sum(waste[trig]) / len(waste[trig])
            entry = {"stall_s": [round(m, 3), round(lo, 3), round(hi, 3)],
                     "preship_bytes_mean": round(wb, 1)}
            if trig == "predictive":
                entry["wasted_bytes_by_p_wrong"] = {
                    str(p): round(wb * p, 1) for p in P_WRONG}
            out["scenarios"][sname][trig] = entry
    json.dump(out, open(os.path.join(RES, "e5_triggers.json"), "w"), indent=1)
    rt = json.load(open(os.path.join(RES, "runtimes.json"))) \
        if os.path.exists(os.path.join(RES, "runtimes.json")) else {}
    rt["run_e5"] = round(time.time() - t0, 1)
    json.dump(rt, open(os.path.join(RES, "runtimes.json"), "w"), indent=1)
    print(f"e5 done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
