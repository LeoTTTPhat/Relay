"""E16: where a handoff actually spends its joules.

The first version of this system reported *radio* energy and concluded that
energy was not where the interesting differences lived -- on BLE the policies
differed by a few millijoules, and on LTE the post-transfer tail swamped
everyone equally.  That conclusion was an artifact of the missing term.  The
target has to prefill whatever arrives, prefill is seconds of sustained SoC
load, and seconds of SoC load is joules, not millijoules.  A cost model that
prices the radio and not the NPU prices the cheap half of the handoff.

This experiment adds the compute term (``devices/constants.py: infer_mw``)
and reports:

  **E16a** radio, compute and total energy per policy and scenario, with the
     compute share, so the reader can see which term the design controls;
  **E16b** a sweep of the inference-power constants over 16x, because they are
     the least well sourced numbers in the model and a conclusion that does
     not survive them does not deserve to be stated;
  **E16c** energy per unit of fidelity -- joules spent per point of
     future-relevance recall -- which is the comparison a battery-limited
     device would actually make and which no single-axis number gives.

The inference-power constants are modelled, like the throughput and link
constants before them, and the paper says so.  What changes is not the
confidence, it is the *sign* of the conclusion: with the compute term
included, capability-aware sizing saves energy in the units that matter,
and it saves it for the same reason it saves latency.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from explib import (load_all, sample_positions, real_fidelity,          # noqa
                    policy_budget, RES)
from state.schema import Distiller                                      # noqa
from policies.policies import POLICIES                                  # noqa
from devices.constants import DEVICES, SCENARIOS                        # noqa
from devices.cost import handoff_cost                                   # noqa

OUT = os.path.join(RES, "e16_energy.json")
SEEDS = [0, 1, 2]
POLS = ["full", "trunc25", "recency", "typed-recency", "relay", "relay-lite",
        "restart", "randdrop"]


def _sweep_infer(scale):
    """Temporarily scale every device's inference power."""
    saved = {d: DEVICES[d]["infer_mw"] for d in DEVICES}
    for d in DEVICES:
        DEVICES[d]["infer_mw"] = saved[d] * scale
    return saved


def _restore(saved):
    for d, v in saved.items():
        DEVICES[d]["infer_mw"] = v


def run():
    import time
    t0 = time.time()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    dist = Distiller(idx)

    # one pass collecting (tokens, bytes, serialize, recall) per policy/scenario
    per = {sc: {p: {"lat": [], "radio": [], "compute": [], "total": [],
                    "recall": []} for p in POLS} for sc in SCENARIOS}
    for t in trajs:
        for seed in SEEDS:
            for k in sample_positions(len(t.steps), t.tid, seed):
                if k >= len(t.steps):
                    continue
                A = idx.future_relevant(t, k)
                for sc, v in SCENARIOS.items():
                    src, dst, link = v["src"], v["dst"], v["link"]
                    for p in POLS:
                        kw = {"distiller": dist} if p.startswith("relay") else {}
                        o = POLICIES[p](t, k, dst, seed=seed,
                                        budget=policy_budget(p, dst), **kw)
                        c = handoff_cost(o.bytes_wire, o.tokens_wire, src, dst,
                                         link, o.serialize_cpu_s)
                        r, _, _ = real_fidelity(idx, t, k, o.resume_text,
                                                anchors=A)
                        d = per[sc][p]
                        d["lat"].append(c.latency_s)
                        d["radio"].append(c.energy_j)
                        d["compute"].append(c.compute_energy_j)
                        d["total"].append(c.total_energy_j)
                        d["recall"].append(r)

    e16a = {}
    for sc, pol in per.items():
        e16a[sc] = {}
        for p, d in pol.items():
            radio = float(np.mean(d["radio"]))
            comp = float(np.mean(d["compute"]))
            tot = float(np.mean(d["total"]))
            rec = float(np.mean(d["recall"]))
            e16a[sc][p] = {
                "latency_s": round(float(np.mean(d["lat"])), 2),
                "radio_j": round(radio, 5),
                "compute_j": round(comp, 3),
                "total_j": round(tot, 3),
                "compute_share": round(comp / tot, 4) if tot else 0.0,
                "recall": round(rec, 4),
                # E16c: joules per point of recall -- the battery-limited view
                "j_per_recall": round(tot / rec, 3) if rec > 0 else None,
            }
    # savings against the ship-everything baseline, in both energy units
    for sc, pol in e16a.items():
        base = pol["full"]
        for p, d in pol.items():
            d["radio_saving_mj"] = round((base["radio_j"] - d["radio_j"]) * 1000, 3)
            d["compute_saving_j"] = round(base["compute_j"] - d["compute_j"], 3)

    # --- E16b: does "compute dominates" survive the constant being wrong? ---
    e16b = {}
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        saved = _sweep_infer(scale)
        row = {}
        for sc, v in SCENARIOS.items():
            src, dst, link = v["src"], v["dst"], v["link"]
            # prefill *seconds* do not depend on the power constant, so the
            # compute term scales linearly with it and the radio term does not
            rel_c = float(np.mean(per[sc]["relay"]["compute"])) * scale
            rel_r = float(np.mean(per[sc]["relay"]["radio"]))
            base_c = float(np.mean(per[sc]["full"]["compute"])) * scale
            row[sc] = {
                "relay_compute_share": round(rel_c / (rel_c + rel_r), 4)
                if (rel_c + rel_r) else 0.0,
                "relay_vs_full_saving_j": round(base_c - rel_c, 3),
                "relay_over_full": round(rel_c / base_c, 4) if base_c else None}
        _restore(saved)
        e16b[f"infer_mw x{scale}"] = row

    out = {"e16a_per_scenario": e16a, "e16b_infer_power_sweep": e16b,
           "n_seeds": len(SEEDS), "runtime_s": round(time.time() - t0, 1)}
    json.dump(out, open(OUT, "w"), indent=1)

    print("E16a  go-run (phone->watch, BLE)")
    print(f"{'policy':14s} {'lat s':>7s} {'radio mJ':>9s} {'compute J':>10s} "
          f"{'total J':>8s} {'cmp share':>10s} {'J/recall':>9s}")
    for p, d in e16a["go-run"].items():
        print(f"{p:14s} {d['latency_s']:7.1f} {d['radio_j']*1000:9.2f} "
              f"{d['compute_j']:10.2f} {d['total_j']:8.2f} "
              f"{d['compute_share']:10.4f} {d['j_per_recall']:9.2f}")
    print("\nE16b sweep (relay compute share / saving vs full, go-run):")
    for k, v in e16b.items():
        r = v["go-run"]
        print(f"  {k:16s} share={r['relay_compute_share']:.4f} "
              f"saving={r['relay_vs_full_saving_j']:8.2f} J  "
              f"ratio={r['relay_over_full']:.3f}")
    print("->", OUT)


if __name__ == "__main__":
    run()
