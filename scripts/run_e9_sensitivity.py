"""E9: sensitivity of the deployability conclusions to the modeled constants.

Every latency and energy number in E3 is produced by a model whose
constants are read off the measurement literature.  A reader is entitled to
ask which conclusions survive being wrong about them.  E9 re-runs the E3
comparison while sweeping, one at a time:

  * target prefill throughput  (x0.5 .. x16 of the modeled value -- the
    upper end covers the regime our own measured on-device prefill (E10a)
    would imply if a wearable ran as fast as an M-series laptop),
  * target context window      (1k / 2k / 4k on the watch class),
  * link bandwidth             (x0.25 .. x133, i.e. BLE .. WiFi-class),

and reports, for each setting, the policy ordering, the Relay-vs-recency
and Relay-vs-full stall ratios, and the share of latency that prefill
accounts for.  The claim under test is not "62 seconds" -- it is
"prefill dominates and Relay is the cheapest faithful policy"; E9 states
the region of the parameter space in which that holds.
"""
import copy
import json
import os
import time
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, sample_positions, RES
from state.schema import Distiller
from policies.policies import POLICIES, usable
from devices import constants as C
from devices.cost import handoff_cost

POLS = ["full", "recency", "relay", "relay-lite", "restart"]
SCEN = ("phone", "watch", "ble")
SEEDS = [0]


def _collect(trajs, dist):
    """Per-policy mean (bytes, tokens, serialize) at the go-run handoff."""
    acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for traj in trajs:
        for seed in SEEDS:
            for k in sample_positions(traj.n, traj.tid, seed, 2):
                for p in POLS:
                    kw = {"distiller": dist} if p.startswith("relay") else {}
                    o = POLICIES[p](traj, k, "watch", **kw)
                    a = acc[p]
                    a[0] += o.bytes_wire
                    a[1] += o.tokens_wire
                    a[2] += o.serialize_cpu_s
                    a[3] += 1
    return {p: (a[0] / a[3], a[1] / a[3], a[2] / a[3]) for p, a in acc.items()}


def _latency(stats, dst="watch"):
    src, _, link = SCEN
    out = {}
    for p, (b, t, ser) in stats.items():
        c = handoff_cost(int(b), int(t), src, dst, link, ser)
        out[p] = {"latency_s": c.latency_s, "prefill_s": c.prefill_s,
                  "transfer_s": c.transfer_s, "energy_j": c.energy_j}
    return out


def run():
    _t0 = time.time()
    trajs, idx, eps, synth, idx_s = load_all()
    dist = Distiller(idx)
    base_dev = copy.deepcopy(C.DEVICES)
    base_link = copy.deepcopy(C.LINKS)
    out = {"sweeps": {}}

    def record(tag, key, val, stats):
        lat = _latency(stats)
        order = sorted(POLS, key=lambda p: lat[p]["latency_s"])
        pref_share = (lat["relay"]["prefill_s"] /
                      max(lat["relay"]["latency_s"], 1e-9))
        out["sweeps"].setdefault(tag, []).append({
            key: val,
            "latency_s": {p: round(lat[p]["latency_s"], 2) for p in POLS},
            "tokens": {p: round(stats[p][1], 1) for p in POLS},
            "order_by_latency": order,
            "relay_vs_recency": round(lat["relay"]["latency_s"] /
                                      lat["recency"]["latency_s"], 3),
            "relay_vs_full": round(lat["relay"]["latency_s"] /
                                   lat["full"]["latency_s"], 3),
            "prefill_share_relay": round(pref_share, 4)})

    # Policy outputs depend only on the target window, so the shipped
    # bytes/tokens are collected once per window setting and re-costed for
    # the throughput and bandwidth sweeps.
    base_stats = _collect(trajs, dist)

    # --- prefill throughput ---
    for m in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        C.DEVICES["watch"]["prefill_tps"] = base_dev["watch"]["prefill_tps"] * m
        record("prefill_tps", "multiplier", m, base_stats)
    C.DEVICES["watch"]["prefill_tps"] = base_dev["watch"]["prefill_tps"]

    # --- target window (policies re-run: the window changes what fits) ---
    for ctx in (1024, 2048, 4096):
        C.DEVICES["watch"]["ctx"] = ctx
        record("watch_ctx", "ctx", ctx,
               base_stats if ctx == base_dev["watch"]["ctx"]
               else _collect(trajs, dist))
    C.DEVICES["watch"]["ctx"] = base_dev["watch"]["ctx"]

    # --- link bandwidth ---
    for m in (0.25, 1.0, 10.0, 133.0):
        C.LINKS["ble"]["bw_mbps"] = base_link["ble"]["bw_mbps"] * m
        record("ble_bw", "multiplier", m, base_stats)
    C.LINKS["ble"]["bw_mbps"] = base_link["ble"]["bw_mbps"]

    # crossover: bandwidth at which transfer overtakes prefill for full ctx
    stats = base_stats
    b_full, t_full, _ = stats["full"]
    pre = min(t_full, usable("watch")) / base_dev["watch"]["prefill_tps"]
    bw_cross = b_full * 8 / max(pre - base_link["ble"]["rtt_s"], 1e-9) / 1e6
    out["transfer_overtakes_prefill_bw_mbps"] = round(bw_cross, 5)
    out["note"] = ("bandwidth below which transfer exceeds prefill for the "
                   "full-context policy on a watch target")
    with open(os.path.join(RES, "e9_sensitivity.json"), "w") as f:
        json.dump(out, f, indent=1)
    _rt_p = os.path.join(RES, "runtimes.json")
    _rt = json.load(open(_rt_p)) if os.path.exists(_rt_p) else {}
    _rt["run_e9"] = round(time.time() - _t0, 1)
    json.dump(_rt, open(_rt_p, "w"), indent=1)
    for tag, rows in out["sweeps"].items():
        print("==", tag)
        for r in rows:
            print("  ", {k: v for k, v in r.items()
                         if k not in ("latency_s", "tokens")},
                  {p: r["latency_s"][p] for p in ("full", "recency", "relay")})
    print("crossover bw (Mbps):", out["transfer_overtakes_prefill_bw_mbps"])


if __name__ == "__main__":
    run()
