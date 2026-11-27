"""E7: scaling of state size with trajectory length + extractor audit recap.

For every real trajectory and positions k in {25%, 50%, 75%, N-2}, compute:
  full-context gzipped bytes (linear in prefix), and
  Relay natural (unbudgeted) block gzipped bytes and tokens.
Fit log-log slopes of size vs prefix tokens: full ~ 1.0, Relay sublinear.
The extractor-precision audit summary is copied in for the paper tables.
"""
import gzip
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, RES
from traces.parser import est_tokens, exact_tokens_cached
from state.schema import Distiller
from policies.policies import relay_budget

FRACS = [0.25, 0.5, 0.75, 1.0]


def run():
    t0 = time.time()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    dist = Distiller(idx)
    pts = []
    for traj in trajs:
        for f in FRACS:
            k = max(2, min(traj.n - 2, int(round(f * traj.n)) - 1))
            prefix = traj.prefix_text(k)
            ptok = exact_tokens_cached(prefix)
            full_b = len(gzip.compress(prefix.encode(), 6))
            blk = dist.distill(traj, k, budget_tokens=None)
            blk_w = dist.distill(traj, k,
                                 budget_tokens=relay_budget("watch"))
            pts.append({"tid": traj.tid, "k": k, "n": traj.n,
                        "prefix_tokens": ptok, "full_bytes": full_b,
                        "relay_bytes": len(blk.serialize()),
                        "relay_tokens": exact_tokens_cached(blk.render()),
                        "relay_watch_bytes": len(blk_w.serialize()),
                        "relay_watch_tokens":
                            exact_tokens_cached(blk_w.render())})
    x = np.log([p["prefix_tokens"] for p in pts])
    slope_full = float(np.polyfit(x, np.log([p["full_bytes"] for p in pts]), 1)[0])
    slope_relay = float(np.polyfit(x, np.log([p["relay_bytes"] for p in pts]), 1)[0])
    slope_relay_tok = float(np.polyfit(
        x, np.log([p["relay_tokens"] for p in pts]), 1)[0])
    slope_watch = float(np.polyfit(
        x, np.log([p["relay_watch_bytes"] for p in pts]), 1)[0])
    ratio_big = [p["full_bytes"] / p["relay_bytes"] for p in pts
                 if p["prefix_tokens"] > 8000]
    ratio_watch = [p["full_bytes"] / p["relay_watch_bytes"] for p in pts
                   if p["prefix_tokens"] > 8000]
    audit = json.load(open(os.path.join(RES, "anchor_audit_summary.json")))
    hist = json.load(open(os.path.join(RES, "anchor_audit_history.json")))
    rec_p = os.path.join(RES, "anchor_recall_summary.json")
    recall_audit = json.load(open(rec_p)) if os.path.exists(rec_p) else {}
    out = {"points": pts,
           "slopes_loglog": {"full_bytes": round(slope_full, 3),
                             "relay_bytes": round(slope_relay, 3),
                             "relay_tokens": round(slope_relay_tok, 3),
                             "relay_watch_bytes": round(slope_watch, 3),
                             "full_tokens": 1.0},
           "compression_ratio_prefix_gt8k": {
               "mean": round(float(np.mean(ratio_big)), 2),
               "median": round(float(np.median(ratio_big)), 2),
               "n": len(ratio_big)},
           "full_over_watch_ratio_gt8k": {
               "mean": round(float(np.mean(ratio_watch)), 2),
               "median": round(float(np.median(ratio_watch)), 2)},
           "audit": audit, "audit_history_final": hist["final"],
           "recall_audit": recall_audit,
           "watch_budget": relay_budget("watch")}
    json.dump(out, open(os.path.join(RES, "e7_scaling.json"), "w"), indent=1)
    rt_p = os.path.join(RES, "runtimes.json")
    rt = json.load(open(rt_p)) if os.path.exists(rt_p) else {}
    rt["run_e7"] = round(time.time() - t0, 1)
    json.dump(rt, open(rt_p, "w"), indent=1)
    print(f"e7 done in {time.time()-t0:.0f}s; slopes full={slope_full:.3f} "
          f"relay={slope_relay:.3f}")


if __name__ == "__main__":
    run()
