"""Conservative mode's real-trace cost, aggregated like the state-preservation
results (model-free).

``run_e27_conservative.py: real_cost`` averages future-relevance recall over
the 1,641 handoffs, so its base-mode value (0.897) differs from the 0.903 that
the state-preservation paragraph reports for the same Relay configuration:
there (``run_core.py``, E1, phone->watch pair) recall is averaged per
trajectory first and the trajectory is the unit of the bootstrap and of the
paired test.  This script recomputes both modes on exactly the same handoffs
(Who&When, seeds 0-4 x 2 positions, watch budget = usable window, the default
distiller) and reports both aggregations, with the paired Wilcoxon test over
trajectories, so the paper can quote one consistent number.

Output: ``results/e27_real_cost_units.json``.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from explib import (load_all, sample_positions, real_fidelity, boot_ci,  # noqa
                    policy_budget, RES)
from stats import compare                                               # noqa
from state.schema import Distiller                                      # noqa
from policies.policies import POLICIES                                  # noqa

SEEDS = [0, 1, 2, 3, 4]
DST = "watch"


def main():
    trajs, idx, *_ = load_all()
    base, cons = Distiller(idx), Distiller(idx, conservative=True)
    per_h = {"base": [], "cons": []}
    per_t = {"base": [], "cons": []}
    tok_t = {"base": [], "cons": []}
    for t in trajs:
        acc = defaultdict(list)
        tok = defaultdict(list)
        for seed in SEEDS:
            for k in sample_positions(len(t.steps), t.tid, seed):
                if k >= len(t.steps):
                    continue
                A = idx.future_relevant(t, k)
                for name, dist in (("base", base), ("cons", cons)):
                    o = POLICIES["relay"](t, k, DST, distiller=dist,
                                          budget=policy_budget("relay", DST))
                    r, _, _ = real_fidelity(idx, t, k, o.resume_text,
                                            anchors=A)
                    acc[name].append(r)
                    tok[name].append(o.tokens_resume)
                    per_h[name].append(r)
        for name in ("base", "cons"):
            if acc[name]:
                per_t[name].append(float(np.mean(acc[name])))
                tok_t[name].append(float(np.mean(tok[name])))
    e1 = json.load(open(os.path.join(RES, "e1_pareto.json")))
    out = {
        "n_trajectories": len(per_t["base"]),
        "n_handoffs": len(per_h["base"]),
        "per_handoff_mean": {k: round(float(np.mean(v)), 4)
                             for k, v in per_h.items()},
        "per_trajectory": {k: [round(x, 4) for x in boot_ci(v)]
                           for k, v in per_t.items()},
        "tokens_per_trajectory_mean": {k: round(float(np.mean(v)), 1)
                                       for k, v in tok_t.items()},
        "paired_over_trajectories": compare(per_t["cons"],
                                            {"base": per_t["base"]}),
        "e1_relay_go_run_recall": round(
            e1["pairs"]["relay"]["go-run"]["recall"][0], 4),
    }
    json.dump(out, open(os.path.join(RES, "e27_real_cost_units.json"), "w"),
              indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
