"""E19: how often is a handoff immediately non-actionable on its target?

E15 reported the capability-repair verdict distribution over *plan steps*.
A reviewer asked the per-*handoff* question a user would feel: per target
class, how often does repair choose DELEGATE, DEFER or BLOCKED at all, and
what fraction of handoffs cannot take their very next step locally?

For every sampled handoff whose shipped block carries a pending plan (the
same plans E15 classifies; watch-budget distillation, 5 seeds x 2 positions)
and for each target manifest, with the source reachable and unreachable:

  ``any_<verdict>``          the handoff's plan has >= 1 step with that verdict;
  ``next_not_local``         the *first* pending step is not executable on the
                             target as written or after a modality
                             substitution (DELEGATE, DEFER or BLOCKED);
  ``next_stalled``           the first pending step is DEFER or BLOCKED, i.e.
                             nothing reachable can perform it now and the user
                             must be told (a user-visible interruption);
  ``all_not_local``          no step of the plan is locally executable.

"First pending step" on real traces means the first step of the plan text the
block ships (the most recent explicit plan), which can include steps already
done: real traces carry no ground-truth pending step.  On synthetic episodes
the ground-truth next pending step is known, and ``true_next_not_local`` /
``true_next_stalled`` classify *that* step instead.

These are counts of *user-visible states the design creates*, which is a
structural proxy for interruption burden.  They are not a measurement of user
burden; no user was involved.  The requirement model is the same audited-by-
construction regex heuristic as E15, so outputs are "capability-gap-like".
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, boot_ci, RES                               # noqa
from state.capability import (DEFAULT_MANIFESTS, CapabilityManifest,     # noqa
                              Verdict, repair_plan, repair_step)
from state.schema import Distiller                                       # noqa
from run_e15_capability import _plans, CLASSES                           # noqa

OUT = os.path.join(RES, "e19_actionability.json")
LOCAL = {Verdict.FEASIBLE, Verdict.SUBSTITUTE}


def per_handoff(plans, manifests, episodes=None):
    out = {}
    ep = {e.eid: e for e in (episodes or [])}
    for dev, m in manifests.items():
        flags = {k: [] for k in ("any_delegate", "any_defer", "any_blocked",
                                 "any_substitute", "any_nonfeasible",
                                 "next_not_local", "next_stalled",
                                 "all_not_local", "true_next_not_local",
                                 "true_next_stalled")}
        for t, k, plan, _ in plans:
            steps = repair_plan(plan, m)
            if not steps:
                continue
            v = [s.verdict for s in steps]
            flags["any_delegate"].append(float(Verdict.DELEGATE in v))
            flags["any_defer"].append(float(Verdict.DEFER in v))
            flags["any_blocked"].append(float(Verdict.BLOCKED in v))
            flags["any_substitute"].append(float(Verdict.SUBSTITUTE in v))
            flags["any_nonfeasible"].append(
                float(any(x is not Verdict.FEASIBLE for x in v)))
            flags["next_not_local"].append(float(v[0] not in LOCAL))
            flags["next_stalled"].append(
                float(v[0] in (Verdict.DEFER, Verdict.BLOCKED)))
            flags["all_not_local"].append(float(all(x not in LOCAL for x in v)))
            e = ep.get(t.tid)
            if e is not None and e.states[k].plan_remaining:
                tv = repair_step(e.states[k].plan_remaining[0], m).verdict
                flags["true_next_not_local"].append(float(tv not in LOCAL))
                flags["true_next_stalled"].append(
                    float(tv in (Verdict.DEFER, Verdict.BLOCKED)))
        out[dev] = {"n_handoffs": len(flags["next_not_local"])}
        for k, a in flags.items():
            if not a:
                continue
            m_, lo, hi = boot_ci(a)
            out[dev][k] = [round(m_, 4), round(lo, 4), round(hi, 4)]
    return out


def run():
    t0 = time.time()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    real_plans = _plans(trajs, idx, Distiller(idx))
    synth_plans = _plans(synth_trajs, idx_s, Distiller(idx_s))
    offline = {d: CapabilityManifest(m.device, set(m.tools), set(m.modalities),
                                     set(m.scopes), source_reachable=False,
                                     cloud_reachable=False)
               for d, m in DEFAULT_MANIFESTS.items()}
    out = {"real_source_up": per_handoff(real_plans, DEFAULT_MANIFESTS),
           "real_source_down": per_handoff(real_plans, offline),
           "synth_source_up": per_handoff(synth_plans, DEFAULT_MANIFESTS, eps),
           "synth_source_down": per_handoff(synth_plans, offline, eps),
           "n_real_plans": len(real_plans), "n_synth_plans": len(synth_plans),
           "runtime_s": round(time.time() - t0, 1)}
    json.dump(out, open(OUT, "w"), indent=1)
    for tag in ("real_source_up", "real_source_down", "synth_source_up",
                "synth_source_down"):
        print(f"\n{tag}")
        for dev in CLASSES:
            d = out[tag][dev]
            print(f"  {dev:8s} n={d['n_handoffs']} "
                  + " ".join(f"{k}={d[k][0]:.3f}" for k in d
                             if k != "n_handoffs"))
    print("->", OUT)


if __name__ == "__main__":
    run()
