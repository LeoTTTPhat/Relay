"""E14a: does the fidelity ordering survive a model with headroom?

E10's fidelity arm was a null result and we reported it as one: a 3B model
resuming GAIA-class research from a wearable-sized block recovers the
ground-truth answer about 6% of the time whatever it is given, so the study
could confirm that the *ledger* is read but could not arbitrate whether better
state produces better answers.  The obvious objection is that this is a
property of the 3B model rather than of the state: a larger model can be run
offline for *metric validation* even when the deployment target is small, and
if the ordering appears there, the deterministic proxies are doing their job.

This script re-runs the resumed-model arms with a larger model from the same
family (default Qwen2.5-7B-Instruct) and, optionally, one from a different
family, so that "one model, one size" stops being the limitation it was.  The
blocks are unchanged -- still built for a watch-class budget -- so this is not
a claim that a 7B model runs on a watch.  The target device sets the *budget*;
the grader's job is only to say whether the budget was spent well.

``RELAY_E14_MODEL``  model to evaluate (default Qwen/Qwen2.5-7B-Instruct)
``RELAY_E14_TAG``    key to store results under (default the model name)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import run_e10_llm as E10                                          # noqa: E402
from explib import load_all, RES                                   # noqa: E402
from state.schema import Distiller                                 # noqa: E402

MODEL = os.environ.get("RELAY_E14_MODEL", "Qwen/Qwen2.5-7B-Instruct")
TAG = os.environ.get("RELAY_E14_TAG", MODEL.split("/")[-1])
OUT = os.path.join(RES, "e14_stronger_model.json")

# the typed-recency baseline did not exist when E10 was written; it is the
# comparison that separates schema from allocator, so it belongs here too
E10.POLS = ["full", "recency", "typed-recency", "relay", "relay-lite",
            "extractive", "trunc25", "randdrop", "restart"]


def run():
    t0 = time.time()
    trajs, idx, eps, synth, idx_s = load_all()
    dist_r, dist_s = Distiller(idx), Distiller(idx_s)

    lm = E10.LM(MODEL)
    out = {"model": MODEL, "n_policies": len(E10.POLS)}
    out["ledger"] = E10.e10b_ledger(eps, dist_s, lm)
    print("ledger", json.dumps(out["ledger"]["per_policy"], indent=1), flush=True)
    out["state_qa"] = E10.e10c_state_qa(eps, dist_s, lm)
    print("state_qa", json.dumps(out["state_qa"], indent=1), flush=True)
    d, rows = E10.e10d_answer_recovery(trajs, idx, dist_r, lm)
    out["answer_recovery"] = d
    print("answer_recovery", json.dumps(d, indent=1), flush=True)
    out["runtime_s"] = round(time.time() - t0, 1)

    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    prev[TAG] = out
    json.dump(prev, open(OUT, "w"), indent=1)
    json.dump(rows, open(os.path.join(RES, f"e14_rows_{TAG}.json"), "w"),
              indent=1)
    print(f"E14a[{TAG}] done in {out['runtime_s']}s -> {OUT}")


if __name__ == "__main__":
    run()
