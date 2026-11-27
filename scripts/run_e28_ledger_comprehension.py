"""E28: the ledger-comprehension probe re-run on the fixed pipeline.

E10b (Qwen2.5-3B) and E14a (Qwen2.5-7B) asked a resumed model whether an
action recorded in the shipped state had been executed (and, as a control,
whether a not-yet-executed action had been).  Both ran before the entity
tie-ordering fix (AUDIT G.1.3).  This script re-runs exactly that arm -- same
prompt, cases, policies and greedy decoding (``run_e10_llm.e10b_ledger``) --
with blocks rendered by the fixed pipeline, and writes a separate results file
so the earlier outputs stay inspectable.

``RELAY_E28_MODELS``  comma-separated subset of ``3b,7b`` (default both)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import run_e10_llm as E10                                          # noqa: E402
from explib import load_all, RES                                   # noqa: E402
from state.schema import Distiller                                 # noqa: E402

OUT = os.path.join(RES, "e28_ledger_comprehension.json")
#: model, policies, number of cases -- as in the original E10b (80 cases) and
#: E14a (48 cases, per results/e14_stronger_model.json) runs
MODELS = {"3b": ("Qwen/Qwen2.5-3B-Instruct",
                 ["full", "recency", "relay", "relay-lite", "extractive",
                  "trunc25", "randdrop", "restart"], 80),
          "7b": ("Qwen/Qwen2.5-7B-Instruct",
                 ["full", "recency", "typed-recency", "relay", "relay-lite",
                  "extractive", "trunc25", "randdrop", "restart"], 48)}


def main():
    want = os.environ.get("RELAY_E28_MODELS", "3b,7b").split(",")
    trajs, idx, eps, synth, idx_s = load_all()
    dist_s = Distiller(idx_s)
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for tag in want:
        name, pols, n_cases = MODELS[tag]
        t0 = time.time()
        E10.POLS = pols
        E10.N_LEDGER = n_cases
        lm = E10.LM(name)
        res = E10.e10b_ledger(eps, dist_s, lm)
        res["runtime_s"] = round(time.time() - t0, 1)
        out[tag] = res
        json.dump(out, open(OUT, "w"), indent=1)
        print(tag, json.dumps(res["per_policy"]), flush=True)
        del lm
    print("->", OUT)


if __name__ == "__main__":
    main()
