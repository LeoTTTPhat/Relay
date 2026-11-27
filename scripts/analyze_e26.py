"""Attribute E26 constraint violations to the runtime that executed them.

The service's effect log records *that* the distractor ran, not *which*
runtime sent it; the source and target logs say who dispatched it.  Output:
``results/e26_attribution.json`` (read by ``fill_prose.py``).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from explib import RES                                              # noqa: E402
from run_e23_continuation import build_world                        # noqa: E402
from synth.generator import generate                                # noqa: E402


def main():
    eps = {e.eid: e for e in generate(200, seed=0)}
    d = json.load(open(os.path.join(RES, "e26_prototype.json")))
    rows, src, tgt = [], 0, 0
    for r in d["episodes"]:
        items, _, _ = build_world(eps[r["task"]], 0)
        dm = next((it["mid"] for it in items if it["kind"] == "distractor"),
                  None)
        by_src = bool(dm) and dm in r["source_executed_mids"]
        by_tgt = bool(dm) and any(v["mid"] == dm and v["dispatched"]
                                  for v in r["target"]["verdicts"])
        src += by_src
        tgt += by_tgt
        rows.append({"task": r["task"], "distractor_mid": dm,
                     "by_source": by_src, "by_target": by_tgt,
                     "violation_logged": r["violation"]})
    out = {"violations_by_source": src, "violations_by_target": tgt,
           "violations_logged": sum(r["violation"] for r in d["episodes"]),
           "rows": rows}
    json.dump(out, open(os.path.join(RES, "e26_attribution.json"), "w"),
              indent=1)
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
