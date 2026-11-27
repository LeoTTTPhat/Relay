"""Re-measure E10a (on-device prefill throughput) on an idle machine.

E10a is the one number in the artifact that is sensitive to what else the
host is doing: run under load it reads several times slow.  ``run_e10_llm``
measures it inline for convenience; this script re-measures it alone and
rewrites the ``E10a_prefill`` key of ``results/e10_llm.json`` so the
published figure is the uncontended one.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from explib import RES                                            # noqa: E402
from run_e10_llm import LM, e10a_prefill                          # noqa: E402


def main():
    lm = LM()
    pts = e10a_prefill(lm)
    path = os.path.join(RES, "e10_llm.json")
    d = json.load(open(path)) if os.path.exists(path) else {}
    d["E10a_prefill"] = pts
    d["E10a_prefill"]["note"] = ("measured alone on an otherwise idle host; "
                                 "laptop-class M-series, not a wearable")
    json.dump(d, open(path, "w"), indent=1)
    print(json.dumps(pts["points"], indent=1))


if __name__ == "__main__":
    main()
