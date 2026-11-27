"""Paired tests for E23: does enforcement change task success per episode?

For each model and policy, pairs the advisory and enforced outcome of the
same handoff and applies an exact two-sided McNemar test (binomial test on the
discordant pairs).  Also compares Relay with full context under each regime,
paired by episode.  Output: ``results/e23_paired.json``.
"""
import json
import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(__file__))
from explib import RES                                              # noqa: E402


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    out = {}
    for tag in ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"):
        path = os.path.join(RES, f"e23_rows_{tag}.json")
        if not os.path.exists(path):
            continue
        rows = json.load(open(path))
        by = {(r["eid"], r["policy"], r["mode"]): r for r in rows}
        pols = sorted({r["policy"] for r in rows})
        res = {"enforced_vs_advisory": {}, "relay_vs_full": {}}
        for p in pols:
            eids = sorted({r["eid"] for r in rows if r["policy"] == p})
            b = sum(1 for e in eids if by[(e, p, "enforced")]["success"]
                    and not by[(e, p, "advisory")]["success"])
            c = sum(1 for e in eids if by[(e, p, "advisory")]["success"]
                    and not by[(e, p, "enforced")]["success"])
            res["enforced_vs_advisory"][p] = {
                "n": len(eids), "enforced_only": b, "advisory_only": c,
                "p_exact": round(mcnemar_exact(b, c), 6)}
        for mode in ("advisory", "enforced"):
            eids = sorted({r["eid"] for r in rows if r["policy"] == "relay"})
            b = sum(1 for e in eids if by[(e, "relay", mode)]["success"]
                    and not by[(e, "full", mode)]["success"])
            c = sum(1 for e in eids if by[(e, "full", mode)]["success"]
                    and not by[(e, "relay", mode)]["success"])
            res["relay_vs_full"][mode] = {"n": len(eids), "relay_only": b,
                                          "full_only": c,
                                          "p_exact": round(mcnemar_exact(b, c), 6)}
        out[tag] = res
    json.dump(out, open(os.path.join(RES, "e23_paired.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
