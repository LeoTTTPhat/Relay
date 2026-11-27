"""Paired exact McNemar tests for E23b (binding budget).

Per model: enforced vs advisory success for every policy; and, under each
regime, Relay vs every other policy on success and on duplicates.
Output: results/e23b_paired.json.
"""
import json
import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(__file__))
from explib import RES                                              # noqa: E402


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    out = {}
    for tag in ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"):
        path = os.path.join(RES, f"e23b_rows_{tag}.json")
        if not os.path.exists(path):
            continue
        rows = json.load(open(path))
        by = {(r["eid"], r["policy"], r["mode"]): r for r in rows}
        pols = sorted({r["policy"] for r in rows})
        eids = sorted({r["eid"] for r in rows})
        res = {"enforced_vs_advisory_success": {}, "relay_vs": {}}
        for p in pols:
            b = sum(1 for e in eids if by[(e, p, "enforced")]["success"]
                    and not by[(e, p, "advisory")]["success"])
            c = sum(1 for e in eids if by[(e, p, "advisory")]["success"]
                    and not by[(e, p, "enforced")]["success"])
            res["enforced_vs_advisory_success"][p] = {
                "n": len(eids), "enforced_only": b, "advisory_only": c,
                "p_exact": mcnemar_exact(b, c)}
        for q in pols:
            if q == "relay":
                continue
            for mode in ("advisory", "enforced"):
                for fld in ("success", "duplicate", "violation"):
                    b = sum(1 for e in eids if by[(e, "relay", mode)][fld]
                            and not by[(e, q, mode)][fld])
                    c = sum(1 for e in eids if by[(e, q, mode)][fld]
                            and not by[(e, "relay", mode)][fld])
                    res["relay_vs"][f"{q}|{mode}|{fld}"] = {
                        "relay_only": b, "other_only": c,
                        "p_exact": mcnemar_exact(b, c)}
        out[tag] = res
    # Diagnostic of AUDIT H.5: same block, plan labelled as written and tail
    # oldest-first, against the unchanged rendering on the same episodes.
    tag = "Qwen2.5-3B-Instruct"
    dp = os.path.join(RES, f"e23b_rows_{tag}-planlabel.json")
    mp = os.path.join(RES, f"e23b_rows_{tag}.json")
    if os.path.exists(dp) and os.path.exists(mp):
        base = {(r["eid"], r["mode"]): r for r in json.load(open(mp))
                if r["policy"] == "relay"}
        diag = {(r["eid"], r["mode"]): r for r in json.load(open(dp))
                if r["policy"] == "relay-planlabel"}
        keys = sorted(set(base) & set(diag))
        res = {"n_episodes": len({k[0] for k in keys})}
        for mode in ("advisory", "enforced"):
            ks = [k for k in keys if k[1] == mode]
            for fld in ("success", "duplicate", "dup_pre", "dup_post",
                        "violation", "complete"):
                b = sum(1 for k in ks if diag[k][fld] and not base[k][fld])
                c = sum(1 for k in ks if base[k][fld] and not diag[k][fld])
                res[f"{mode}|{fld}"] = {
                    "planlabel": round(sum(diag[k][fld] for k in ks)
                                       / max(len(ks), 1), 4),
                    "relay": round(sum(base[k][fld] for k in ks)
                                   / max(len(ks), 1), 4),
                    "planlabel_only": b, "relay_only": c,
                    "p_exact": mcnemar_exact(b, c)}
        out["planlabel_vs_relay_3b"] = res
    json.dump(out, open(os.path.join(RES, "e23b_paired.json"), "w"), indent=1)
    for tag, r in out.items():
        print(tag)
        if tag == "planlabel_vs_relay_3b":
            for k, v in r.items():
                print("  ", k, v)
            continue
        for p, v in r["enforced_vs_advisory_success"].items():
            print("  enf vs adv", p, v)
        for k, v in r["relay_vs"].items():
            if k.split("|")[2] == "success" or k.split("|")[2] == "duplicate":
                print("  relay vs", k, v)


if __name__ == "__main__":
    main()
