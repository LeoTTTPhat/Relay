"""Revision 2026-11: prototype handoff timing against human response-time
constants (CPU only; reads stored results, runs nothing).

Source: ``results/e26_prototype.json`` (the same-host two-process emulation:
a GPU source process and a watch-class target process, Qwen2.5-0.5B on two CPU
threads, loopback TCP, no radio).  Per handoff we add up the stored stage
times:

  protocol   = fence_snapshot + distil_serialise + seal + send_verify_install_ack
  to_first_decision = protocol + target prefill of the received block
                      + generation of the target's first choice

and compare them with the time constants of Card, Robertson and Mackinlay
(CHI 1991): about 0.1 s (perceptual processing), about 1 s (immediate
response) and about 10 s (unit task).  These are emulation timings on one
workstation, not device measurements.

Output: ``results_rev2026-11/prototype_timing.json``.
"""
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(ROOT, "results", "e26_prototype.json")
OUT_DIR = os.path.join(ROOT, "results_rev2026-11")


def main():
    d = json.load(open(SRC))
    rows = []
    for e in d["episodes"]:
        ms = e["ms"]
        t = e["target"]
        protocol_s = sum(ms.values()) / 1000.0
        first_gen = t["turns"][0]["gen_s"] if t["turns"] else 0.0
        rows.append({"task": e["task"],
                     "protocol_s": round(protocol_s, 4),
                     "target_prefill_s": t["prefill_s"],
                     "target_first_gen_s": first_gen,
                     "to_first_decision_s": round(
                         protocol_s + t["prefill_s"] + first_gen, 4)})

    def summ(key):
        v = [r[key] for r in rows]
        return {"median": round(st.median(v), 4), "min": round(min(v), 4),
                "max": round(max(v), 4)}

    out = {"source": "results/e26_prototype.json",
           "n_handoffs": len(rows),
           "protocol_s": summ("protocol_s"),
           "target_prefill_s": summ("target_prefill_s"),
           "target_first_gen_s": summ("target_first_gen_s"),
           "to_first_decision_s": summ("to_first_decision_s"),
           "protocol_under_0.1s": sum(r["protocol_s"] < 0.1 for r in rows),
           "to_first_decision_under_1s": sum(
               r["to_first_decision_s"] < 1.0 for r in rows),
           "to_first_decision_under_10s": sum(
               r["to_first_decision_s"] < 10.0 for r in rows),
           "rows": rows}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "prototype_timing.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
