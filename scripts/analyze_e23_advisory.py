"""Why advisory re-execution differs by policy (E23/E23b log analysis).

Replays advisory traces and reports, per model and policy:
  ``dup_pre``            a pre-handoff action performed again;
  ``dup_post``           a post-handoff action performed twice;
  ``first_effect_done``  the first world-changing choice is already done;
  ``first_effect_is_first_action``  it is the episode's very first action.
Output: ``results/e23_advisory_analysis.json``.
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import RES                                              # noqa: E402
from run_e23_continuation import build_world                        # noqa: E402
from synth.generator import generate                                # noqa: E402


def first_action_order(e):
    order = []
    for st in e.states:
        for se in st.side_effects:
            if se["action_id"] not in order:
                order.append(se["action_id"])
    return order


def analyse(rows, E):
    st = defaultdict(Counter)
    for r in rows:
        if r["mode"] != "advisory":
            continue
        e = E[r["eid"]]
        items, done, req = build_world(e, r["k"])
        by = {it["mid"]: it for it in items}
        executed = Counter()
        pre = post = False
        effects = []
        for ch in r["trace"]:
            it = by.get(ch)
            if not it or it["kind"] != "effect":
                continue
            effects.append(it["id"])
            executed[it["id"]] += 1
            if it["id"] in done:
                pre = True
            elif executed[it["id"]] >= 2:
                post = True
        s = st[r["policy"]]
        s["n"] += 1
        s["dup_pre"] += pre
        s["dup_post"] += post
        if effects:
            s["n_with_effect"] += 1
            s["first_effect_done"] += effects[0] in done
            s["first_effect_is_first_action"] += \
                effects[0] == first_action_order(e)[0]
    out = {}
    for p, s in st.items():
        n, ne = s["n"], max(s["n_with_effect"], 1)
        out[p] = {"n": n, "dup_pre": round(s["dup_pre"] / n, 4),
                  "dup_post": round(s["dup_post"] / n, 4),
                  "first_effect_done": round(s["first_effect_done"] / ne, 4),
                  "first_effect_is_first_action":
                      round(s["first_effect_is_first_action"] / ne, 4),
                  "n_with_effect": s["n_with_effect"]}
    return out


def main():
    E = {e.eid: e for e in generate(200, seed=0)}
    out = {}
    for label, fname in (("e23_3b", "e23_rows_Qwen2.5-3B-Instruct.json"),
                         ("e23_7b", "e23_rows_Qwen2.5-7B-Instruct.json"),
                         ("e23b_3b", "e23b_rows_Qwen2.5-3B-Instruct.json"),
                         ("e23b_7b", "e23b_rows_Qwen2.5-7B-Instruct.json"),
                         ("e23b_3b_planlabel",
                          "e23b_rows_Qwen2.5-3B-Instruct-planlabel.json")):
        p = os.path.join(RES, fname)
        if os.path.exists(p):
            out[label] = analyse(json.load(open(p)), E)
    json.dump(out, open(os.path.join(RES, "e23_advisory_analysis.json"), "w"),
              indent=1)
    for label, d in out.items():
        print(label)
        for pol, v in d.items():
            print("  ", pol, v)


if __name__ == "__main__":
    main()
