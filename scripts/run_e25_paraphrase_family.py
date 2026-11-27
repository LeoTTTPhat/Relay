"""E25: the out-of-schema corpus re-worded by a different generator.

Scores the LLM-written paraphrase family produced by
``gen_e25_paraphrases.py`` with the unchanged E14b code
(``run_e14_independence.e14b_out_of_schema``) and the unchanged distiller.
Nothing in ``state/schema.py`` was edited after the paraphrases were
generated.

For each condition (``natural``: free rewrite; ``disjoint``: rewrite that
avoids every cue word the template generator uses) the template sentences of
each episode are replaced by their paraphrases.  An episode is retained only
if *every* one of its sentences produced a validated paraphrase; the template
corpus is re-scored on exactly the retained episodes, so the comparison is
paired.

The paraphrases are LLM-generated (gemma3:27b), not written by independent
human authors.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from explib import RES                                               # noqa: E402
from run_e14_independence import e14b_out_of_schema                  # noqa: E402
from synth.oos import generate_oos                                   # noqa: E402
from traces.parser import est_tokens                                 # noqa: E402
from state.schema import REVISION_RE, OPEN_QUESTION_RE               # noqa: E402

IN = os.path.join(RES, "e25_paraphrases.json")
OUT = os.path.join(RES, "e25_paraphrase_family.json")


def reworded(cond, para):
    eps = generate_oos(120, seed=7)
    keep, dropped = [], 0
    cue_hits = {"revision_re": 0, "open_question_re": 0, "n": 0}
    for e in eps:
        rec = para["episodes"].get(f"{e.eid}|{cond}")
        if rec is None or any(it["paraphrase"] is None for it in rec["items"]):
            dropped += 1
            continue
        for it in rec["items"]:
            s = e.steps[it["step"]]
            assert it["replace"] in s.content, (e.eid, it["slot"])
            s.content = s.content.replace(it["replace"], it["paraphrase"])
            s.tokens = est_tokens(s.content)
            cue_hits["n"] += 1
            if it["slot"] in ("goal_revision", "constraint_correction",
                              "value_correction"):
                cue_hits["revision_re"] += int(bool(
                    REVISION_RE.search(it["paraphrase"])))
            if it["slot"] == "open_question":
                cue_hits["open_question_re"] += int(bool(
                    OPEN_QUESTION_RE.search(it["paraphrase"])))
        keep.append(e)
    return keep, dropped, cue_hits


def template_subset(eids):
    return [e for e in generate_oos(120, seed=7) if e.eid in eids]


def run():
    para = json.load(open(IN))
    out = {"generator_model": para.get("model"), "n_failed_items":
           para.get("n_failed"), "n_items": para.get("n_items"),
           "conditions": {}}
    for cond in ("natural", "disjoint"):
        eps, dropped, cues = reworded(cond, para)
        ids = {e.eid for e in eps}
        n_rev = sum(1 for e in eps for it in
                    para["episodes"][f"{e.eid}|{cond}"]["items"]
                    if it["slot"] in ("goal_revision", "constraint_correction",
                                      "value_correction"))
        n_open = sum(1 for e in eps for it in
                     para["episodes"][f"{e.eid}|{cond}"]["items"]
                     if it["slot"] == "open_question")
        res = e14b_out_of_schema(eps=eps)
        base = e14b_out_of_schema(eps=template_subset(ids))
        out["conditions"][cond] = {
            "n_retained": len(eps), "n_dropped": dropped,
            "revision_cue_detected": round(cues["revision_re"] / max(n_rev, 1), 4),
            "open_question_cue_detected": round(
                cues["open_question_re"] / max(n_open, 1), 4),
            "paraphrased": res, "template_same_episodes": base}
        lr, lb = res["live_state_recall"], base["live_state_recall"]
        print(f"[{cond}] retained {len(eps)} dropped {dropped}; revision cue "
              f"hit {out['conditions'][cond]['revision_cue_detected']}, "
              f"open-q cue hit {out['conditions'][cond]['open_question_cue_detected']}")
        for p in ("full", "recency", "typed-recency", "relay", "randdrop"):
            print(f"   {p:14s} live {lb[p]['mean']:.3f} -> {lr[p]['mean']:.3f}"
                  f"   stale-alone {base['superseded_shipped_alone'][p]} -> "
                  f"{res['superseded_shipped_alone'][p]}")
        print("   relay per-field", base["per_field"]["relay"], "->",
              res["per_field"]["relay"])
    json.dump(out, open(OUT, "w"), indent=1)
    print("->", OUT)


if __name__ == "__main__":
    run()
