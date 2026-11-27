"""Why did E17's llm-summary column disagree with a direct measurement?

E17 reported 0.097 recall on 88 output tokens for the learned summariser; a
standalone run of the *same* function over the *same* cases with the *same*
greedy decoding reported 0.674 on 243 tokens. With identical inputs and no
sampling those cannot both be right, so something about E17's execution
context differs. The one structural difference is that E17 runs the
self-information baselines first: it builds a ``SelfInfoScorer`` (a second
model on the same accelerator), uses it, deletes it, and only then loads the
summariser.

This script runs the same six cases twice in one process -- once cold, once
after standing up and deleting a ``SelfInfoScorer`` exactly as E17 does -- and
prints both. If the two agree, E17's context is exonerated and the fault is
elsewhere; if they disagree, the experiment has an ordering dependency and its
llm-summary column cannot be reported as measured.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, sample_positions, real_fidelity, RES  # noqa
from policies.baselines import p11_llm_summary, SelfInfoScorer     # noqa
from policies.policies import relay_budget                         # noqa

DST = "watch"
N = 72


def measure(lm, cases, idx, b, label):
    rows = []
    for traj, k in cases:
        A = idx.future_relevant(traj, k)
        o = p11_llm_summary(traj, k, DST, lm, budget=b)
        r, _, _ = real_fidelity(idx, traj, k, o.resume_text, anchors=A)
        rows.append({"tid": traj.tid, "k": k, "out_tokens": o.tokens_wire,
                     "recall": round(r, 4)})
        print(f"  [{label}] {traj.tid:14s} k={k:3d} out={o.tokens_wire:4d} "
              f"recall={r:.3f}", flush=True)
    return rows


def run():
    trajs, idx, eps, synth, idx_s = load_all()
    b = relay_budget(DST)
    cases = []
    for traj in trajs:
        for k in sample_positions(traj.n, traj.tid, 0, 1):
            cases.append((traj, k))
    cases.sort(key=lambda c: (c[0].tid, c[1]))
    cases = cases[::max(1, len(cases) // N)][:N][:6]

    import run_e10_llm as E10
    print("== A: cold (no scorer built first) ==", flush=True)
    lm = E10.LM("Qwen/Qwen2.5-7B-Instruct")
    a = measure(lm, cases, idx, b, "cold")
    del lm

    print("== B: after a SelfInfoScorer, as E17 does ==", flush=True)
    scorer = SelfInfoScorer()
    _ = scorer.token_self_info("warm up the scorer on a little text")
    del scorer
    lm2 = E10.LM("Qwen/Qwen2.5-7B-Instruct")
    bb = measure(lm2, cases, idx, b, "after-scorer")

    same = all(x["out_tokens"] == y["out_tokens"]
               for x, y in zip(a, bb))
    out = {"identical_outputs": same, "cold": a, "after_scorer": bb}
    json.dump(out, open(os.path.join(RES, "diag_e17_context.json"), "w"),
              indent=1)
    print(f"\nidentical output lengths across the two orders: {same}")


if __name__ == "__main__":
    run()
