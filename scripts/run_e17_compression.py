"""E17: the compression comparison, run properly.

The first version cited LLMLingua and LongLLMLingua as the closest related
work and reported neither, running only Selective-Context and TextRank.  That
is the wrong two to stop at: TextRank is query-independent extractive
summarisation from 2004, and Selective-Context prunes by self-information
without ever looking at the task.  Both are weak against a *typed* policy for
a reason that has nothing to do with typing -- they do not know what the task
is.  A comparison that only beats query-blind baselines has not tested the
claim.

This experiment adds the two comparators that close that gap, on one common
stratified subsample with every policy under the identical budget:

  ``llmlingua``    LongLLMLingua's distinctive mechanism -- question-aware
      *contrastive* pruning, keeping units the task makes less surprising.
      We implement the mechanism, not the released system (no budget
      controller, no document reordering), and label it with an asterisk
      everywhere it appears.
  ``llm-summary``  a capable model given Relay's own field list as an
      instruction and asked to write the resumption state.  This is the
      strongest fair comparator for the schema claim: it is *told* what to
      keep, so any remaining gap is attributable to deterministic
      construction rather than to Relay knowing what matters.  It is also the
      baseline with the worst deployment story, and the experiment prices
      that too -- construction latency on the source, at the moment of a
      handoff.

Reported alongside them: ``typed-recency``, which reserves the schema's
mandatory core and spends the rest on recency.  Between it and
``llm-summary`` the two halves of the claim are separated -- what the schema
is worth, and what deterministic allocation is worth on top of it.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from explib import (load_all, sample_positions, real_fidelity, boot_ci,   # noqa
                    policy_budget, RES)
from stats import compare                                                # noqa
from state.schema import Distiller                                       # noqa
from policies.policies import POLICIES, relay_budget                     # noqa
from policies.baselines import (p7_extractive, p8_selective,             # noqa
                                p10_llmlingua, p11_llm_summary,
                                SelfInfoScorer)

OUT = os.path.join(RES, "e17_compression.json")
N = int(os.environ.get("RELAY_E17_N", "120"))
SUMMARY_MODEL = os.environ.get("RELAY_E17_SUMMARY_MODEL",
                               "Qwen/Qwen2.5-7B-Instruct")
DST = "watch"
DETERMINISTIC = ["full", "recency", "typed-recency", "relay", "relay-lite",
                 "randdrop", "trunc25", "restart"]


def run():
    t0 = time.time()
    trajs, idx, eps, synth, idx_s = load_all()
    dist = Distiller(idx)
    b = relay_budget(DST)

    cases = []
    for traj in trajs:
        for k in sample_positions(traj.n, traj.tid, 0, 1):
            cases.append((traj, k))
    cases.sort(key=lambda c: (c[0].tid, c[1]))
    stride = max(1, len(cases) // N)
    cases = cases[::stride][:N]

    acc, strict, tok, cpu = (defaultdict(list), defaultdict(list),
                             defaultdict(list), defaultdict(list))

    rows = []

    def _record(p, traj, k, o, A):
        r, rs, _ = real_fidelity(idx, traj, k, o.resume_text, anchors=A)
        rows.append({"policy": p, "tid": traj.tid, "k": k,
                     "prefix_tokens": traj.prefix_tokens(k),
                     "out_tokens": o.tokens_resume, "recall": round(r, 4),
                     "n_anchors": len(A)})
        acc[p].append(r)
        strict[p].append(rs)
        tok[p].append(o.tokens_resume)
        cpu[p].append(o.serialize_cpu_s)

    # --- deterministic policies + TextRank (no model needed) ---
    for traj, k in cases:
        A = idx.future_relevant(traj, k)
        for p in DETERMINISTIC:
            kw = {"distiller": dist} if p.startswith("relay") else {}
            _record(p, traj, k, POLICIES[p](traj, k, DST, seed=0,
                                            budget=policy_budget(p, DST),
                                            **kw), A)
        _record("extractive", traj, k, p7_extractive(traj, k, DST), A)
    print("deterministic done", flush=True)

    # --- learned summariser told to emit the schema ---
    import run_e10_llm as E10
    lm = E10.LM(SUMMARY_MODEL)
    for traj, k in cases:
        A = idx.future_relevant(traj, k)
        _record("llm-summary", traj, k,
                p11_llm_summary(traj, k, DST, lm, budget=b), A)
    print("llm-summary done", flush=True)
    # Free the summariser before the scorer loads. Order matters here and the
    # reason is empirical: running the 7B *after* the 0.6B scorer had made two
    # forward passes over every prefix on the same accelerator produced
    # systematically shorter generations (88 vs 237 mean output tokens) than
    # the identical call on the identical cases in a clean process, which
    # silently understated this baseline by an order of magnitude. The arms are
    # independent, so we run the one that is sensitive to accelerator state
    # first and release it before loading the other.
    del lm

    # --- self-information baselines (one small LM, two passes each) ---
    scorer = SelfInfoScorer()
    for traj, k in cases:
        A = idx.future_relevant(traj, k)
        _record("selective", traj, k, p8_selective(traj, k, DST, scorer,
                                                   budget=b), A)
        _record("llmlingua", traj, k, p10_llmlingua(traj, k, DST, scorer,
                                                    budget=b), A)
    scorer_name = scorer.model_name
    del scorer
    print("self-information baselines done", flush=True)

    json.dump(rows, open(os.path.join(RES, "e17_rows.json"), "w"), indent=1)
    pols = list(acc)
    out = {
        "n": len(cases), "budget_tokens": b, "target": DST,
        "scorer_model": scorer_name, "summary_model": SUMMARY_MODEL,
        "recall": {p: [round(v, 4) for v in boot_ci(acc[p])] for p in pols},
        "recall_strict": {p: round(float(np.mean(strict[p])), 4)
                          for p in pols},
        "tokens": {p: round(float(np.mean(tok[p])), 1) for p in pols},
        "construction_s_median": {p: round(float(np.median(cpu[p])), 4)
                                  for p in pols},
        "paired_vs_relay": compare(acc["relay"],
                                   {p: acc[p] for p in pols if p != "relay"}),
        "runtime_s": round(time.time() - t0, 1),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n{'policy':16s} {'recall':>8s} {'strict':>8s} {'tokens':>8s} "
          f"{'build s':>9s}")
    for p in sorted(pols, key=lambda x: -out["recall"][x][0]):
        print(f"{p:16s} {out['recall'][p][0]:8.4f} "
              f"{out['recall_strict'][p]:8.4f} {out['tokens'][p]:8.0f} "
              f"{out['construction_s_median'][p]:9.4f}")
    print("\npaired vs relay:")
    for r in out["paired_vs_relay"]:
        print(f"  {r['vs']:16s} delta={r['delta']:+.4f} p={r['p_holm']:.2g} "
              f"cliff={r['cliffs_delta']:+.3f} ({r['effect']})")
    print("->", OUT)


if __name__ == "__main__":
    run()
