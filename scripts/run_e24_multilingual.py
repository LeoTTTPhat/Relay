"""E24: the extractor, the allocator and the ledger on machine-translated traces.

Scores the trajectories translated by ``gen_e24_translate.py`` (German,
Spanish, French; Helsinki-NLP opus-mt) against the same English subset.
Everything downstream is the unchanged pipeline: the parser recomputes
side-effect-like labels on the translated text with the English-authored
patterns, the anchor index and document-frequency cut are rebuilt on each
language's subset (and on the English subset the same way), and every policy
and metric is the one used for the main frontier.

Reported per language, paired with English on identical (trajectory, k):

  * future-relevance recall for full context, recency, typed-recency,
    random-drop and Relay at the watch budget, all anchor types and the
    high-precision url/file/number/answer subset;
  * Relay's paired lead over recency (Wilcoxon, Cliff's delta);
  * **state-field transfer**: constraints extracted from the task statement,
    presence of a pending plan, and -- the safety-relevant one -- ledger
    completeness: the fraction of prefix steps labelled side-effect-like in
    English that still receive a ledger entry once translated.

These are machine-translated traces, not native multilingual agent logs.
"""
import gzip
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from explib import (sample_positions, real_fidelity, boot_ci,        # noqa: E402
                    policy_budget, RES)
from gen_e24_translate import translate_text, MTDIR                  # noqa: E402
from stats import compare                                           # noqa: E402
from traces.parser import load_trajectories, _mk_step, Trajectory    # noqa: E402
from traces.anchors import AnchorIndex, build_df, DF_MAX_FRAC      # noqa: E402
from state.schema import Distiller, build_side_fx, extract_constraints  # noqa
from policies.policies import POLICIES                               # noqa: E402

OUT = os.path.join(RES, "e24_multilingual.json")
SEEDS = [0, 1, 2, 3, 4]
POLS = ["full", "recency", "typed-recency", "randdrop", "relay"]
DST = "watch"


def translated(trajs, table, keep_tool_status=False):
    out = []
    for t in trajs:
        steps = [_mk_step(s.idx, s.role, s.agent,
                          translate_text(s.content, table, keep_tool_status))
                 for s in t.steps]
        out.append(Trajectory(tid=t.tid, source=t.source,
                              question=translate_text(t.question, table,
                                                      keep_tool_status),
                              ground_truth=t.ground_truth, steps=steps))
    return out


def score(trajs, en_trajs):
    # Document frequencies are computed on *this* subset.  ``AnchorIndex``
    # would silently load the English full-corpus DF cache, which is not the
    # construction this experiment states (first scoring runs did that; the
    # outputs were discarded -- AUDIT G).
    idx = AnchorIndex.__new__(AnchorIndex)
    idx.df = build_df(trajs, cache=False)
    idx.n_trajs = len(trajs)
    idx.df_cut = DF_MAX_FRAC * len(trajs)
    dist = Distiller(idx)
    rec = defaultdict(list)
    strict = defaultdict(list)
    n_anchor = []
    ledger_keep, cons_n, plan_on, flags_kept, flags_new = [], [], [], [], []
    for t, te in zip(trajs, en_trajs):
        cons_n.append(len(extract_constraints(t.question)))
        en_flag = [s.side_effect_like for s in te.steps]
        tr_flag = [s.side_effect_like for s in t.steps]
        flags_kept += [float(b) for a, b in zip(en_flag, tr_flag) if a]
        flags_new += [float(b) for a, b in zip(en_flag, tr_flag) if not a]
        for seed in SEEDS:
            for k in sample_positions(len(t.steps), t.tid, seed):
                if k >= len(t.steps):
                    continue
                A = idx.future_relevant(t, k)
                n_anchor.append(len(A))
                for p in POLS:
                    kw = {"distiller": dist} if p.startswith("relay") else {}
                    o = POLICIES[p](t, k, DST, seed=seed,
                                    budget=policy_budget(p, DST), **kw)
                    r, rs, _ = real_fidelity(idx, t, k, o.resume_text,
                                             anchors=A)
                    rec[p].append(r)
                    strict[p].append(rs)
                en_n = sum(1 for s in te.steps[:k + 1] if s.side_effect_like)
                if en_n:
                    kept = sum(1 for s, se in zip(t.steps[:k + 1],
                                                  te.steps[:k + 1])
                               if se.side_effect_like and s.side_effect_like)
                    ledger_keep.append(kept / en_n)
                b = dist.distill(t, k, policy_budget("relay", DST))
                plan_on.append(float(bool(b.plan)))
    res = {"n_handoffs": len(rec["relay"]),
           "mean_anchors": round(float(np.mean(n_anchor)), 2),
           "recall": {p: [round(x, 4) for x in boot_ci(v)]
                      for p, v in rec.items()},
           "recall_strict": {p: round(float(np.mean(v)), 4)
                             for p, v in strict.items()},
           "relay_vs": compare(rec["relay"], {b: rec[b] for b in POLS
                                              if b != "relay"}),
           "constraints_per_task": round(float(np.mean(cons_n)), 3),
           "tasks_with_constraints": round(float(np.mean(
               [c > 0 for c in cons_n])), 4),
           "plan_present": round(float(np.mean(plan_on)), 4),
           "ledger_entries_kept_vs_en": [round(x, 4) for x in boot_ci(
               ledger_keep)] if ledger_keep else None,
           "n_handoffs_with_en_entries": len(ledger_keep),
           "step_label_recall_vs_en": round(float(np.mean(flags_kept)), 4)
           if flags_kept else None,
           "step_label_new_vs_en": round(float(np.mean(flags_new)), 4)
           if flags_new else None}
    return res, rec


def run():
    t0 = time.time()
    meta = json.load(open(os.path.join(MTDIR, "meta.json")))
    allt = {t.tid: t for t in load_trajectories()}
    en = [allt[tid] for tid in meta["tids"]]
    out = {"n_traces": len(en), "mt_meta": meta, "languages": {}}
    en_res, en_rec = score(en, en)
    out["languages"]["en"] = en_res
    print("[en]", {p: v[0] for p, v in en_res["recall"].items()}, flush=True)
    out["tool_status_verbatim"] = {}
    for lang in meta["langs"]:
        table = json.load(gzip.open(os.path.join(MTDIR,
                                                 f"table_{lang}.json.gz"), "rt"))
        for variant, keep in (("languages", False),
                              ("tool_status_verbatim", True)):
            tr = translated(en, table, keep_tool_status=keep)
            res, rec = score(tr, en)
            res["relay_drop_vs_en"] = compare(rec["relay"],
                                              {"en": en_rec["relay"]})
            res["recency_drop_vs_en"] = compare(rec["recency"],
                                                {"en": en_rec["recency"]})
            out[variant][lang] = res
            print(f"[{lang}|{variant}]",
                  {p: v[0] for p, v in res["recall"].items()},
                  "ledger kept", res["ledger_entries_kept_vs_en"],
                  "labels", res["step_label_recall_vs_en"],
                  "constraints", res["tasks_with_constraints"],
                  "plan", res["plan_present"], flush=True)
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print("->", OUT, out["runtime_s"], "s")


if __name__ == "__main__":
    run()
