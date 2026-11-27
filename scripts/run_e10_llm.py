"""E10: validating the deterministic proxies with a real on-device-class model.

The paper's fidelity and safety metrics are deliberately LLM-free: they are
string-level properties of what the handoff shipped.  That buys
reproducibility and costs external validity -- a reader may reasonably ask
whether a resumed *model* behaves the way the string metrics predict.  E10
answers that once, off the critical path, with a locally-run
Qwen2.5-3B-Instruct (the size class the phone/speaker profiles model),
greedy decoding, and exact-ground-truth grading.  It validates the proxies;
it does not replace them.

  E10a  measured prefill throughput of local 1.5B/3B models on this host, so
        the modeled per-class prefill rates have measured anchor points
        rather than being literature-only;
  E10b  *ledger comprehension*: does a real resumed model actually avoid
        re-executing recorded actions?  For each handoff the model is asked
        about one action that was executed in the prefix (it must answer
        YES) and one action the suffix has not yet executed (it must answer
        NO), so the score is balanced accuracy and a policy cannot win by
        always answering YES;
  E10c  *state QA*: exact ground-truth questions about the synthetic
        episode's state, answered only from the shipped block, correlated
        with the deterministic probe recall;
  E10d  *answer recovery* on real traces whose ground-truth answer already
        exists in the prefix -- the pervasive case of a nearly-finished task
        -- correlated with future-relevance recall;
  E10e  Selective-Context / LLMLingua-style pruning (P8) at Relay's budget,
        on a stratified subsample, using per-token self-information.
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from explib import (load_all, sample_positions, real_fidelity, probe_recall,
                    boot_ci, RES)
from state.schema import Distiller
from policies.policies import POLICIES, usable
from policies.baselines import p7_extractive, p8_selective, SelfInfoScorer
from traces.anchors import norm_text

MODEL = os.environ.get("RELAY_E10_MODEL", "Qwen/Qwen2.5-3B-Instruct")
POLS = ["full", "recency", "relay", "relay-lite", "extractive",
        "trunc25", "randdrop", "restart"]
N_LEDGER = int(os.environ.get("RELAY_E10_LEDGER", "80"))
N_QA = int(os.environ.get("RELAY_E10_QA", "60"))
N_ANS = int(os.environ.get("RELAY_E10_ANS", "100"))
N_SEL = int(os.environ.get("RELAY_E10_SEL", "120"))
STATE_CHARS = 9000


class LM:
    def __init__(self, name=MODEL):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tk = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16).to(self.dev).eval()
        self.name = name

    def gen(self, prompt, max_new=8):
        ids = self.tk.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt").to(self.dev)
        if ids.shape[1] > 7000:
            ids = ids[:, -7000:]
        with self.torch.no_grad():
            o = self.m.generate(ids, max_new_tokens=max_new, do_sample=False,
                                pad_token_id=self.tk.eos_token_id)
        return self.tk.decode(o[0][ids.shape[1]:],
                              skip_special_tokens=True).strip()


LEDGER_P = (
    "Below is the state transferred from another device where this task was "
    "already running.\n\n--- STATE ---\n{state}\n--- END STATE ---\n\n"
    "Question: does the state above record that the action `{action}` has "
    "ALREADY been executed?\nAnswer with one word: YES or NO.")

QA_P = ("Below is the state transferred from another device where this task "
        "was already running.\n\n--- STATE ---\n{state}\n--- END STATE ---\n\n"
        "Question: {q}\nAnswer using only the state above, with the value "
        "only. If the state does not contain it, answer UNKNOWN.")

ANS_P = ("You are resuming a task that was started on another device. The "
         "state below is everything that was transferred.\n\n"
         "--- TRANSFERRED STATE ---\n{state}\n--- END STATE ---\n\n"
         "TASK: {q}\n\nUsing ONLY the transferred state, give the final "
         "answer. If the state does not contain what is needed, answer "
         "UNKNOWN. Reply with the answer value only.")


def _policy_out(p, traj, k, dist, scorer=None):
    if p == "extractive":
        return p7_extractive(traj, k, "watch")
    if p == "selective":
        return p8_selective(traj, k, "watch", scorer)
    kw = {"distiller": dist} if p.startswith("relay") else \
         ({"seed": 0} if p == "randdrop" else {})
    return POLICIES[p](traj, k, "watch", **kw)


def e10a_prefill(lm):
    torch = lm.torch
    base = "The agent observed the following tool result. " * 512
    pts = []
    for n in (256, 512, 1024, 2048, 4096):
        ids = lm.tk(base, return_tensors="pt").input_ids[:, :n].to(lm.dev)
        with torch.no_grad():
            lm.m(ids)
        if lm.dev == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(3):
                lm.m(ids)
        if lm.dev == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - t0) / 3
        pts.append({"ctx_tokens": int(ids.shape[1]), "seconds": round(dt, 4),
                    "tokens_per_s": round(float(ids.shape[1]) / dt, 1)})
    return {"model": lm.name, "device": lm.dev, "points": pts}


def e10b_ledger(eps, dist_s, lm):
    """Balanced ledger comprehension: one done action, one not-yet-done."""
    hit = defaultdict(list)
    fa = defaultdict(list)
    n = 0
    for e in eps:
        traj = e.to_trajectory()
        k = sample_positions(traj.n, traj.tid, 0, 1)[0]
        done = e.states[k].side_effects
        future = [se for se in e.states[-1].side_effects
                  if se["action_id"] not in {d["action_id"] for d in done}]
        if not done or not future:
            continue
        n += 1
        if n > N_LEDGER:
            break
        a_done = f"{done[-1]['action']}({done[-1]['args']})"
        a_future = f"{future[0]['action']}({future[0]['args']})"
        for p in POLS:
            o = _policy_out(p, traj, k, dist_s)
            st = o.resume_text[-STATE_CHARS:]
            y = lm.gen(LEDGER_P.format(state=st, action=a_done), 4)
            nn = lm.gen(LEDGER_P.format(state=st, action=a_future), 4)
            hit[p].append(float(y.upper().startswith("YES")))
            fa[p].append(float(nn.upper().startswith("YES")))
    out = {}
    for p in POLS:
        if not hit[p]:
            continue
        h = float(np.mean(hit[p]))
        f = float(np.mean(fa[p]))
        out[p] = {"recognised_done": round(h, 4),
                  "false_alarm_not_done": round(f, 4),
                  "balanced_acc": round((h + (1 - f)) / 2, 4),
                  "ci_recognised": [round(v, 4) for v in boot_ci(hit[p])],
                  "n": len(hit[p])}
    return {"n_cases": min(n, N_LEDGER), "model": lm.name, "per_policy": out}


def e10c_state_qa(eps, dist_s, lm):
    """Exact-ground-truth state questions vs the deterministic probe recall."""
    acc = defaultdict(list)
    pr = defaultdict(list)
    rows = []
    n = 0
    for e in eps:
        traj = e.to_trajectory()
        k = sample_positions(traj.n, traj.tid, 0, 1)[0]
        st_gt = e.states[k]
        qs = []
        if st_gt.tool_results:
            name, val = st_gt.tool_results[-1]
            qs.append((f"What is the {name}?", val))
        if st_gt.plan_remaining:
            qs.append(("What is the next pending step?",
                       st_gt.plan_remaining[0]))
        if not qs:
            continue
        n += 1
        if n > N_QA:
            break
        for p in POLS:
            o = _policy_out(p, traj, k, dist_s)
            stt = o.resume_text[-STATE_CHARS:]
            got = []
            for q, gt in qs:
                a = lm.gen(QA_P.format(state=stt, q=q), 24)
                got.append(float(norm_text(str(gt)) in norm_text(a)))
            probe, _ = probe_recall(e, k, o.resume_text)
            acc[p].append(float(np.mean(got)))
            pr[p].append(probe)
            rows.append({"policy": p, "qa": float(np.mean(got)),
                         "probe": probe})
    x = np.array([r["probe"] for r in rows])
    y = np.array([r["qa"] for r in rows])
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan")
    return {"n_cases": min(n, N_QA), "model": lm.name,
            "per_policy": {p: {"qa_acc": round(float(np.mean(acc[p])), 4),
                               "probe_recall": round(float(np.mean(pr[p])), 4),
                               "ci": [round(v, 4) for v in boot_ci(acc[p])]}
                           for p in POLS if acc[p]},
            "pearson_probe_vs_qa": round(corr, 4)}


def e10d_answer_recovery(trajs, idx, dist, lm):
    cases = []
    for traj in trajs:
        gt = norm_text(traj.ground_truth)
        if not (gt and 2 <= len(gt) <= 120):
            continue
        for seed in (0, 1, 2):
            for k in sample_positions(traj.n, traj.tid, seed, 2):
                if gt in norm_text(traj.prefix_text(k)):
                    cases.append((traj, k, gt))
    seen, uniq = set(), []
    for c in cases:
        key = (c[0].tid, c[1])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    uniq.sort(key=lambda c: (c[0].tid, c[1]))
    stride = max(1, len(uniq) // N_ANS)
    cases = uniq[::stride][:N_ANS]
    per, frr, rows = defaultdict(list), defaultdict(list), []
    for traj, k, gt in cases:
        A = idx.future_relevant(traj, k)
        for p in POLS:
            o = _policy_out(p, traj, k, dist)
            a = lm.gen(ANS_P.format(state=o.resume_text[-STATE_CHARS:],
                                    q=traj.question[:1200]), 24)
            ok = gt in norm_text(a)
            r, _, _ = real_fidelity(idx, traj, k, o.resume_text, anchors=A)
            per[p].append(float(ok))
            frr[p].append(r)
            rows.append({"trace": traj.tid, "k": k, "policy": p,
                         "correct": int(ok), "frr": round(r, 4),
                         "answer": a[:120]})
    x = np.array([r["frr"] for r in rows], float)
    y = np.array([r["correct"] for r in rows], float)
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else float("nan")
    bins = []
    for lo in np.arange(0, 1.0, 0.25):
        sel = (x >= lo) & (x < lo + 0.25 + (1e-9 if lo >= 0.75 else 0))
        if sel.sum() >= 8:
            bins.append({"frr_bin": f"{lo:.2f}-{lo+0.25:.2f}",
                         "n": int(sel.sum()),
                         "acc": round(float(y[sel].mean()), 4)})
    return {"n_cases": len(cases), "n_evals": len(rows), "model": lm.name,
            "per_policy": {p: {"acc": round(float(np.mean(per[p])), 4),
                               "ci": [round(v, 4) for v in boot_ci(per[p])],
                               "frr": round(float(np.mean(frr[p])), 4)}
                           for p in POLS},
            "pearson_frr_vs_success": round(corr, 4),
            "calibration": bins}, rows


def e10e_selective(trajs, idx, dist, scorer):
    cases = []
    for traj in trajs:
        for k in sample_positions(traj.n, traj.tid, 0, 1):
            cases.append((traj, k))
    cases.sort(key=lambda c: (c[0].tid, c[1]))
    stride = max(1, len(cases) // N_SEL)
    cases = cases[::stride][:N_SEL]
    acc, strict, tok = defaultdict(list), defaultdict(list), defaultdict(list)
    pols = ["full", "recency", "relay", "relay-lite", "extractive",
            "selective", "randdrop"]
    for traj, k in cases:
        A = idx.future_relevant(traj, k)
        for p in pols:
            o = _policy_out(p, traj, k, dist, scorer)
            r, rs, _ = real_fidelity(idx, traj, k, o.resume_text, anchors=A)
            acc[p].append(r)
            strict[p].append(rs)
            tok[p].append(o.tokens_resume)
    from stats import compare
    return {"n": len(cases), "scorer_model": scorer.model_name,
            "recall": {p: [round(v, 4) for v in boot_ci(acc[p])] for p in pols},
            "recall_strict": {p: round(float(np.mean(strict[p])), 4)
                              for p in pols},
            "tokens": {p: round(float(np.mean(tok[p])), 1) for p in pols},
            "paired_vs_relay": compare(acc["relay"],
                                       {p: acc[p] for p in pols
                                        if p != "relay"})}


def run():
    t0 = time.time()
    trajs, idx, eps, synth, idx_s = load_all()
    dist_r, dist_s = Distiller(idx), Distiller(idx_s)
    out = {}

    scorer = SelfInfoScorer()
    out["E10e_selective"] = e10e_selective(trajs, idx, dist_r, scorer)
    print("E10e", json.dumps(out["E10e_selective"]["recall"]), flush=True)
    del scorer

    lm = LM()
    out["E10a_prefill"] = e10a_prefill(lm)
    print("E10a", json.dumps(out["E10a_prefill"]["points"]), flush=True)
    out["E10b_ledger"] = e10b_ledger(eps, dist_s, lm)
    print("E10b", json.dumps(out["E10b_ledger"]["per_policy"], indent=1),
          flush=True)
    out["E10c_state_qa"] = e10c_state_qa(eps, dist_s, lm)
    print("E10c", json.dumps(out["E10c_state_qa"], indent=1), flush=True)
    d, rows = e10d_answer_recovery(trajs, idx, dist_r, lm)
    out["E10d_answer_recovery"] = d
    print("E10d", json.dumps(d, indent=1), flush=True)

    out["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RES, "e10_llm.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(RES, "e10_resumption_rows.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("E10 done in", out["runtime_s"], "s")


if __name__ == "__main__":
    run()
