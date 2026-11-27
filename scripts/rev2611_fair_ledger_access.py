"""Revision 2026-11: E23b compared under equal ledger access (CPU only, no model).

Question (Review B, "compare baselines under the same conditions"): in the
enforced E23b runs every policy uses the same dispatcher, but the dispatcher
can only match the action keys that survive in the text a policy ships.
Truncated baselines lose some keys, Relay never does.  Is Relay's enforced
advantage only the result of that unequal access?

Method.  For every one of the 200 padded E23b handoffs we re-render each
baseline's block exactly as ``run_e23b_binding.py`` did (same padding, same
handoff, same budget, deterministic; no model) and record whether *every*
pre-handoff action key is present in the shipped text, which is the condition
under which the enforced dispatcher knows every finished action.  On the
episodes where a baseline kept every key, we compare its stored enforced
outcome with Relay's stored enforced outcome on the same episodes (exact
McNemar).  Nothing is re-run with a model; the outcomes are the stored rows
``results/e23b_rows_<model>.json``.

Output: ``results_rev2026-11/fair_ledger_access.json``.
Reads only ``results/`` and ``src/``; writes nothing under ``results/``.
"""
import json
import os
import sys
import time
from math import comb

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from explib import load_all, sample_positions, policy_budget, RES   # noqa: E402
from policies.policies import POLICIES                              # noqa: E402
from run_e23b_binding import padded, choose_P, DST                  # noqa: E402
from run_e23_continuation import build_world                        # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(ROOT, "results_rev2026-11")
BASELINES = ["full", "trunc25", "recency", "randdrop"]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    t0 = time.time()
    trajs, idx, eps, synth, idx_s = load_all()
    P = choose_P(eps)
    all_keys = {p: {} for p in BASELINES}
    kinds = {}
    for e in eps:
        k = sample_positions(e.n, e.eid, 0, 1)[0]
        traj, imap = padded(e, P)
        kp = imap[k]
        items, done_before, required = build_world(e, k)
        keys = [it["key"].lower() for it in items
                if it["kind"] == "effect" and it["id"] in done_before]
        kinds[e.eid] = {it["mid"]: it["kind"] for it in items}
        for p in BASELINES:
            o = POLICIES[p](traj, kp, DST, seed=0,
                            budget=policy_budget(p, DST))
            low = o.resume_text.lower()
            all_keys[p][e.eid] = bool(keys) and all(kk in low for kk in keys)
    out = {"pad_pairs_per_step": P, "n_episodes": len(eps),
           "all_keys_kept_fraction": {
               p: round(sum(v.values()) / len(v), 4)
               for p, v in all_keys.items()},
           "models": {}}
    for tag in ("Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"):
        rows = json.load(open(os.path.join(RES, f"e23b_rows_{tag}.json")))
        by = {(r["eid"], r["policy"], r["mode"]): r for r in rows}
        eids = sorted({r["eid"] for r in rows})
        pols = sorted({r["policy"] for r in rows})
        res = {}
        for p in BASELINES:
            if p not in pols:
                continue
            sub = [e for e in eids if all_keys[p].get(e)]
            n = len(sub)
            d = {"n_episodes_all_keys_kept": n}
            if n:
                for fld in ("success", "duplicate", "dup_pre", "complete",
                            "violation"):
                    d[f"{p}_enforced_{fld}"] = round(
                        sum(by[(e, p, "enforced")][fld] for e in sub) / n, 4)
                    d[f"relay_enforced_{fld}"] = round(
                        sum(by[(e, "relay", "enforced")][fld] for e in sub) / n,
                        4)
                b = sum(1 for e in sub if by[(e, "relay", "enforced")]["success"]
                        and not by[(e, p, "enforced")]["success"])
                c = sum(1 for e in sub if by[(e, p, "enforced")]["success"]
                        and not by[(e, "relay", "enforced")]["success"])
                d["success_relay_only"] = b
                d["success_baseline_only"] = c
                d["success_p_exact_mcnemar"] = round(mcnemar_exact(b, c), 4)
                # Round R2 (2026-11-25): why violation rates differ on the
                # subset.  A violation needs the model to keep acting; count,
                # per enforced row, the world-changing choices it made (effects
                # and the distractor, suppressed ones included) and whether it
                # declared the task done.
                for q in (p, "relay"):
                    rs = [by[(e, q, "enforced")] for e in sub]
                    acts = [sum(1 for c in r["trace"]
                                if kinds[r["eid"]].get(c) in
                                ("effect", "distractor")) for r in rs]
                    d[f"{q}_enforced_mean_world_choices"] = round(
                        sum(acts) / n, 3)
                    d[f"{q}_enforced_mean_turns"] = round(
                        sum(r["turns"] for r in rs) / n, 3)
                    d[f"{q}_enforced_stopped"] = round(
                        sum(r["stopped"] for r in rs) / n, 4)
                    lo = [r["violation"] for r, a_ in zip(rs, acts) if a_ <= 1]
                    hi = [r["violation"] for r, a_ in zip(rs, acts) if a_ >= 2]
                    d[f"{q}_violation_if_world_choices_le1"] = (
                        [round(sum(lo) / len(lo), 4), len(lo)] if lo else None)
                    d[f"{q}_violation_if_world_choices_ge2"] = (
                        [round(sum(hi) / len(hi), 4), len(hi)] if hi else None)
                b = sum(1 for e in sub if by[(e, "relay", "enforced")]["violation"]
                        and not by[(e, p, "enforced")]["violation"])
                c = sum(1 for e in sub if by[(e, p, "enforced")]["violation"]
                        and not by[(e, "relay", "enforced")]["violation"])
                d["violation_p_exact_mcnemar"] = round(mcnemar_exact(b, c), 4)
                d["subset_eids"] = sub
            res[p] = d
        out["models"][tag] = res
    out["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "fair_ledger_access.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
