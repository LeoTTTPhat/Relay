"""E18: how large does the mandatory ledger get, and when must a handoff
block or delegate instead of shipping?

Reviewers asked for the number the ledger-saturation figure (E8c) only
implied: the *maximum* observed mandatory-ledger size, how many side-effect
entries fit a watch-class budget, and the point at which a handoff can no
longer carry every action identity.

Everything here is measured on existing corpora with exact Qwen2.5 BPE
counts; no model and no new data.

  **E18a size distribution.**  Ledger entries (side-effect-like steps) per
  real trajectory at its *final* step -- the worst case for a handoff of
  that task -- and at the sampled handoff positions every other experiment
  uses (5 seeds x 2 positions).  Same for the 200 synthetic episodes.

  **E18b per-entry cost.**  Exact tokens of one ledger line in the full form
  the distiller renders (key | fingerprint | status + provenance + head) and
  in the *saturated* identity-only form (key | fingerprint | status) the
  trim loop falls back to.  Two key formats are priced: the trace-derived key
  our experiments ship (``<trace>:<step>:<fp8>``) and the protocol format a
  deployment mints (``relay-<20 hex>`` from ``protocol.derive_key``).

  **E18c capacity.**  The largest number of saturated entries whose ledger,
  together with the minimum mandatory core (goal clipped to the distiller's
  120-character floor), fits each target's usable window -- found by exact
  tokenisation of real entries, not by dividing averages.  One entry more is
  the point at which the handoff must block or delegate: shipping would
  otherwise drop an action identity, which the design forbids.

  **E18d exposure.**  The fraction of sampled handoffs whose full-form ledger
  (i) occupies more than half the budget, (ii) forces head clipping, and
  (iii) exceeds the target window even saturated, i.e. must block/delegate.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, sample_positions, RES                  # noqa: E402
from state.schema import build_side_fx, _head                       # noqa: E402
from state.protocol import derive_key                               # noqa: E402
from policies.policies import relay_budget                          # noqa: E402
from traces.parser import exact_tokens_cached                       # noqa: E402

OUT = os.path.join(RES, "e18_ledger_scale.json")
SEEDS = [0, 1, 2, 3, 4]
CLASSES = ["watch", "speaker", "phone"]
HDR = "== SIDE-EFFECT LEDGER (do NOT re-execute) =="


def full_line(e):
    return (f"[{e.idem_key}|{e.fingerprint}|{e.status}] "
            f"step {e.step} {e.agent}: {e.head}")


def sat_line(e):
    return f"[{e.idem_key}|{e.fingerprint}|{e.status}]"


def proto_line(e):
    key = derive_key("task-secret", f"{e.agent}|{e.fingerprint}")
    return f"[{key}|{e.fingerprint}|committed]"


def ledger_tokens(lines):
    if not lines:
        return 0
    return exact_tokens_cached(HDR + "\n" + "\n".join(lines))


def core_min_tokens(goal):
    """Goal at the distiller's trim floor (never below 120 characters)."""
    g = goal
    while len(g) > 120:
        g = _head(g, max(120, len(g) // 2))
    return exact_tokens_cached("== GOAL ==\n" + g)


def pct(a, q):
    return float(np.percentile(np.asarray(a, float), q)) if len(a) else 0.0


def dist(a):
    return {"n": len(a), "median": pct(a, 50), "p90": pct(a, 90),
            "p99": pct(a, 99), "max": int(max(a)) if a else 0,
            "mean": round(float(np.mean(a)), 2) if a else 0.0}


def capacity(entries, core_tok, budget, line_fn):
    """Largest n such that core + ledger of n real entries fits ``budget``.

    Entries are cycled in corpus order so the count uses real keys and
    fingerprints rather than an averaged per-line cost.
    """
    if not entries:
        return 0
    lo, hi = 0, 1
    def fits(n):
        lines = [line_fn(entries[i % len(entries)]) for i in range(n)]
        return core_tok + 1 + ledger_tokens(lines) <= budget
    while fits(hi):
        lo, hi = hi, hi * 2
        if hi > 20000:
            break
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def analyse(trajs, label):
    finals, handoff_n = [], []
    all_entries = []
    per_full, per_sat, per_proto = [], [], []
    expo = {c: {"over_half": 0, "clip": 0, "block": 0, "n": 0}
            for c in CLASSES}
    goal_core = []
    for t in trajs:
        fin = build_side_fx(t, len(t.steps) - 1)
        finals.append(len(fin))
        all_entries.extend(fin)
        core = core_min_tokens(t.question)
        goal_core.append(core)
        for seed in SEEDS:
            for k in sample_positions(len(t.steps), t.tid, seed):
                if k >= len(t.steps):
                    continue
                led = build_side_fx(t, k)
                handoff_n.append(len(led))
                full_tok = ledger_tokens([full_line(e) for e in led])
                sat_tok = ledger_tokens([sat_line(e) for e in led])
                full_core = exact_tokens_cached("== GOAL ==\n"
                                                + _head(t.question, 600))
                for c in CLASSES:
                    b = relay_budget(c)
                    d = expo[c]
                    d["n"] += 1
                    d["over_half"] += int(full_tok > b / 2)
                    d["clip"] += int(full_core + full_tok > b)
                    d["block"] += int(core + 1 + sat_tok > b)
    # per-entry exact cost, measured line by line
    for e in all_entries:
        per_full.append(exact_tokens_cached(full_line(e)) + 1)
        per_sat.append(exact_tokens_cached(sat_line(e)) + 1)
        per_proto.append(exact_tokens_cached(proto_line(e)) + 1)
    core_med = int(np.median(goal_core)) if goal_core else 0
    cap = {}
    for c in CLASSES:
        b = relay_budget(c)
        cap[c] = {"budget": b,
                  "fit_saturated_trace_keys": capacity(
                      all_entries, core_med, b, sat_line),
                  "fit_saturated_protocol_keys": capacity(
                      all_entries, core_med, b, proto_line),
                  "fit_full_form": capacity(all_entries, core_med, b,
                                            full_line)}
        # The fitting rule, spelled out: the ledger section is tokenised as
        # one text, so its cost is not the sum of per-line counts.
        for form, fn, key in (("protocol", proto_line,
                               "fit_saturated_protocol_keys"),
                              ("full", full_line, "fit_full_form")):
            n = cap[c][key]
            lines = [fn(all_entries[i % len(all_entries)]) for i in range(n)]
            led_tok = ledger_tokens(lines)
            hdr_tok = exact_tokens_cached(HDR)
            cap[c][f"{form}_ledger_tokens_at_fit"] = led_tok
            cap[c][f"{form}_per_entry_in_block"] = round(
                (led_tok - hdr_tok) / max(n, 1), 2)
            cap[c][f"{form}_total_at_fit"] = core_med + 1 + led_tok
        cap[c]["core_tokens"] = core_med
        cap[c]["ledger_header_tokens"] = exact_tokens_cached(HDR)
    return {
        "label": label,
        "n_traj": len(trajs),
        "entries_final": dist(finals),
        "entries_at_handoff": dist(handoff_n),
        "traj_with_any_entry": sum(1 for f in finals if f),
        "tokens_per_entry": {"full": dist(per_full), "saturated": dist(per_sat),
                             "protocol_key": dist(per_proto)},
        "core_min_tokens_median": core_med,
        "capacity": cap,
        "final_ledger_would_block": {
            c: sum(1 for f in finals
                   if f > cap[c]["fit_saturated_trace_keys"])
            for c in CLASSES},
        "exposure_at_handoff": {
            c: {k: (round(v / d["n"], 4) if k != "n" else v)
                for k, v in d.items()} for c, d in expo.items()},
    }


def run():
    t0 = time.time()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    out = {"real": analyse(trajs, "who&when"),
           "synthetic": analyse(synth_trajs, "synthetic-200")}
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    for k in ("real", "synthetic"):
        d = out[k]
        print(f"\n[{k}] final entries {d['entries_final']}")
        print(f"   at handoff {d['entries_at_handoff']}")
        print(f"   tokens/entry full {d['tokens_per_entry']['full']['mean']} "
              f"sat {d['tokens_per_entry']['saturated']['mean']} "
              f"proto {d['tokens_per_entry']['protocol_key']['mean']}")
        print(f"   capacity {d['capacity']}")
        print(f"   final ledgers that would block {d['final_ledger_would_block']}")
        print(f"   exposure {d['exposure_at_handoff']}")
    print("->", OUT, out["runtime_s"], "s")


if __name__ == "__main__":
    run()
