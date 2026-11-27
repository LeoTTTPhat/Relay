"""E23: does a resumed model actually *continue the task*?

Every fidelity number in the paper is a preservation metric: anchors or exact
state fields that survive into the shipped block.  Two reviewers asked for a
task-level measure on the 200 synthetic episodes with the models already used
(Qwen2.5-3B/7B-Instruct).  This is it, as a closed-loop, menu-constrained
continuation in a simulated tool world with exact ground truth.

**Set-up (fixed before running).**  For each episode, one handoff position
(``sample_positions(n, eid, seed=0, per=1)``, as E10b), watch target.  Each
policy's shipped block is given to the model together with a menu of typed
actions for the episode:

  * every distinct world-changing action of the episode (those executed before
    the handoff *and* those still to do), by exact action identity;
  * every read-only lookup of the episode, which returns its true value;
  * one constraint-violating distractor per family (errand: a purchase over
    the stated budget; home: scheduling the device after the stated hour;
    shopping: shipping to an address other than the saved one, or buying over
    the cap; travel: a non-refundable booking; fleet: none -- its constraint
    is "never actuate a device twice", which is the duplicate criterion);
  * DONE.

The model picks one action ID per turn; the world executes it and appends the
observation; the loop ends at DONE or after ``|required| + 3`` turns.  Menu
order is a fixed per-episode shuffle.  Greedy decoding.

**Two execution regimes.**  ``advisory``: every chosen action reaches the
world.  ``enforced``: a trusted dispatcher below the model suppresses an
action whose record it holds -- a pre-handoff action whose idempotency key
survives in the shipped block (the E13 visibility rule), or an action it has
itself dispatched since resumption -- and tells the model it was skipped.

**Task-level continuation success** (defined before running) requires all of
  S1 every world-changing action still required at the handoff is executed;
  S2 no action is executed twice in the world (including re-executing a
     pre-handoff action);
  S3 no constraint-violating distractor is executed.
Rates are over handoffs, with bootstrap CIs over episodes; per-criterion
failure rates are reported alongside so a success rate cannot hide why.

This is a controlled proxy for continuation, not open-ended task completion:
the action space is enumerated, arguments are given, and the world is
simulated.  It measures whether the shipped state lets a small model pick the
remaining actions without repeating or violating anything.
"""
import copy
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, sample_positions, boot_ci, policy_budget, RES  # noqa
from policies.policies import POLICIES                                     # noqa
from state.schema import Distiller                                         # noqa
from traces.anchors import norm_text                                       # noqa

MODEL = os.environ.get("RELAY_E23_MODEL", "Qwen/Qwen2.5-3B-Instruct")
TAG = MODEL.split("/")[-1]
POLS = os.environ.get("RELAY_E23_POLICIES",
                      "full,recency,typed-recency,relay,trunc25,randdrop,"
                      "restart").split(",")
N_EP = int(os.environ.get("RELAY_E23_N", "200"))
OUT = os.path.join(RES, "e23_continuation.json")
ROWS = os.path.join(RES, f"e23_rows_{TAG}.json")
DST = "watch"

PREFIX = (
    "You are an assistant resuming a task that was started on another "
    "device. Everything that was transferred from the other device is "
    "below.\n\n--- TRANSFERRED STATE ---\n{state}\n--- END STATE ---\n\n"
    "You can act only through these actions. Choosing an action performs it "
    "in the real world.\n{menu}\nD0: DONE (the task is complete; stop)\n\n"
    "Rules: choose exactly one action per turn. Never perform an action that "
    "has already been performed, before or after the handoff. Respect every "
    "constraint stated in the goal.\n\n")
SUFFIX = ("Actions taken on this device so far:\n{history}\n\n"
          "Reply with the ID of the next action only (for example A3), or D0 "
          "if the task is complete.")

TOOL_RE = re.compile(r"^TOOL (\S+)\((.*?)\) -> (.+?) = (.+?)\. ", re.S)


# ---------------------------------------------------------------------------
# The world of one episode
# ---------------------------------------------------------------------------
def _distractor(e):
    g = e.goal
    finals = {se["action"]: se for se in e.states[-1].side_effects}
    if e.family == "errand":
        item = re.search(r"pick up the (.+?) at ", g).group(1)
        budget = int(re.search(r"under \$(\d+)", g).group(1))
        return ("purchase", f"item={item}, amount=${budget * 1.25:.2f}",
                "budget")
    if e.family == "home":
        dev3 = re.search(r"schedule the (.+?)\. Must not run", g).group(1)
        return ("device_actuation", f"schedule {dev3} at 11:45 pm", "hour")
    if e.family == "shopping":
        prod = re.search(r"^Buy (.+?) at the best price", g).group(1)
        if "purchase" in finals:
            args = finals["purchase"]["args"].replace("saved address",
                                                      "new address")
            return ("purchase", args, "address")
        offers = [float(m) for m in re.findall(
            r"offer = \$(\d+(?:\.\d+)?)",
            " ".join(s.content for s in e.steps))]
        return ("purchase", f"{prod}, ${min(offers)}, saved address", "cap")
    if e.family == "travel":
        bk = [se for se in e.states[-1].side_effects
              if se["action"] == "booking" and "nights at" in se["args"]]
        return ("booking", bk[0]["args"] + ", non-refundable rate",
                "refundable")
    return None


def build_world(e, k):
    effects, seen = [], set()
    for se in e.states[-1].side_effects:
        if se["action_id"] in seen:
            continue
        seen.add(se["action_id"])
        effects.append({"kind": "effect", "id": se["action_id"],
                        "label": f"{se['action']}({se['args']})",
                        "key": se["key"]})
    lookups, lseen = [], set()
    for s in e.steps:
        m = TOOL_RE.match(s.content) if s.is_tool_result else None
        if m and (m.group(1), m.group(2)) not in lseen:
            lseen.add((m.group(1), m.group(2)))
            lookups.append({"kind": "lookup", "id": f"lookup|{m.group(1)}|"
                            f"{m.group(2)}",
                            "label": f"{m.group(1)}({m.group(2)})",
                            "result": f"{m.group(3)} = {m.group(4)}"})
    items = effects + lookups
    dis = _distractor(e)
    if dis:
        items.append({"kind": "distractor", "id": "distractor",
                      "label": f"{dis[0]}({dis[1]})", "constraint": dis[2]})
    rng = random.Random(f"menu|{e.eid}")
    rng.shuffle(items)
    for i, it in enumerate(items):
        it["mid"] = f"A{i + 1}"
    done_before = {se["action_id"] for se in e.states[k].side_effects}
    required = [it["id"] for it in effects if it["id"] not in done_before]
    return items, done_before, required


# ---------------------------------------------------------------------------
class LM:
    def __init__(self, name):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tk = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.float16).to(self.dev).eval()

    def ids(self, text):
        return self.tk(text, add_special_tokens=False,
                       return_tensors="pt").input_ids.to(self.dev)

    def split_template(self, prefix_content, suffix_content):
        full = self.tk.apply_chat_template(
            [{"role": "user", "content": prefix_content + suffix_content}],
            add_generation_prompt=True, tokenize=False)
        cut = full.index(prefix_content) + len(prefix_content)
        return full[:cut], full[cut:]

    def prefix_cache(self, prefix_text):
        from transformers import DynamicCache
        pid = self.ids(prefix_text)
        cache = DynamicCache()
        with self.torch.no_grad():
            cache = self.m(pid, past_key_values=cache,
                           use_cache=True).past_key_values
        return pid, cache

    def gen(self, pid, cache, suffix_text, max_new=6):
        sid = self.ids(suffix_text)
        ids = self.torch.cat([pid, sid], dim=1)
        with self.torch.no_grad():
            o = self.m.generate(ids, attention_mask=self.torch.ones_like(ids),
                                past_key_values=copy.deepcopy(cache),
                                max_new_tokens=max_new, do_sample=False,
                                pad_token_id=self.tk.eos_token_id)
        return self.tk.decode(o[0][ids.shape[1]:],
                              skip_special_tokens=True).strip()


def run_loop(lm, pid, cache, suffix_tmpl_parts, items, done_before, required,
             state_text, mode):
    by_mid = {it["mid"]: it for it in items}
    # dispatcher records: pre-handoff actions whose key survived the budget
    h = norm_text(state_text)
    known = {it["id"] for it in items if it["kind"] == "effect"
             and it["id"] in done_before and it["key"].lower() in h}
    executed = defaultdict(int)          # world executions after resumption
    history, trace = [], []
    suppressed = invalid = 0
    cap = len(required) + 3
    for turn in range(cap):
        hist = "\n".join(history) if history else "(none yet)"
        pre_s, post_s = suffix_tmpl_parts
        out = lm.gen(pid, cache, pre_s + SUFFIX.format(history=hist) + post_s)
        m = re.search(r"\b([AD]\d+)\b", out.upper())
        choice = m.group(1) if m else None
        trace.append(choice or f"?{out[:20]}")
        if choice == "D0":
            break
        it = by_mid.get(choice)
        if it is None:
            invalid += 1
            history.append(f"Turn {turn + 1}: invalid reply ({out[:20]!r})")
            continue
        if it["kind"] == "lookup":
            history.append(f"Turn {turn + 1}: {it['mid']} {it['label']} -> "
                           f"RESULT {it['result']}")
            continue
        if mode == "enforced" and it["kind"] == "effect" and \
                (it["id"] in known or executed[it["id"]] > 0):
            suppressed += 1
            history.append(f"Turn {turn + 1}: {it['mid']} {it['label']} -> "
                           f"SKIPPED by the runtime: already performed")
            continue
        executed[it["id"]] += 1
        if mode == "enforced" and it["kind"] == "effect":
            known.add(it["id"])
        history.append(f"Turn {turn + 1}: {it['mid']} {it['label']} -> OK, "
                       f"performed")
    s1 = all(executed[r] >= 1 for r in required)
    dup = any(executed[i] >= 1 for i in done_before) or \
        any(v >= 2 for i, v in executed.items() if i != "distractor")
    viol = executed["distractor"] >= 1
    return {"success": bool(s1 and not dup and not viol), "complete": s1,
            "duplicate": dup, "violation": viol, "suppressed": suppressed,
            "invalid": invalid, "turns": len(trace), "trace": trace,
            "stopped": trace[-1] == "D0" if trace else False}


def summarize(rows, n_episodes, runtime_s, note=""):
    """Aggregate per-handoff rows into the reported summary."""
    summ = {}
    for p in POLS:
        for mode in ("advisory", "enforced"):
            for subset in ("all", "has_required"):
                sel = [r for r in rows if r["policy"] == p and
                       r["mode"] == mode and
                       (subset == "all" or r["n_required"] > 0)]
                if not sel:
                    continue
                d = {"n": len(sel)}
                for fld in ("success", "complete", "duplicate", "violation",
                            "stopped"):
                    m_, lo, hi = boot_ci([float(r[fld]) for r in sel])
                    d[fld] = [round(m_, 4), round(lo, 4), round(hi, 4)]
                d["mean_turns"] = round(sum(r["turns"] for r in sel)
                                        / len(sel), 2)
                d["mean_suppressed"] = round(sum(r["suppressed"] for r in sel)
                                             / len(sel), 3)
                d["invalid_rate"] = round(sum(r["invalid"] for r in sel)
                                          / max(sum(r["turns"] for r in sel),
                                                1), 4)
                summ[f"{p}|{mode}|{subset}"] = d
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    prev[TAG] = {"model": MODEL, "policies": POLS, "n_episodes": n_episodes,
                 "target": DST, "summary": summ, "runtime_s": runtime_s,
                 "note": note}
    json.dump(prev, open(OUT, "w"), indent=1)
    for key, d in summ.items():
        if key.endswith("|all"):
            print(f"{key:32s} success={d['success'][0]:.3f} "
                  f"complete={d['complete'][0]:.3f} dup={d['duplicate'][0]:.3f}"
                  f" viol={d['violation'][0]:.3f} turns={d['mean_turns']}")
    print("->", OUT, runtime_s, "s")


def summarize_only():
    """Summarise the first ``RELAY_E23_N`` episodes of an existing rows file.

    Used when a run is stopped at a pre-set episode count because the full
    200 episodes do not fit the experiment's time box; the rows kept are
    exactly those a fresh ``RELAY_E23_N=<n>`` run would produce.
    """
    from synth.generator import generate
    keep = {e.eid for e in generate(200, seed=0)[:N_EP]}
    rows = [r for r in json.load(open(ROWS)) if r["eid"] in keep]
    done = {r["eid"] for r in rows
            if sum(1 for q in rows if q["eid"] == r["eid"]) ==
            2 * len(POLS)}
    rows = [r for r in rows if r["eid"] in done]
    json.dump(rows, open(ROWS, "w"), indent=1)
    summarize(rows, len(done), None,
              note=f"stopped at {len(done)} episodes (time box)")


def run():
    t0 = time.time()
    trajs, idx, eps, synth, idx_s = load_all()
    dist = Distiller(idx_s)
    lm = LM(MODEL)
    rows = []
    cases = []
    for e in eps[:N_EP]:
        k = sample_positions(e.n, e.eid, 0, 1)[0]
        cases.append((e, k))
    for ci, (e, k) in enumerate(cases):
        traj = e.to_trajectory()
        items, done_before, required = build_world(e, k)
        menu = "\n".join(f"{it['mid']}: {it['label']}" for it in items)
        for p in POLS:
            kw = {"distiller": dist} if p.startswith("relay") else {}
            o = POLICIES[p](traj, k, DST, seed=0,
                            budget=policy_budget(p, DST), **kw)
            prefix_content = PREFIX.format(state=o.resume_text, menu=menu)
            pre_text, post_text = lm.split_template(prefix_content,
                                                    "@@SUFFIX@@")
            # post_text is "@@SUFFIX@@<|im_end|>...assistant\n"
            post_s = post_text.replace("@@SUFFIX@@", "", 1)
            pid, cache = lm.prefix_cache(pre_text)
            for mode in ("advisory", "enforced"):
                r = run_loop(lm, pid, cache, ("", post_s), items,
                             done_before, required, o.resume_text, mode)
                rows.append({"eid": e.eid, "family": e.family, "k": k,
                             "policy": p, "mode": mode,
                             "n_required": len(required),
                             "n_done_before": len(done_before),
                             "resume_tokens": o.tokens_resume, **r})
            del cache
        if ci % 10 == 9:
            print(f"{ci + 1}/{len(cases)} episodes, {time.time() - t0:.0f}s",
                  flush=True)
            json.dump(rows, open(ROWS, "w"), indent=1)
    json.dump(rows, open(ROWS, "w"), indent=1)
    summarize(rows, len(cases), round(time.time() - t0, 1))


if __name__ == "__main__":
    if os.environ.get("RELAY_E23_SUMMARIZE_ONLY"):
        summarize_only()
    else:
        run()
