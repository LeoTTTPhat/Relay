"""E23b: closed-loop continuation when the watch budget *binds*.

E23 ran on the 200 synthetic episodes as generated, and every one of them fits
the watch-class window, so full context, recency and the typed block all
shipped the whole trajectory's information and the study could not test the
condition a bounded migration object exists for.  E23b repeats the study with
the budget binding.  Everything below was fixed, and recorded in AUDIT.md
(H.1), before the first model call.

**Budget rule (padding).**  After every original step of every episode we
insert ``P`` filler pairs -- a verbose read-only tool result and a planner note
that changes nothing -- generated exactly as the out-of-schema corpus pads its
episodes (``synth.oos._pad``), with an RNG seeded by the episode id.  Fillers
do not change ground-truth state.  ``P`` is the smallest integer in 1..12 such
that at the handoff of *every* one of the 200 episodes the raw prefix exceeds
the watch-class usable window, so full context is truncated at the target in
all 200 handoffs (verified and stored in the results file).  The handoff is
taken right after the same original step as in E23
(``sample_positions(n, eid, seed=0, per=1)`` on the unpadded episode).

**Unchanged from E23.**  The menu is built from the *unpadded* episode (every
distinct world-changing action, every original read-only lookup, one
constraint-violating distractor per family, DONE; fixed per-episode shuffle);
turn cap ``|required| + 3``; greedy decoding through ``generate`` with the
model's own generation config; advisory vs enforced execution; success =
S1 (every still-required action performed) and S2 (nothing performed twice,
including pre-handoff actions) and S3 (distractor never performed).

**Recorded in addition.**  ``dup_pre`` (a pre-handoff action performed again),
``dup_post`` (an action first performed after the handoff performed twice),
the turn of the first duplicate, and whether the policy's block was truncated
at the target.

``RELAY_E23B_MODEL``    model (default Qwen/Qwen2.5-3B-Instruct)
``RELAY_E23B_N``        number of episodes, taken in corpus order (default 200)
``RELAY_E23B_POLICIES`` comma-separated policies (default all seven; the
                      diagnostic ``relay-planlabel`` of AUDIT H.5 is opt-in)
``RELAY_E23B_SHARD``   ``i/n``: run the episodes whose index % n == i
``RELAY_E23B_TAG``     output tag (default: model name)
``RELAY_E23B_MERGE``   merge ``n`` shards instead of running
                      (``RELAY_E23B_MERGE_N``: declared episode count)
"""
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, sample_positions, boot_ci, policy_budget, RES  # noqa
from policies.policies import POLICIES, usable                              # noqa
from state.schema import Distiller                                          # noqa
from traces.anchors import norm_text                                        # noqa
from traces.parser import Step, Trajectory, est_tokens, exact_tokens_cached  # noqa
from synth.generator import _Builder                                        # noqa
from synth.oos import _pad                                                  # noqa
from run_e23_continuation import build_world, PREFIX, SUFFIX                # noqa

MODEL = os.environ.get("RELAY_E23B_MODEL", "Qwen/Qwen2.5-3B-Instruct")
# RELAY_E23B_TAG separates a diagnostic run (e.g. relay-planlabel) from the
# main rows and summary of the same model.
TAG = os.environ.get("RELAY_E23B_TAG", MODEL.split("/")[-1])
POLS = os.environ.get("RELAY_E23B_POLICIES",
                      "full,trunc25,recency,typed-recency,randdrop,relay,"
                      "restart").split(",")
N_EP = int(os.environ.get("RELAY_E23B_N", "200"))
SHARD = os.environ.get("RELAY_E23B_SHARD", "0/1")      # "i/n": index % n == i
SH_I, SH_N = (int(x) for x in SHARD.split("/"))
OUT = os.path.join(RES, "e23b_binding.json")
ROWS = os.path.join(RES, f"e23b_rows_{TAG}_s{SH_I}of{SH_N}.json")
DST = "watch"


# ---------------------------------------------------------------------------
# Padded episodes
# ---------------------------------------------------------------------------
def fillers(eid, n_pairs):
    """``n_pairs`` (tool, note) filler steps, deterministic per episode."""
    b = _Builder(f"{eid}-pad")
    _pad(b, random.Random(f"pad|{eid}"), n_pairs, topic_seed=0)
    return b.steps


def padded(e, P):
    """Trajectory with P filler pairs after every original step.

    Returns ``(trajectory, index_map)`` where ``index_map[i]`` is the padded
    index of original step ``i``.
    """
    pool = fillers(e.eid, P * e.n)
    steps, imap = [], {}
    j = 0
    for i, s in enumerate(e.steps):
        imap[i] = len(steps)
        steps.append(Step(idx=len(steps), role=s.role, agent=s.agent,
                          content=s.content,
                          side_effect_like=s.side_effect_like,
                          is_tool_result=s.is_tool_result, tokens=s.tokens))
        for _ in range(2 * P):
            f = pool[j]
            j += 1
            steps.append(Step(idx=len(steps), role=f.role, agent=f.agent,
                              content=f.content,
                              side_effect_like=f.side_effect_like,
                              is_tool_result=f.is_tool_result,
                              tokens=f.tokens))
    return Trajectory(tid=e.eid, source="synthetic", question=e.goal,
                      ground_truth="", steps=steps), imap


def choose_P(eps):
    limit = usable(DST)
    for P in range(1, 13):
        ok = True
        for e in eps:
            k = sample_positions(e.n, e.eid, 0, 1)[0]
            t, imap = padded(e, P)
            if exact_tokens_cached(t.prefix_text(imap[k])) <= limit:
                ok = False
                break
        if ok:
            return P
    raise RuntimeError("no P in 1..12 makes every handoff exceed the window")


# ---------------------------------------------------------------------------
# Diagnostic rendering (AUDIT H.5): same block, plan labelled as written, tail
# oldest-first.  Implemented here, on the rendered block, so the distiller the
# rest of the paper evaluates is untouched.
# ---------------------------------------------------------------------------
def relay_planlabel(traj, k, dist):
    from state.schema import PLAN_HINT_RE
    from tok.tokenizer import cap_suffix
    block = dist.distill(traj, k, policy_budget("relay", DST))
    plan_step = None
    for st in reversed(traj.steps[:k + 1]):
        if PLAN_HINT_RE.search(st.content):
            plan_step = st.idx
            break

    def _step_no(line):
        m = re.match(r"\[step (\d+) ", line)
        return int(m.group(1)) if m else 0
    block.tail = sorted(block.tail, key=_step_no)
    text = block.render()
    if block.plan and plan_step is not None:
        text = text.replace(
            "== PENDING PLAN ==",
            f"== LATEST PLAN (as written at step {plan_step}; the ledger "
            f"below lists what is done) ==", 1)
    resume, tok, trunc = cap_suffix(text, usable(DST))

    class _Out:
        pass
    o = _Out()
    o.resume_text, o.tokens_resume = resume, min(tok, usable(DST))
    o.tokens_wire, o.truncated_at_target = tok, trunc
    return o


# ---------------------------------------------------------------------------
class LM:
    """Prefix-cached greedy generation; the prefix cache is cropped back
    after every turn instead of deep-copied (identical outputs, less memory
    traffic)."""

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

    def split_template(self, prefix_content):
        full = self.tk.apply_chat_template(
            [{"role": "user", "content": prefix_content + "@@S@@"}],
            add_generation_prompt=True, tokenize=False)
        cut = full.index(prefix_content) + len(prefix_content)
        return full[:cut], full[cut:].replace("@@S@@", "", 1)

    def prefix_cache(self, prefix_text):
        from transformers import DynamicCache
        pid = self.ids(prefix_text)
        with self.torch.no_grad():
            cache = self.m(pid, past_key_values=DynamicCache(),
                           use_cache=True).past_key_values
        return pid, cache

    def gen(self, pid, cache, suffix_text, max_new=6):
        sid = self.ids(suffix_text)
        ids = self.torch.cat([pid, sid], dim=1)
        with self.torch.no_grad():
            o = self.m.generate(ids, attention_mask=self.torch.ones_like(ids),
                                past_key_values=cache, max_new_tokens=max_new,
                                do_sample=False,
                                pad_token_id=self.tk.eos_token_id)
        cache.crop(pid.shape[1])
        return self.tk.decode(o[0][ids.shape[1]:],
                              skip_special_tokens=True).strip()


def run_loop(lm, pid, cache, post, items, done_before, required, state_text,
             mode):
    by_mid = {it["mid"]: it for it in items}
    h = norm_text(state_text)
    known = {it["id"] for it in items if it["kind"] == "effect"
             and it["id"] in done_before and it["key"].lower() in h}
    executed = defaultdict(int)
    history, trace = [], []
    suppressed = invalid = 0
    first_dup_turn = None
    cap = len(required) + 3
    for turn in range(cap):
        hist = "\n".join(history) if history else "(none yet)"
        out = lm.gen(pid, cache, SUFFIX.format(history=hist) + post)
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
        if it["kind"] == "effect" and first_dup_turn is None and (
                it["id"] in done_before or executed[it["id"]] >= 2):
            first_dup_turn = turn + 1
        if mode == "enforced" and it["kind"] == "effect":
            known.add(it["id"])
        history.append(f"Turn {turn + 1}: {it['mid']} {it['label']} -> OK, "
                       f"performed")
    s1 = all(executed[r] >= 1 for r in required)
    dup_pre = any(executed[i] >= 1 for i in done_before)
    dup_post = any(v >= 2 for i, v in executed.items()
                   if i != "distractor" and i not in done_before)
    dup = dup_pre or dup_post
    viol = executed["distractor"] >= 1
    return {"success": bool(s1 and not dup and not viol), "complete": s1,
            "duplicate": dup, "dup_pre": dup_pre, "dup_post": dup_post,
            "first_dup_turn": first_dup_turn, "violation": viol,
            "suppressed": suppressed, "invalid": invalid, "turns": len(trace),
            "trace": trace, "stopped": trace[-1] == "D0" if trace else False}


FIELDS = ("success", "complete", "duplicate", "dup_pre", "dup_post",
          "violation", "stopped")


def summarize(rows, meta):
    summ = {}
    pols = sorted({r["policy"] for r in rows}, key=POLS.index)
    for p in pols:
        for mode in ("advisory", "enforced"):
            sel = [r for r in rows if r["policy"] == p and r["mode"] == mode]
            if not sel:
                continue
            d = {"n": len(sel)}
            for fld in FIELDS:
                m_, lo, hi = boot_ci([float(r[fld]) for r in sel])
                d[fld] = [round(m_, 4), round(lo, 4), round(hi, 4)]
            d["mean_turns"] = round(sum(r["turns"] for r in sel) / len(sel), 2)
            d["mean_resume_tokens"] = round(
                sum(r["resume_tokens"] for r in sel) / len(sel), 1)
            d["truncated_at_target"] = round(
                sum(r["truncated"] for r in sel) / len(sel), 4)
            summ[f"{p}|{mode}"] = d
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    prev[TAG] = {**meta, "summary": summ}
    json.dump(prev, open(OUT, "w"), indent=1)
    for key, d in summ.items():
        print(f"{key:24s} succ={d['success'][0]:.3f} dup={d['duplicate'][0]:.3f}"
              f" pre={d['dup_pre'][0]:.3f} post={d['dup_post'][0]:.3f} "
              f"viol={d['violation'][0]:.3f} tok={d['mean_resume_tokens']}",
              flush=True)


def run():
    t0 = time.time()
    trajs, idx, all_eps, synth, idx_s = load_all()
    P = choose_P(all_eps)            # always over all 200 episodes
    # Shards interleave by index, so shards advancing together complete a
    # contiguous prefix of the corpus order.
    eps = [e for i, e in enumerate(all_eps[:N_EP]) if i % SH_N == SH_I]
    dist = Distiller(idx_s)
    lm = LM(MODEL)
    rows = []
    trunc_full = 0
    for ci, e in enumerate(eps):
        k = sample_positions(e.n, e.eid, 0, 1)[0]
        traj, imap = padded(e, P)
        kp = imap[k]
        items, done_before, required = build_world(e, k)
        menu = "\n".join(f"{it['mid']}: {it['label']}" for it in items)
        for p in POLS:
            if p == "relay-planlabel":
                o = relay_planlabel(traj, kp, dist)
            else:
                kw = {"distiller": dist} if p.startswith("relay") else {}
                o = POLICIES[p](traj, kp, DST, seed=0,
                                budget=policy_budget(p, DST), **kw)
            if p == "full":
                trunc_full += int(o.truncated_at_target)
            pre, post = lm.split_template(
                PREFIX.format(state=o.resume_text, menu=menu))
            pid, cache = lm.prefix_cache(pre)
            for mode in ("advisory", "enforced"):
                r = run_loop(lm, pid, cache, post, items, done_before,
                             required, o.resume_text, mode)
                rows.append({"eid": e.eid, "family": e.family, "k": k,
                             "k_padded": kp, "policy": p, "mode": mode,
                             "n_required": len(required),
                             "n_done_before": len(done_before),
                             "resume_tokens": o.tokens_resume,
                             "wire_tokens": o.tokens_wire,
                             "truncated": bool(o.truncated_at_target), **r})
            del cache
        if ci % 10 == 9:
            print(f"{ci + 1}/{len(eps)} episodes, {time.time() - t0:.0f}s",
                  flush=True)
            json.dump(rows, open(ROWS, "w"), indent=1)
    json.dump(rows, open(ROWS, "w"), indent=1)
    json.dump({"model": MODEL, "shard": SHARD, "n_episodes": len(eps),
               "pad_pairs_per_step": P, "full_truncated_handoffs": trunc_full,
               "runtime_s": round(time.time() - t0, 1)},
              open(ROWS.replace(".json", "_meta.json"), "w"), indent=1)
    print("shard done", SHARD, round(time.time() - t0, 1), "s", flush=True)


def merge(n_shards, n_episodes=None):
    """Merge shard rows; keep only episodes completed for every policy/mode.

    With ``n_episodes`` the merged set is the first ``n_episodes`` episodes in
    corpus order, and every one of them must be complete.
    """
    from synth.generator import generate
    order = [e.eid for e in generate(200, seed=0)]
    rows, metas = [], []
    for i in range(n_shards):
        p = os.path.join(RES, f"e23b_rows_{TAG}_s{i}of{n_shards}.json")
        rows += json.load(open(p))
        mp = p.replace(".json", "_meta.json")
        if os.path.exists(mp):
            metas.append(json.load(open(mp)))
    need = 2 * len(POLS)
    cnt = defaultdict(int)
    for r in rows:
        if r["policy"] in POLS:
            cnt[r["eid"]] += 1
    complete = [eid for eid in order if cnt[eid] == need]
    if n_episodes is not None:
        want = order[:n_episodes]
        missing = [eid for eid in want if cnt[eid] != need]
        if missing:
            raise RuntimeError(f"{len(missing)} of the first {n_episodes} "
                               f"episodes are incomplete")
        keep = set(want)
    else:
        # the longest contiguous completed prefix of the corpus order
        keep = set()
        for eid in order:
            if cnt[eid] != need:
                break
            keep.add(eid)
    rows = [r for r in rows if r["eid"] in keep and r["policy"] in POLS]
    json.dump(rows, open(os.path.join(RES, f"e23b_rows_{TAG}.json"), "w"),
              indent=1)
    P = metas[0]["pad_pairs_per_step"] if metas else None
    full_trunc = sum(1 for r in rows if r["policy"] == "full"
                     and r["mode"] == "advisory" and r["truncated"])
    meta = {"model": MODEL, "policies": POLS, "n_episodes": len(keep),
            "target": DST, "pad_pairs_per_step": P,
            "full_truncated_handoffs": full_trunc,
            "shards": n_shards,
            "runtime_s_per_shard": [m.get("runtime_s") for m in metas]}
    summarize(rows, meta)


if __name__ == "__main__":
    if os.environ.get("RELAY_E23B_MERGE"):
        n = os.environ.get("RELAY_E23B_MERGE_N")
        merge(int(os.environ["RELAY_E23B_MERGE"]),
              int(n) if n else None)
    else:
        run()
