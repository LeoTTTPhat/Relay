"""Content-selection baselines: what a *compression* method does instead.

The paper argues Relay is not prompt compression.  That argument is only
worth making if compression is actually on the table, so two additional
policies select content from the same prefix at the same token budget:

  P7 ``extractive`` -- TextRank over the prefix's lines with TF-IDF
     similarity: the classical query-independent extractive summariser.
     Deterministic, model-free, runs on every evaluation.
  P8 ``selective`` -- Selective-Context / LLMLingua-style pruning
     (Li et al. 2023; Jiang et al. 2023): a small causal LM scores every
     token's self-information ``-log p(t | t_<t)``, scores are pooled to
     lexical units, and the lowest-information units are dropped until the
     budget is met.  This needs a forward pass per prefix, so it is
     evaluated on a stratified subsample (Sec. VI-E1c) rather than on all
     handoffs.

Both receive the task statement for free, exactly as ``recency`` and
``relay`` do, so the comparison is about *selection*, not about who is
told the goal.
"""
import gzip
import math
import os
import re
import time
from typing import List, Optional

import numpy as np

from traces.parser import Trajectory, est_tokens, exact_tokens_cached
from policies.policies import PolicyOutput, _cap, relay_budget, usable
from devices.constants import DEVICES

_UNIT_SPLIT = re.compile(r"\n+")
# TextRank is O(u^2) in the number of units; the longest trajectories in the
# corpus produce tens of thousands of lines, which is neither tractable nor
# how the method is used in practice.  We rank the most recent MAX_UNITS
# lines and disclose the cap: it can only help the baseline, since a
# budget-sized selection is drawn from the region a recency policy would
# also prefer.
MAX_UNITS = 400
_TR_CACHE = {}


def _units(traj: Trajectory, k: int) -> List[str]:
    """Prefix split into selectable units: one per non-empty step line."""
    out = []
    for s in traj.steps[:k + 1]:
        for ln in _UNIT_SPLIT.split(s.content):
            ln = ln.strip()
            if ln:
                out.append(f"[step {s.idx} {s.agent}] {ln}")
    return out[-MAX_UNITS:]


def _assemble(traj, kept: List[str]) -> str:
    return traj.question + "\n" + "\n".join(kept)


# ---------------------------------------------------------------------------
# P7: TextRank extractive summarisation
# ---------------------------------------------------------------------------
def textrank_scores(units: List[str], damping: float = 0.85,
                    iters: int = 16) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(units) == 1:
        return np.ones(1)
    try:
        X = TfidfVectorizer(lowercase=True, stop_words="english",
                            max_features=20000).fit_transform(units)
    except ValueError:                      # degenerate vocabulary
        return np.ones(len(units))
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0.0)
    row = S.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    P = S / row
    n = len(units)
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r = (1 - damping) / n + damping * (P.T @ r)
    return r


def p7_extractive(traj: Trajectory, k: int, dst: str,
                  budget: Optional[int] = None, **kw) -> PolicyOutput:
    t0 = time.perf_counter()
    b = budget if budget is not None else relay_budget(dst)
    units = _units(traj, k)
    if not units:
        return _finish_b("extractive", traj.question, dst, t0)
    ck = (traj.tid, k)
    r = _TR_CACHE.get(ck)
    if r is None:
        r = textrank_scores(units)
        if len(_TR_CACHE) > 4000:
            _TR_CACHE.clear()
        _TR_CACHE[ck] = r
    used = est_tokens(traj.question)
    keep = np.zeros(len(units), dtype=bool)
    for i in np.argsort(-r):
        c = est_tokens(units[i]) + 2
        if used + c > b:
            continue
        keep[i] = True
        used += c
    text = _assemble(traj, [u for i, u in enumerate(units) if keep[i]])
    return _finish_b("extractive", text, dst, t0, budget=b)


# ---------------------------------------------------------------------------
# P8: Selective-Context / LLMLingua-style self-information pruning
# ---------------------------------------------------------------------------
class SelfInfoScorer:
    """Per-token self-information from a small local causal LM."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B-Base",
                 window: int = 2048, max_tokens: int = 8192):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tk = AutoTokenizer.from_pretrained(model_name)
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16).to(self.dev).eval()
        self.window = window
        self.max_tokens = max_tokens
        self.model_name = model_name

    def token_self_info(self, text: str):
        """(offsets, self_information) for the first ``max_tokens`` tokens."""
        torch = self.torch
        enc = self.tk(text, return_offsets_mapping=True,
                      add_special_tokens=False)
        ids = enc.input_ids[:self.max_tokens]
        offs = enc.offset_mapping[:self.max_tokens]
        si = np.zeros(len(ids), dtype=np.float32)
        step = self.window // 2
        for start in range(0, len(ids), step):
            chunk = ids[start:start + self.window]
            if len(chunk) < 2:
                break
            t = torch.tensor([chunk], device=self.dev)
            with torch.no_grad():
                logits = self.lm(t).logits[0].float()
            lp = torch.log_softmax(logits[:-1], dim=-1)
            tgt = t[0, 1:]
            vals = (-lp.gather(1, tgt[:, None])[:, 0]).cpu().numpy()
            # first token of the very first chunk has no context: give it the
            # chunk mean so it is never pruned for lack of a predecessor
            lo = start + 1
            keep_from = 0 if start == 0 else step - 1
            for j in range(keep_from, len(vals)):
                idx = lo + j
                if idx < len(si):
                    si[idx] = vals[j]
            if start == 0 and len(si):
                si[0] = float(vals.mean())
            if start + self.window >= len(ids):
                break
        return offs, si


def p8_selective(traj: Trajectory, k: int, dst: str, scorer: SelfInfoScorer,
                 budget: Optional[int] = None, **kw) -> PolicyOutput:
    """Drop the lowest-self-information lexical units until the budget fits."""
    t0 = time.perf_counter()
    b = budget if budget is not None else relay_budget(dst)
    units = _units(traj, k)
    if not units:
        return _finish_b("selective", traj.question, dst, t0, budget=b)
    body = "\n".join(units)
    offs, si = scorer.token_self_info(body)
    # pool token self-information onto units (mean over the unit's tokens)
    bounds, pos = [], 0
    for u in units:
        bounds.append((pos, pos + len(u)))
        pos += len(u) + 1
    scores = np.zeros(len(units), dtype=np.float32)
    counts = np.zeros(len(units), dtype=np.float32)
    ui = 0
    for (a, _bz), v in zip(offs, si):
        while ui < len(bounds) - 1 and a >= bounds[ui][1]:
            ui += 1
        scores[ui] += v
        counts[ui] += 1
    mean_si = scores / np.maximum(counts, 1)
    # units the LM never reached (beyond max_tokens) keep the corpus mean so
    # they are neither favoured nor silently discarded
    mean_si[counts == 0] = float(mean_si[counts > 0].mean()) if (counts > 0).any() else 1.0
    used = est_tokens(traj.question)
    keep = np.zeros(len(units), dtype=bool)
    for i in np.argsort(-mean_si):
        c = est_tokens(units[i]) + 2
        if used + c > b:
            continue
        keep[i] = True
        used += c
    text = _assemble(traj, [u for i, u in enumerate(units) if keep[i]])
    return _finish_b("selective", text, dst, t0, budget=b)


def _finish_b(name, text, dst, t0, budget=None) -> PolicyOutput:
    ser = time.perf_counter() - t0
    blob = gzip.compress(text.encode(), 6)
    ctx = usable(dst)
    resume, tok, trunc = _cap(text, ctx)
    return PolicyOutput(name, text, resume, tok, min(tok, ctx), len(blob),
                        ser, trunc)


# ---------------------------------------------------------------------------
# P10: LongLLMLingua-style question-aware contrastive pruning
# ---------------------------------------------------------------------------
def _unit_bounds(units):
    bounds, pos = [], 0
    for u in units:
        bounds.append((pos, pos + len(u)))
        pos += len(u) + 1
    return bounds


def _pool(units, offs, vals):
    """Mean token score per unit; units the LM never reached get the mean."""
    bounds = _unit_bounds(units)
    tot = np.zeros(len(units), dtype=np.float32)
    cnt = np.zeros(len(units), dtype=np.float32)
    ui = 0
    for (a, _b), v in zip(offs, vals):
        while ui < len(bounds) - 1 and a >= bounds[ui][1]:
            ui += 1
        tot[ui] += v
        cnt[ui] += 1
    m = tot / np.maximum(cnt, 1)
    m[cnt == 0] = float(m[cnt > 0].mean()) if (cnt > 0).any() else 1.0
    return m


def p10_llmlingua(traj: Trajectory, k: int, dst: str, scorer: SelfInfoScorer,
                  budget: Optional[int] = None, **kw) -> PolicyOutput:
    """LongLLMLingua's distinctive mechanism: *question-aware* pruning.

    Selective-Context (P8) ranks by self-information alone, which keeps text
    that is surprising whether or not it bears on the task.  LongLLMLingua's
    contribution is the contrast: a unit is worth keeping to the extent the
    *question* makes it less surprising, i.e.

        s(u) = SI(u)  -  SI(u | question)

    is large.  We compute exactly that -- one pass with the question prepended
    and one without -- and rank units by the contrast, keeping the highest
    until the same budget every other policy competes under is full.  Like
    P8 this needs two forward passes per prefix, so it runs on the same
    stratified subsample.

    This is the mechanism, not the published implementation: we do not
    reproduce its budget controller or document reordering, and we say so
    rather than claiming to have run the released system.
    """
    t0 = time.perf_counter()
    b = budget if budget is not None else relay_budget(dst)
    units = _units(traj, k)
    if not units:
        return _finish_b("llmlingua", traj.question, dst, t0, budget=b)
    body = "\n".join(units)
    offs_u, si_u = scorer.token_self_info(body)
    mean_u = _pool(units, offs_u, si_u)

    q = f"Question: {traj.question}\n\n"
    offs_c, si_c = scorer.token_self_info(q + body)
    # re-base the conditioned offsets onto the unconditioned body
    shift = len(q)
    offs_c2, si_c2 = [], []
    for (a, bb), v in zip(offs_c, si_c):
        if a >= shift:
            offs_c2.append((a - shift, bb - shift))
            si_c2.append(v)
    mean_c = _pool(units, offs_c2, np.asarray(si_c2, dtype=np.float32)) \
        if si_c2 else mean_u

    contrast = mean_u - mean_c          # how much the question explains the unit
    used = est_tokens(traj.question)
    keep = np.zeros(len(units), dtype=bool)
    for i in np.argsort(-contrast):
        c = est_tokens(units[i]) + 2
        if used + c > b:
            continue
        keep[i] = True
        used += c
    text = _assemble(traj, [u for i, u in enumerate(units) if keep[i]])
    return _finish_b("llmlingua", text, dst, t0, budget=b)


# ---------------------------------------------------------------------------
# P11: LLM summarisation prompted for the same typed fields
# ---------------------------------------------------------------------------
SUMMARY_PROMPT = (
    "You are preparing a handoff of an in-progress task to another device "
    "with a small context window. Read the transcript and write a compact "
    "resumption state using EXACTLY these sections:\n"
    "GOAL: the task, in one sentence.\n"
    "CONSTRAINTS: bullet list of requirements the answer must satisfy.\n"
    "COMPLETED ACTIONS: bullet list of actions already performed that must "
    "NOT be performed again. Copy identifiers, keys and arguments verbatim.\n"
    "FINDINGS: bullet list of facts established so far. Copy URLs, file "
    "names, numbers and names verbatim.\n"
    "NEXT PLAN: bullet list of the remaining steps.\n\n"
    "Be terse. Do not explain. Do not invent anything not in the transcript.\n\n"
    "--- TRANSCRIPT ---\n{body}\n--- END TRANSCRIPT ---\n\n"
    # The instruction is repeated after the transcript on purpose: a long
    # prefix can push the leading copy out of the summariser's own context,
    # and a baseline that loses its instructions is not the baseline we mean
    # to compare against.
    "Now write the resumption state with the five sections above (GOAL, "
    "CONSTRAINTS, COMPLETED ACTIONS, FINDINGS, NEXT PLAN). Copy URLs, file "
    "names, numbers, identifiers and keys verbatim rather than describing "
    "them.")


def p11_llm_summary(traj: Trajectory, k: int, dst: str, lm,
                    budget: Optional[int] = None, max_new: int = 700,
                    src_ctx: int = 4200, **kw) -> PolicyOutput:
    """A learned summariser told to emit \sys{}'s own field list.

    This is the strongest fair comparator we can build for the *schema*
    claim: it receives the schema as an instruction, so any remaining gap is
    attributable to deterministic construction rather than to Relay knowing
    what to keep.  It is also the baseline with the worst deployment
    properties -- it needs a capable model on the *source* at handoff time,
    which is the moment a wearable handoff can least afford one -- and the
    comparison reports its construction cost alongside its fidelity.

    Two harness details matter for fairness, and getting them wrong is what a
    first run of this experiment did.  ``src_ctx`` bounds the transcript in
    *tokens* so that instruction plus transcript plus output fit the
    summariser's own window; feeding it more than fits silently truncates the
    instruction away.  ``max_new`` is large enough for the summary to reach
    the target's budget rather than stopping mid-sentence; a baseline cut off
    at 420 tokens is not a baseline, it is a bug.
    """
    t0 = time.perf_counter()
    b = budget if budget is not None else relay_budget(dst)
    from tok.tokenizer import cap_suffix
    # bound the transcript in tokens (most recent material, standard practice)
    body, _tot, _tr = cap_suffix(traj.prefix_text(k), src_ctx)
    text = lm.gen(SUMMARY_PROMPT.format(body=body), max_new)
    while exact_tokens_cached(text) > b and len(text) > 200:
        text = text[:int(len(text) * 0.9)]
    return _finish_b("llm-summary", text, dst, t0, budget=b)
