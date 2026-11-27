"""Relay trajectory state model (the paper's Fig. 2), distiller, serializer.

A live trajectory prefix is distilled into a typed StateBlock:
  GOAL        task text + extracted constraint sentences
  REVISIONS   bounded latest-wins records for revised goals/constraints/values
  OPEN_Q      explicitly deferred questions that still block completion
  ENTITIES    anchor -> short context snippet (salience-ranked)
  TOOL_DIGEST per tool-result step: args-hash, bounded summary, full-result ptr
  SIDE_FX     ordered ledger: action fingerprint (agent + canonical head hash),
              status, idempotency key -- ALWAYS shipped, never truncated
  PLAN        most recent explicit plan / next-step text, bounded
  TAIL        last m raw steps (bounded) for local coherence

The distiller sizes the block to a token budget: GOAL and SIDE_FX are
mandatory; PLAN and TAIL are next; the remaining budget is filled by salience
(tool-result anchors ranked above plain entities; within a tier, by
recency x frequency), matching SPEC Section 'Handoff policies'.
"""
import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from traces.parser import Trajectory, est_tokens, exact_tokens_cached
from traces.anchors import AnchorIndex, extract_anchors_cached as extract_anchors, norm_text

CONSTRAINT_RE = re.compile(
    r"[^.\n]*\b(must|at least|at most|no more than|between|within|before|"
    r"after|under \$|over \$|only|rounded|nearest|alphabetical|comma.separated|"
    r"give (me|them|your)|answer (in|using|with)|express your answer|format)\b[^.\n]*",
    re.I)

# Structural value bonuses in "anchor-equivalent" units, used by the
# budget allocator.  They encode what E6 measures: the plan and the tool
# digest carry exact state the anchor metric is blind to, so a purely
# anchor-density fill would delete them.
W_PLAN, W_DIGEST, W_TAIL = 2.0, 0.6, 0.5

PLAN_HINT_RE = re.compile(
    r"(plan for solving|here is the plan|### plan|## plan|^plan:|next step|"
    r"step-by-step plan|remaining steps|we will:|i will|let's proceed)",
    re.I | re.M)


@dataclass
class ToolDigestEntry:
    step: int
    agent: str
    args_hash: str      # sha1 of normalized step head (proxy for call args)
    summary: str        # bounded summary: head chars + anchor-bearing lines
    ptr: str            # pointer to full result (trace id + step)


@dataclass
class SideFxEntry:
    step: int
    agent: str
    fingerprint: str    # tool + canonical-args hash
    status: str         # "done"
    idem_key: str       # idempotency key: trace-scoped stable id
    head: str           # bounded normalized head of the action text


@dataclass
class RevisionEntry:
    """A later trajectory statement that supersedes earlier state.

    ``kind`` is deliberately coarse: the runtime needs latest-wins semantics
    before it needs an ontology.  The original step is retained for provenance,
    and the bounded text keeps both the replacement and what it supersedes
    visible to the resumed model.
    """
    step: int
    kind: str           # goal | constraint | value | other
    text: str


@dataclass
class EntityGroup:
    """Salient anchors that were observed in the same source step, sharing
    one context window.

    Per-anchor snippets are the obvious encoding and the wasteful one: a
    60-character window costs ~4x the anchor it explains, and co-occurring
    anchors repeat nearly the same window.  Grouping by source step both
    amortises the context and preserves the relation between anchors that
    were observed together (``CHN 4, JPN 43`` is more useful than two
    independent codes).
    """
    step: int
    agent: str
    anchors: List[str]
    snippet: str


@dataclass
class StateBlock:
    goal: str
    constraints: List[str] = field(default_factory=list)
    revisions: List[RevisionEntry] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)
    entity_groups: List[EntityGroup] = field(default_factory=list)
    tool_digest: List[ToolDigestEntry] = field(default_factory=list)
    side_fx: List[SideFxEntry] = field(default_factory=list)
    plan: str = ""
    tail: List[str] = field(default_factory=list)
    #: conservative mode only: later turns, verbatim, that restate a constrained
    #: value or name a goal entity without any revision cue firing
    updates: List[str] = field(default_factory=list)

    # ---------------- rendering / serialization ----------------
    def render(self) -> str:
        """Human/model-readable resumption context (what the target model is
        prefilled with; fidelity is measured on this text)."""
        parts = ["== GOAL ==", self.goal]
        if self.constraints:
            parts.append("== CONSTRAINTS ==" if not self.updates else
                         "== CONSTRAINTS (from the task statement; not "
                         "authoritative, see later statements) ==")
            parts += [f"- {c}" for c in self.constraints]
        if self.updates:
            parts.append("== LATER STATEMENTS (verbatim; may supersede the "
                         "goal and constraints) ==")
            parts += list(self.updates)
        if self.revisions:
            parts.append("== ACTIVE REVISIONS (latest statement wins) ==")
            for r in self.revisions:
                parts.append(f"- [step {r.step} {r.kind}] {r.text}")
        if self.open_questions:
            parts.append("== OPEN QUESTIONS (unresolved; do not assume) ==")
            parts += [f"- {q}" for q in self.open_questions]
        if self.tool_digest:
            parts.append("== TOOL RESULTS (digest) ==")
            for d in self.tool_digest:
                parts.append(f"[step {d.step} {d.agent} args#{d.args_hash} "
                             f"full@{d.ptr}] {d.summary}")
        if self.entity_groups:
            parts.append("== ENTITIES ==")
            for g in self.entity_groups:
                line = f"- [step {g.step} {g.agent}] " + "; ".join(g.anchors)
                if g.snippet:
                    line += f" :: {g.snippet}"
                parts.append(line)
        if self.entities:
            parts.append("== ENTITIES ==" if not self.entity_groups
                         else "== ENTITIES (cont.) ==")
            for a, snip in self.entities.items():
                parts.append(f"- {a}: {snip}" if snip else f"- {a}")
        if self.plan:
            parts.append("== PENDING PLAN ==")
            parts.append(self.plan)
        if self.tail:
            parts.append("== DIALOGUE TAIL ==")
            parts += self.tail
        # The ledger is rendered LAST, and that ordering is a safety decision
        # rather than a stylistic one: a target applies its window as a
        # *drop-oldest* cap (``tok.tokenizer.cap_suffix``), so whatever sits at
        # the front of the block is what a too-small window evicts first.
        # Putting the one non-evictable section at the front -- which the first
        # version of this system did -- means the only section that must
        # survive truncation is the first to be thrown away.
        if self.side_fx:
            parts.append("== SIDE-EFFECT LEDGER (do NOT re-execute) ==")
            for e in self.side_fx:
                if e.head:
                    parts.append(f"[{e.idem_key}|{e.fingerprint}|{e.status}] "
                                 f"step {e.step} {e.agent}: {e.head}")
                else:
                    # saturated form: identity only.  Provenance is prose and
                    # goes the way of the head; the key and the fingerprint are
                    # what the dispatcher matches on and never go.
                    parts.append(f"[{e.idem_key}|{e.fingerprint}|{e.status}]")
        return "\n".join(parts)

    def serialize(self) -> bytes:
        """Wire format: compact JSON, gzipped."""
        d = {"goal": self.goal, "constraints": self.constraints,
             "revisions": [asdict(r) for r in self.revisions],
             "open_questions": self.open_questions,
             "entities": self.entities,
             "entity_groups": [asdict(g) for g in self.entity_groups],
             "tool_digest": [asdict(t) for t in self.tool_digest],
             "side_fx": [asdict(s) for s in self.side_fx],
             "plan": self.plan, "tail": self.tail}
        if self.updates:
            d["updates"] = self.updates
        return gzip.compress(json.dumps(d, separators=(",", ":"),
                                        ensure_ascii=False).encode(), 6)

    @staticmethod
    def deserialize(blob: bytes) -> "StateBlock":
        d = json.loads(gzip.decompress(blob).decode())
        return StateBlock(
            goal=d["goal"], constraints=d["constraints"],
            revisions=[RevisionEntry(**r) for r in d.get("revisions", [])],
            open_questions=d.get("open_questions", []),
            entities=d["entities"],
            entity_groups=[EntityGroup(**g)
                           for g in d.get("entity_groups", [])],
            tool_digest=[ToolDigestEntry(**t) for t in d["tool_digest"]],
            side_fx=[SideFxEntry(**s) for s in d["side_fx"]],
            plan=d["plan"], tail=d["tail"], updates=d.get("updates", []))

    def tokens(self) -> int:
        return est_tokens(self.render())


# --------------------------------------------------------------------------
# Distiller
# --------------------------------------------------------------------------

def _head(text: str, n: int) -> str:
    return re.sub(r"\s+", " ", text)[:n].strip()


def _anchor_lines(text: str, anchors, max_lines: int = 3,
                  line_cap: int = 140) -> List[str]:
    """Lines of the step that contain at least one anchor (result payloads)."""
    out = []
    for line in text.splitlines():
        ln = norm_text(line)
        if not ln:
            continue
        if any(a in ln for a in anchors):
            out.append(_head(line, line_cap))
            if len(out) >= max_lines:
                break
    return out


def extract_constraints(question: str, cap: int = 5) -> List[str]:
    out = []
    for m in CONSTRAINT_RE.finditer(question):
        s = _head(m.group(0), 160)
        if len(s) > 15 and s not in out:
            out.append(s)
        if len(out) >= cap:
            break
    return out


REVISION_RE = re.compile(
    r"\b(update from|change of plan|correction|corrected|supersedes?|"
    r"no longer applies|ignore the earlier|was wrong|earlier .{0,80} stale)\b",
    re.I)
OPEN_QUESTION_RE = re.compile(
    r"\b(open question|awaiting (?:a )?reply|awaiting confirmation|"
    r"has not (?:said|confirmed)|do not .{0,80} until)\b", re.I)


def build_revision_state(prefix_steps, revision_cap: int = 4,
                         question_cap: int = 4):
    """Extract bounded, provenance-carrying latest-wins state.

    This intentionally recognizes explicit correction language only.  An
    implicit change remains out of schema rather than being guessed by the
    distiller.  For each coarse kind the latest explicit revision replaces the
    earlier one, which prevents stale constraints from accumulating under an
    authoritative heading.
    """
    latest = {}
    questions = []
    for s in prefix_steps:
        text = _head(s.content, 220)
        if REVISION_RE.search(s.content):
            low = s.content.lower()
            if "goal" in low or "change of plan" in low or "destination" in low:
                kind = "goal"
            elif "constraint" in low or "deadline" in low or "cut-off" in low \
                    or "must stay" in low:
                kind = "constraint"
            elif "supersed" in low or "stale" in low or "recheck" in low:
                kind = "value"
            else:
                kind = "other"
            latest[kind] = RevisionEntry(s.idx, kind, text)
        if OPEN_QUESTION_RE.search(s.content):
            questions.append(f"[step {s.idx} {s.agent}] {text}")
    revisions = sorted(latest.values(), key=lambda r: r.step)[-revision_cap:]
    return revisions, questions[-question_cap:]


#: Constrained-value types recognised without any natural-language cue: the
#: symbols and digit shapes survive paraphrase and, largely, translation.
TYPED_VALUE_RES = {
    "currency": re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d+)?"),
    "clock": re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s?(?:am|pm)\b", re.I),
    "date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "percent": re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    "temperature": re.compile(r"\b\d+(?:\.\d+)?\s?\u00b0?[CF]\b"),
}


def _typed_values(text: str) -> Dict[str, set]:
    out = {}
    for name, rx in TYPED_VALUE_RES.items():
        vals = {re.sub(r"[\s,]", "", m.group(0).lower())
                for m in rx.finditer(text)}
        if vals:
            out[name] = vals
    return out


def conservative_updates(question: str, prefix_steps, cap: int = 3,
                         char_cap: int = 300) -> List[str]:
    """Later turns that could restate a goal or constraint, shipped verbatim.

    The revision ledger recognises explicit correction language, and a
    correction worded without it is silently lost while the task statement's
    constraints ship under an authoritative heading.  Conservative mode does
    not try to understand the correction.  It ships, verbatim, the turns that
    *could* carry one -- non-tool, non-action, non-plan turns in which no
    revision cue fired and which either restate a value of a type the task
    statement constrains (currency, clock time, date, percentage, temperature)
    or name an entity of the task statement -- and relabels the constraints as
    non-authoritative.  Turns restating a *different* value of a constrained
    type rank first; ties go to the most recent.  It cannot catch a correction
    that names neither (e.g. "skip the second errand").
    """
    q_types = _typed_values(question)
    q_ents = {a for a, t in extract_anchors(question).items()
              if t == "entity"}
    scored = []
    for s_ in prefix_steps:
        if s_.is_tool_result or s_.side_effect_like:
            continue
        if REVISION_RE.search(s_.content) or PLAN_HINT_RE.search(s_.content):
            continue
        vals = _typed_values(s_.content)
        new_val = any(v not in q_types[t] for t, vs in vals.items()
                      if t in q_types for v in vs)
        same_type = any(t in q_types for t in vals)
        low = norm_text(s_.content)
        ent = any(e in low for e in q_ents)
        score = 2 if new_val else (1 if (same_type or ent) else 0)
        if score:
            scored.append((score, s_.idx, s_))
    top = sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)[:cap]
    return [f"[step {s_.idx} {s_.agent}] {_head(s_.content, char_cap)}"
            for _, _, s_ in sorted(top, key=lambda x: x[1])]


# An action that reports its own idempotency key already has a stable
# identity assigned upstream; a ledger that minted a fresh one would lose
# the only token the executing service will recognise on retry.
IDEM_RE = re.compile(
    r"idempotency[-_ ]?key[\s:=\[]*([A-Za-z0-9_.:-]{4,64})", re.I)


def build_side_fx(traj: Trajectory, k: int) -> List[SideFxEntry]:
    out = []
    for s in traj.steps[:k + 1]:
        if s.side_effect_like:
            fp = s.fingerprint()
            m = IDEM_RE.search(s.content)
            key = m.group(1) if m else f"{traj.tid}:{s.idx}:{fp[:8]}"
            out.append(SideFxEntry(
                step=s.idx, agent=s.agent, fingerprint=fp, status="done",
                idem_key=key, head=_head(s.content, 110)))
    return out


class Distiller:
    """Builds budgeted StateBlocks from real-trace prefixes."""

    def __init__(self, idx: AnchorIndex, tail_m: int = 2,
                 tail_cap: int = 400, summary_cap: int = 200,
                 group_entities: bool = True, group_cap: int = 8,
                 group_snippet: int = 90, allocator: str = "density",
                 value_types=None, select_types=None,
                 conservative: bool = False):
        self.idx = idx
        # conservative mode: ship uncued, possibly-superseding turns verbatim
        # and mark the task statement's constraints non-authoritative
        self.conservative = conservative
        # Anchor types the allocator is allowed to *count* (``value_types``)
        # and to *select as entities* (``select_types``).  ``None`` means all,
        # which is the deployed configuration.  E14 restricts them to a set
        # disjoint from the one the metric scores, so that "the allocator
        # optimises what the metric rewards" stops being a confound rather
        # than merely being argued about.
        self.value_types = set(value_types) if value_types else None
        self.select_types = set(select_types) if select_types else None
        self.tail_m = tail_m
        self.tail_cap = tail_cap
        self.summary_cap = summary_cap
        # step-grouped entity encoding (Sec. IV); ``False`` restores the
        # per-anchor-snippet encoding the ablation compares against
        self.group_entities = group_entities
        self.group_cap = group_cap
        self.group_snippet = group_snippet
        # "density": fill by value/cost (Sec. IV); "fixed": the fixed section
        # priority plan -> tail -> digest -> entities that the ablation
        # compares against
        self.allocator = allocator

    def distill(self, traj: Trajectory, k: int,
                budget_tokens: Optional[int] = None,
                drop: frozenset = frozenset()) -> StateBlock:
        """drop: components to ablate, subset of
        {'entities','tool_digest','plan','tail','side_fx','constraints',
         'revisions','open_questions'}."""
        prefix_steps = traj.steps[:k + 1]
        # normalized prefix computed once per (traj, k) for snippet lookups
        ck = (traj.tid, k)
        if getattr(self, "_np_key", None) != ck:
            self._np_key = ck
            self._np_text = norm_text(traj.prefix_text(k))

        goal = _head(traj.question, 600)
        constraints = [] if "constraints" in drop else \
            extract_constraints(traj.question)
        revisions, open_questions = build_revision_state(prefix_steps)
        updates = conservative_updates(traj.question, prefix_steps) \
            if self.conservative else []
        if "revisions" in drop:
            revisions = []
        if "open_questions" in drop:
            open_questions = []
        side_fx = [] if "side_fx" in drop else build_side_fx(traj, k)

        # per-step anchors with salience
        step_anchors: List[Dict[str, str]] = []
        freq: Dict[str, int] = {}
        last_seen: Dict[str, int] = {}
        typ: Dict[str, str] = {}
        from_tool: Dict[str, bool] = {}
        for s in prefix_steps:
            A = {a: t for a, t in extract_anchors(s.content).items()
                 if self.idx.keep(a)}
            step_anchors.append(A)
            for a, t in A.items():
                freq[a] = freq.get(a, 0) + 1
                last_seen[a] = s.idx
                typ[a] = t
                if s.is_tool_result:
                    from_tool[a] = True
        qa = {a: t for a, t in extract_anchors(traj.question).items()
              if self.idx.keep(a)}
        for a, t in qa.items():
            freq[a] = freq.get(a, 0) + 1
            last_seen.setdefault(a, 0)
            typ[a] = t

        # tool digest entries (most recent first for budget fill)
        digest: List[ToolDigestEntry] = []
        if "tool_digest" not in drop:
            for s in reversed(prefix_steps):
                if not s.is_tool_result:
                    continue
                A = step_anchors[s.idx] if s.idx < len(step_anchors) else {}
                lines = _anchor_lines(s.content, list(A), 3)
                summary = _head(s.content, self.summary_cap)
                extra = " | ".join(l for l in lines
                                   if l not in summary)[:self.summary_cap]
                if extra:
                    summary = summary + " ... " + extra
                digest.append(ToolDigestEntry(
                    step=s.idx, agent=s.agent,
                    args_hash=s.fingerprint()[:8], summary=summary,
                    ptr=f"{traj.tid}#{s.idx}"))

        # entity candidates by salience: tool-tier first, then
        # recency x frequency within tier
        ents: List[str] = []
        if "entities" not in drop:
            def salience(a):
                rec = (last_seen.get(a, 0) + 1) / (k + 1)
                return (1 if from_tool.get(a) else 0, rec * freq.get(a, 0))
            pool = freq if self.select_types is None else \
                [a for a in freq if typ.get(a) in self.select_types]
            ents = sorted(pool, key=salience, reverse=True)

        plan, plan_step = "", None
        if "plan" not in drop:
            for s in reversed(prefix_steps):
                m = PLAN_HINT_RE.search(s.content)
                if m:
                    plan = _head(s.content[m.start():], 500)
                    plan_step = s.idx
                    break

        tail: List[str] = []
        tail_steps: List[int] = []
        if "tail" not in drop:
            for s in prefix_steps[-self.tail_m:]:
                tail.append(f"[step {s.idx} {s.agent}] "
                            f"{_head(s.content, self.tail_cap)}")
                tail_steps.append(s.idx)

        # ---------------- budget fitting ----------------
        block = StateBlock(goal=goal, constraints=constraints,
                           revisions=revisions,
                           open_questions=open_questions,
                           side_fx=side_fx, plan="", tail=[],
                           updates=updates)
        if budget_tokens is None:
            # unbudgeted: include everything (used by E7 natural-size scaling)
            block.plan, block.tail = plan, list(tail)
            block.tool_digest = list(reversed(digest))
            if self.group_entities:
                self._fill_entity_groups(traj, k, block, ents, last_seen,
                                         salience, 0, 10 ** 9)
            else:
                for a in ents:
                    block.entities[a] = self._snippet(traj, k, a)
            return block

        used = block.tokens()
        if used > budget_tokens:
            # mandatory sections alone exceed the budget: shrink side-effect
            # heads, but never the fingerprints or idempotency keys -- safety
            # metadata has strict priority over prose (E8 measures where this
            # saturates).
            for e in block.side_fx:
                e.head = _head(e.head, 40)
            used = block.tokens()

        # ---- candidate items, ranked by value density ----
        # Value is expressed in "anchor-equivalents": the number of distinct
        # salient anchors an item carries, plus a structural bonus for the
        # sections that carry exact state the anchor metric cannot see (the
        # pending plan, tool-result pointers, the dialogue tail).  Allocating
        # by value/cost rather than by a fixed section order is what stops a
        # verbose digest from starving the entity roster.
        items = []
        anchor_rank = {a: i for i, a in enumerate(ents)}

        # anchors carried by a step are already known from the extraction
        # pass above, so item value needs no extra scanning
        def _n_at(step_idx) -> float:
            if step_idx is None or step_idx >= len(step_anchors):
                return 0.0
            A_ = step_anchors[step_idx]
            if self.value_types is None:
                return float(len(A_))
            return float(sum(1 for _a, _t in A_.items()
                             if _t in self.value_types))

        if plan:
            items.append(("plan", est_tokens(plan) + 3,
                          W_PLAN + _n_at(plan_step), plan))
        for ts_, si in zip(tail, tail_steps):
            items.append(("tail", est_tokens(ts_) + 2,
                          W_TAIL + _n_at(si), ts_))
        for d in digest:
            short = ToolDigestEntry(d.step, d.agent, d.args_hash,
                                    _head(d.summary, 110), d.ptr)
            v = W_DIGEST + _n_at(d.step)
            items.append(("digest", est_tokens(d.summary) + 14, v, d))
            items.append(("digest", est_tokens(short.summary) + 14,
                          v * 0.75, short))
        groups = self._build_groups(traj, k, ents, last_seen)
        for g, cost in groups:
            val = sum(1.0 / (1 + anchor_rank.get(a, 999) / 50.0)
                      for a in g.anchors
                      if self.value_types is None or
                      typ.get(a) in self.value_types)
            items.append(("entities", cost, val + 0.05 * len(g.anchors), g))

        # structural floors: the plan, the most recent tool result, and one
        # tail step are added first when they fit at all, so that a purely
        # density-ranked fill cannot delete the structured state that E6
        # shows the probe metric depends on.
        chosen, seen_steps = [], set()
        def _take(kind, cost, val, payload):
            nonlocal used
            if used + cost > budget_tokens:
                return False
            used += cost
            chosen.append((kind, payload))
            return True

        floors = [i for i in items if i[0] == "plan"][:1]
        floors += sorted([i for i in items if i[0] == "digest"],
                         key=lambda i: (-i[3].step, i[1]))[:1]
        floors += [i for i in items if i[0] == "tail"][-1:]
        for it in floors:
            _take(*it)
        taken = {id(i[3]) for i in floors}

        rest = [i for i in items if id(i[3]) not in taken]
        if self.allocator == "density":
            rest.sort(key=lambda i: i[2] / max(i[1], 1), reverse=True)
        else:
            sec = {"plan": 0, "tail": 1, "digest": 2, "entities": 3}
            rest.sort(key=lambda i: (sec[i[0]],
                                     -getattr(i[3], "step", 0)))
        for kind, cost, val, payload in rest:
            if kind == "digest":
                if payload.step in seen_steps:
                    continue
            if _take(kind, cost, val, payload):
                if kind == "digest":
                    seen_steps.add(payload.step)

        for kind, payload in chosen:
            if kind == "plan":
                block.plan = payload
            elif kind == "tail":
                block.tail.append(payload)
            elif kind == "digest":
                if payload.step not in {d.step for d in block.tool_digest}:
                    block.tool_digest.append(payload)
            else:
                block.entity_groups.append(payload)
        block.tool_digest.sort(key=lambda d: d.step)
        block.entity_groups.sort(key=lambda g: g.step)
        return self._verify_exact(block, budget_tokens)

    # ------------------------------------------------------------------
    def _build_groups(self, traj, k, ents, last_seen):
        """Candidate entity groups: salient anchors bucketed by their most
        recent source step, each with one shared context window."""
        order, by_step = [], {}
        for a in ents:
            st = last_seen.get(a, 0)
            if st not in by_step:
                by_step[st] = []
                order.append(st)
            by_step[st].append(a)
        agent_of = {s.idx: s.agent for s in traj.steps[:k + 1]}
        out = []
        cap = self.group_cap if self.group_entities else 1
        for st in order:
            members = by_step[st]
            for c in range(0, len(members), cap):
                chunk = members[c:c + cap]
                snippet = self._snippet(traj, k, chunk[0],
                                        width=self.group_snippet)
                g = EntityGroup(step=st, agent=agent_of.get(st, "task"),
                                anchors=chunk, snippet=snippet)
                cost = est_tokens(f"- [step {st} {g.agent}] "
                                  + "; ".join(chunk) + " :: " + snippet) + 2
                out.append((g, cost))
        return out

    def _fill_entity_groups(self, traj, k, block, ents, last_seen, salience,
                            used, budget_tokens):
        for g, cost in self._build_groups(traj, k, ents, last_seen):
            if used + cost > budget_tokens:
                break
            block.entity_groups.append(g)
            used += cost
        return used

    def _verify_exact(self, block: "StateBlock",
                      budget_tokens: Optional[int]) -> "StateBlock":
        """Exact-BPE verification: the fill loop uses a conservative
        chars/3.5 estimate, so a block should be under-full -- but the
        estimate is not a guarantee and the target window is hard.

        The trim order encodes the safety priority.  Optional material goes
        first (entities, then digest, then tail).  If that is not enough the
        *goal* is clipped, and only then are ledger heads shortened; the
        fingerprints and idempotency keys are never touched.  A block that
        leaves here over budget would be truncated at the target by a
        drop-oldest cap, which evicts the ledger -- exactly the failure the
        mandatory ledger exists to prevent -- so the loop runs until it fits
        rather than for a fixed number of steps.  (An earlier version capped
        it at 64 iterations, which was not enough for a block with many
        entity groups; the resulting overflow was the source of every
        residual real-trace hazard we measured.)
        """
        if budget_tokens is None:
            return block

        def _drop_one() -> int:
            """Remove the least-salient remaining item; return est. tokens freed.

            Order encodes the safety priority: optional material first, then
            the goal, and only then ledger *heads*.  Returns 0 when there is
            nothing left to give.
            """
            if block.entities:
                a_, snip = block.entities.popitem()
                return est_tokens(f"- {a_}: {snip}") + 1
            if block.entity_groups:
                g = block.entity_groups[-1]
                if len(g.anchors) > 1:
                    return est_tokens(g.anchors.pop()) + 1
                block.entity_groups.pop()
                return est_tokens("- [step] " + "; ".join(g.anchors)
                                  + " :: " + g.snippet) + 2
            if block.tool_digest:
                d_ = block.tool_digest.pop()
                return est_tokens(d_.summary) + 14
            if block.tail:
                return est_tokens(block.tail.pop()) + 2
            if len(block.updates) > 1:
                return est_tokens(block.updates.pop(0)) + 2
            if block.constraints:
                return est_tokens(block.constraints.pop()) + 2
            if len(block.open_questions) > 1:
                return est_tokens(block.open_questions.pop(0)) + 2
            if len(block.revisions) > 1:
                r_ = block.revisions.pop(0)
                return est_tokens(r_.text) + 6
            if len(block.goal) > 120:
                before = est_tokens(block.goal)
                block.goal = _head(block.goal, max(120, len(block.goal) // 2))
                return before - est_tokens(block.goal)
            # last resort: shorten, then drop, the ledger *heads*.  The
            # fingerprint and the idempotency key are never touched -- they
            # are what the dispatcher matches on, and a head is only prose.
            freed = 0
            for e in block.side_fx:
                if len(e.head) > 12:
                    before = est_tokens(e.head)
                    e.head = _head(e.head, max(12, len(e.head) // 2))
                    freed += before - est_tokens(e.head)
            if freed:
                return freed
            for e in block.side_fx:
                if e.head:
                    freed += est_tokens(e.head) + 1
                    e.head = ""
            return freed

        # Bulk-drop against the cheap estimate, then verify exactly.  Dropping
        # one item per exact re-render is correct and quadratic; the estimate
        # is only used to decide *how much* to remove before paying for a
        # render, never to decide whether the block fits.
        for _round in range(64):
            exact = exact_tokens_cached(block.render())
            if exact <= budget_tokens:
                return block
            over = exact - budget_tokens
            freed = 0
            while freed < over + 8:
                got = _drop_one()
                if got <= 0:
                    return block          # nothing left to give
                freed += got
        return block

    def _snippet(self, traj: Trajectory, k: int, anchor: str,
                 width: int = 60) -> str:
        text = self._np_text
        i = text.find(anchor)
        if i < 0:
            return ""
        return text[max(0, i - width // 2): i + len(anchor) + width]
