"""Handoff policies P1--P6.

Every policy maps (trajectory, handoff step k, target device) to shipped
content. All policies respect the same target budget at RESUMPTION time:
the wire carries whatever the policy chooses to ship (that is what costs
bytes/energy), and the target keeps only the most recent
``usable(dst) = ctx - headroom(dst)`` tokens of it (drop-oldest) -- so
full-context on a watch is silently truncated, which is exactly the failure
mode the paper measures.

The generation headroom is applied to *every* policy, not just to Relay: a
resumption context that fills the whole window leaves the resumed model no
room to emit anything, so "ship until the window is full" is not a
deployable baseline, and charging Relay for headroom while exempting the
baselines would understate them nowhere and overstate them everywhere.

  P1 full       -- ship the entire raw prefix (question + all steps)
  P2 trunc25    -- ship the last 25% of steps (naive recency heuristic)
  P3 recency    -- goal + as many most-recent raw steps as fit target ctx
  P4 relay      -- schema-distilled StateBlock sized to the target class's
                   usable window (ctx - generation headroom); the block only
                   grows to what salience demands, so short trajectories
                   still ship far less
  P4b relay-lite - the same distiller at half the budget: the stall-optimised
                   operating point of the same knob
  P5 restart    -- goal only (restart-with-goal)
  P6 randdrop   -- random subset of steps matching Relay's token budget
                   (byte-matched control)
"""
import gzip
import random
import time
from dataclasses import dataclass
from typing import Optional

from traces.parser import Trajectory, est_tokens, exact_tokens_cached
from tok.tokenizer import cap_suffix
from devices.constants import DEVICES
from state.schema import Distiller


@dataclass
class PolicyOutput:
    policy: str
    wire_text: str        # what is shipped
    resume_text: str      # what the target holds after its ctx cap
    tokens_wire: int
    tokens_resume: int
    bytes_wire: int       # gzipped wire size
    serialize_cpu_s: float
    truncated_at_target: bool


def _cap(text: str, ctx: int):
    """Apply the target's usable window (drop-oldest), on exact BPE tokens."""
    return cap_suffix(text, ctx)


def _finish(name, text, dst, t0) -> PolicyOutput:
    ser = time.perf_counter() - t0
    blob = gzip.compress(text.encode(), 6)
    ctx = usable(dst)
    resume, tok, trunc = _cap(text, ctx)
    return PolicyOutput(name, text, resume, tok, min(tok, ctx), len(blob),
                        ser, trunc)


def _steps_text(traj, steps):
    return "\n".join(f"[step {s.idx} {s.agent}] {s.content}" for s in steps)


def headroom(dst: str) -> int:
    """Tokens reserved on the target for its own generation."""
    return max(256, DEVICES[dst]["ctx"] // 8)


def usable(dst: str) -> int:
    """Resumption tokens a target of this class can actually accept."""
    return DEVICES[dst]["ctx"] - headroom(dst)


def relay_budget(dst: str) -> int:
    """Deployed budget: the target's usable resumption window.

    The block is *bounded* by this, not sized to it -- the distiller stops
    when salient state runs out, so short trajectories still ship a small
    block (E7).  Earlier versions of this system used a fixed ctx/2, which
    left half of a wearable window unused precisely on the long
    trajectories where the budget binds (E2).
    """
    return usable(dst)


def relay_lite_budget(dst: str) -> int:
    """Stall-optimised operating point of the same knob."""
    return max(128, relay_budget(dst) // 2)


def p1_full(traj: Trajectory, k: int, dst: str, **kw) -> PolicyOutput:
    t0 = time.perf_counter()
    text = traj.question + "\n" + _steps_text(traj, traj.steps[:k + 1])
    return _finish("full", text, dst, t0)


def p2_trunc25(traj: Trajectory, k: int, dst: str, **kw) -> PolicyOutput:
    t0 = time.perf_counter()
    n = k + 1
    keep = max(1, n // 4)
    text = _steps_text(traj, traj.steps[n - keep:k + 1])
    return _finish("trunc25", text, dst, t0)


def p3_recency(traj: Trajectory, k: int, dst: str, **kw) -> PolicyOutput:
    t0 = time.perf_counter()
    ctx = usable(dst)
    budget = ctx - est_tokens(traj.question) - 8
    chosen = []
    used = 0
    for s in reversed(traj.steps[:k + 1]):
        t = s.tokens + 6
        if used + t > budget and chosen:
            break
        if used + t > budget:
            break
        chosen.append(s)
        used += t
    chosen.reverse()
    text = traj.question + "\n" + _steps_text(traj, chosen)
    return _finish("recency", text, dst, t0)


def p4_relay(traj: Trajectory, k: int, dst: str,
             distiller: Distiller = None, budget: Optional[int] = None,
             drop=frozenset(), **kw) -> PolicyOutput:
    t0 = time.perf_counter()
    b = budget if budget is not None else relay_budget(dst)
    block = distiller.distill(traj, k, b, drop=drop)
    blob = block.serialize()          # actual wire format (typed, gzipped)
    text = block.render()
    ser = time.perf_counter() - t0
    ctx = usable(dst)
    resume, tok, trunc = _cap(text, ctx)
    return PolicyOutput(kw.get("name", "relay"), text, resume, tok,
                        min(tok, ctx), len(blob), ser, trunc)


def p4b_relay_lite(traj: Trajectory, k: int, dst: str,
                   distiller: Distiller = None, budget: Optional[int] = None,
                   **kw) -> PolicyOutput:
    b = budget if budget is not None else relay_lite_budget(dst)
    return p4_relay(traj, k, dst, distiller=distiller, budget=b,
                    name="relay-lite", **kw)


def p9_typed_recency(traj: Trajectory, k: int, dst: str,
                     budget: Optional[int] = None, **kw) -> PolicyOutput:
    """P9 typed-recency: the schema's mandatory core, then recency.

    This baseline exists to answer a question the ablations could not: how
    much of Relay's margin comes from *typing* the state at all (shipping a
    goal and a non-evictable ledger) rather than from value-density
    allocation over typed candidates?  It reserves exactly what Relay
    reserves -- goal, constraints, side-effect ledger -- and then spends the
    entire remaining budget on the most recent raw steps, which is the
    strongest thing a policy can do without a salience model.

    A margin that survives this baseline is attributable to allocation; a
    margin that does not is attributable to the schema, and either answer is
    worth reporting.
    """
    t0 = time.perf_counter()
    from state.schema import build_side_fx, extract_constraints, _head
    b = budget if budget is not None else relay_budget(dst)
    cons = extract_constraints(traj.question)
    sfx = build_side_fx(traj, k)

    def _core(head_cap=None):
        parts = ["== GOAL ==", _head(traj.question, 600)]
        if cons:
            parts.append("== CONSTRAINTS ==")
            parts += [f"- {c}" for c in cons]
        if sfx:
            parts.append("== SIDE-EFFECT LEDGER (do NOT re-execute) ==")
            for e in sfx:
                h = _head(e.head, head_cap) if head_cap else e.head
                parts.append(f"[{e.idem_key}|{e.fingerprint}|{e.status}] "
                             f"step {e.step} {e.agent}: {h}")
        return "\n".join(parts)

    core = _core()
    if est_tokens(core) > b:
        # mandatory core over budget: clip ledger heads, exactly as the
        # distiller does -- never the fingerprints or the idempotency keys.
        core = _core(head_cap=40)
    used = est_tokens(core)

    chosen = []
    for s in reversed(traj.steps[:k + 1]):
        t = s.tokens + 6
        if used + t > b:
            break
        chosen.append(s)
        used += t
    chosen.reverse()

    def _render(steps):
        return core + ("\n== RECENT STEPS ==\n" + _steps_text(traj, steps)
                       if steps else "")

    text = _render(chosen)
    # exact-BPE guard, same discipline as the distiller's _verify_exact
    while chosen and exact_tokens_cached(text) > b:
        chosen.pop(0)
        text = _render(chosen)
    return _finish("typed-recency", text, dst, t0)


def p5_restart(traj: Trajectory, k: int, dst: str, **kw) -> PolicyOutput:
    t0 = time.perf_counter()
    return _finish("restart", traj.question, dst, t0)


def p6_randdrop(traj: Trajectory, k: int, dst: str,
                seed: int = 0, budget: Optional[int] = None,
                **kw) -> PolicyOutput:
    t0 = time.perf_counter()
    b = budget if budget is not None else relay_budget(dst)
    rng = random.Random(f"{seed}|{traj.tid}|{k}")   # str seed: see explib
    order = list(range(k + 1))
    rng.shuffle(order)
    used, keep = 0, []
    for i in order:
        t = traj.steps[i].tokens + 6
        if used + t > b:
            continue
        keep.append(i)
        used += t
    keep.sort()
    text = _steps_text(traj, [traj.steps[i] for i in keep])
    return _finish("randdrop", text, dst, t0)


POLICIES = {"full": p1_full, "trunc25": p2_trunc25, "recency": p3_recency,
            "relay": p4_relay, "relay-lite": p4b_relay_lite,
            "restart": p5_restart, "randdrop": p6_randdrop,
            "typed-recency": p9_typed_recency}
POLICY_ORDER = ["full", "trunc25", "recency", "typed-recency", "relay",
                "relay-lite", "restart", "randdrop"]
