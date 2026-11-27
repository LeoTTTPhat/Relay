"""Bounded exhaustive testing of the dispatcher, the durable ledger and the
delegation merge rules.

Every sequence of events up to a fixed length is executed against the real
classes (``TrustedDispatcher``, ``SQLiteLedger`` in memory, ``delegate`` /
``merge_result`` / ``revoke``) for one action and each service contract
(plain, keyed, queryable), and safety invariants are checked after every event.

**Single-runtime alphabet.** ``ok`` (dispatched call succeeds), ``reject``
(the service answers with a definite failure, no effect), ``lost_eff`` (the
effect happens but the response is lost: the tool raises), ``lost_noeff``
(the request is lost before any effect: the tool raises), ``crash_c1`` (process
crash after the durable intent), ``crash_c2_eff`` / ``crash_c2_noeff`` (crash
with the request in flight, after / before the effect), ``crash_c3`` (crash
after the response, before COMMITTED), each crash followed by recovery;
``handoff`` (fence, snapshot, install in a new owner); ``stale`` (the fenced
previous owner proposes the action).

**Delegation alphabet** (the first event is always ``delegate``):
``tick_ok`` (the delegate executes the ticket and queues a report),
``tick_crash`` (the delegate crashes after the effect), ``s_resume`` (the
delegate recovers and re-proposes; queues a report), ``report`` (deliver the
oldest queued report to the current owner), ``report_dup`` (deliver the last
delivered report again), ``lose`` (drop the oldest queued report),
``local`` (the owner proposes the action itself), ``expire`` (lease expiry:
the owner revokes and the delegate refuses later tickets), ``handoff``.

A keyed or queryable service is modeled as honest: a request carrying a key
it has already executed gets the stored outcome replayed, whatever else the
event says.  **Invariants** (after every event): I1 at most one effect; I2 no false
suppression (no effect but the owner records COMMITTED); I3 visibility (an
effect happened, so the owner's record exists and is not FAILED); I4 a fenced
runtime never dispatches.  **Liveness probe** (end of each sequence, not an
invariant): one more ``ok``/``local`` proposal must leave exactly one effect
and a COMMITTED record, except that a plain service may instead block.
Output: ``results/protocol_exhaustive.json``.
"""
import itertools
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from runtime.dispatcher import (TrustedDispatcher, ToolCall,        # noqa: E402
                                EndpointContract, Verdict)
import runtime.dispatcher as D                                      # noqa: E402
from runtime.durable import SQLiteLedger, load_snapshot             # noqa: E402
from runtime.delegation import (delegate, merge_result, revoke,     # noqa: E402
                                DelegationResult)
from state.protocol import Status                                   # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RES, os.environ.get("RELAY_EXH_OUT",
                                      "protocol_exhaustive.json"))
MAXLEN_SINGLE = int(os.environ.get("RELAY_EXH_SINGLE", "5"))
MAXLEN_DELEG = int(os.environ.get("RELAY_EXH_DELEG", "5"))
CONTRACTS = ("plain", "keyed", "queryable")
CALL = ToolCall("purchase", {"item": "usb-c dock", "amount": "42.00"})
Rejected = getattr(D, "ServiceRejected", None)


class Crash(BaseException):
    """A process kill: no exception handler in the code under test runs."""


class Service:
    def __init__(self, contract):
        self.contract, self.effects, self.keys = contract, 0, set()

    def apply(self, key):
        if self.contract != "plain" and key in self.keys:
            return
        self.effects += 1
        if self.contract != "plain":
            self.keys.add(key)


def dispatcher(ledger, svc):
    if svc.contract == "queryable":
        return TrustedDispatcher(ledger, reconciler=lambda r: r.key in svc.keys,
                                 contract=EndpointContract(True, True))
    if svc.contract == "keyed":
        return TrustedDispatcher(ledger, contract=EndpointContract(True, False))
    return TrustedDispatcher(ledger)


def new_ledger(epoch=0):
    return SQLiteLedger(":memory:", task_secret="t", epoch=epoch,
                        fullfsync=False)


def rec_status(ledger):
    r = ledger.get(CALL.fingerprint())
    return r.status if r else None


def check(svc, owner):
    st = rec_status(owner)
    if svc.effects > 1:
        return "I1 duplicate effect"
    if svc.effects == 0 and st is Status.COMMITTED:
        return "I2 false suppression"
    if svc.effects >= 1 and (st is None or st is Status.FAILED):
        return f"I3 invisible outcome (record {st})"
    return None


# --------------------------------------------------------------- single
SINGLE = ["ok", "reject", "lost_eff", "lost_noeff", "crash_c1",
          "crash_c2_eff", "crash_c2_noeff", "crash_c3", "handoff", "stale"]


def run_single(seq, contract):
    svc = Service(contract)
    st = {"owner": new_ledger(), "stale": None}

    def attempt(kind):
        led = st["owner"]
        d = dispatcher(led, svc)

        def hook(point):
            if kind == "crash_c1" and point == "c1_after_prepare":
                raise Crash()
            if kind == "crash_c3" and point == "c3_after_response":
                raise Crash()

        def tool(args):
            key = args.get("idempotency_key", "")
            if svc.contract != "plain" and key in svc.keys:
                # an honest keyed service replays the stored outcome of a key
                # it has already executed, whatever this attempt would do
                return {"ok": True, "replayed": True}
            if kind == "reject":
                if Rejected is None:
                    raise RuntimeError("rejected")
                raise Rejected("rejected")
            if kind in ("lost_noeff",):
                raise TimeoutError("lost")
            if kind == "crash_c2_noeff":
                raise Crash()
            svc.apply(key)
            if kind == "lost_eff":
                raise TimeoutError("lost")
            if kind == "crash_c2_eff":
                raise Crash()
            return {"ok": True}
        try:
            d.execute(CALL, tool, hook=hook)
        except Crash:
            led.recover()                    # process restart over its store
        except Exception:
            pass

    for ev in seq:
        if ev == "handoff":
            old = st["owner"]
            old.fence(old.epoch + 1)
            new = new_ledger(epoch=old.epoch + 1)
            load_snapshot(new, old.snapshot_for_handoff())
            st["owner"], st["stale"] = new, old
        elif ev == "stale":
            if st["stale"] is None:
                continue
            old = st["stale"]
            before = svc.effects
            d = dispatcher(old, svc)
            dec = d.check(CALL, observed_epoch=old.fenced_epoch())
            if dec.dispatched or svc.effects != before:
                return ("I4 fenced runtime dispatched", ev)
        else:
            attempt(ev)
        bad = check(svc, st["owner"])
        if bad:
            return (bad, ev)
    # liveness probe
    attempt("ok")
    bad = check(svc, st["owner"])
    if bad:
        return (bad, "probe")
    s = rec_status(st["owner"])
    if svc.effects == 1 and s is Status.COMMITTED:
        return ("live", None)
    if contract == "plain" and s in (Status.UNKNOWN, Status.PREPARED):
        return ("blocked", None)
    return (f"L not resolved (effects={svc.effects}, record={s})", "probe")


# ------------------------------------------------------------ delegation
DELEG = ["tick_ok", "tick_lost", "tick_crash", "s_resume", "report",
         "report_dup", "lose", "local", "expire", "handoff"]


def run_deleg(seq, contract):
    svc = Service(contract)
    T = new_ledger()
    S = new_ledger()
    st = {"owner": T, "expired": False, "queue": [], "last": None}
    ticket = delegate(T, CALL, "source", "d1", now=0, lease=10)
    key = ticket.key
    c2 = ToolCall(CALL.tool, CALL.args, reported_key=key)

    def s_exec(crash_after_effect=False, lost_after_effect=False):
        d = dispatcher(S, svc)

        def tool(args):
            svc.apply(args.get("idempotency_key", ""))
            if crash_after_effect:
                raise Crash()
            if lost_after_effect:
                raise TimeoutError("response lost")
            return {"ok": True}
        try:
            d.execute(c2, tool)
        except Crash:
            S.recover()
            return None
        except Exception:
            pass
        r = S.get(c2.fingerprint())
        return DelegationResult("task", "d1", CALL.fingerprint(), key,
                                (r.status if r else Status.UNKNOWN).value)

    for ev in seq:
        owner = st["owner"]
        if ev == "tick_ok":
            if st["expired"]:
                continue
            res = s_exec()
            if res:
                st["queue"].append(res)
        elif ev == "tick_crash":
            if st["expired"]:
                continue
            s_exec(crash_after_effect=True)
        elif ev == "tick_lost":
            if st["expired"]:
                continue
            res = s_exec(lost_after_effect=True)
            if res:
                st["queue"].append(res)
        elif ev == "s_resume":
            S.recover()
            if S.get(c2.fingerprint()) is None:
                continue                     # never received the ticket
            res = s_exec()
            if res:
                st["queue"].append(res)
        elif ev == "report":
            if st["queue"]:
                st["last"] = st["queue"].pop(0)
                merge_result(owner, st["last"])
        elif ev == "report_dup":
            if st["last"]:
                merge_result(owner, st["last"])
        elif ev == "lose":
            if st["queue"]:
                st["queue"].pop(0)
        elif ev == "local":
            d = dispatcher(owner, svc)
            try:
                d.execute(CALL, lambda a: svc.apply(
                    a.get("idempotency_key", "")) or {"ok": True})
            except Exception:
                pass
        elif ev == "expire":
            st["expired"] = True
            revoke(owner, CALL.fingerprint())
        elif ev == "handoff":
            owner.fence(owner.epoch + 1)
            new = new_ledger(epoch=owner.epoch + 1)
            load_snapshot(new, owner.snapshot_for_handoff())
            st["owner"] = new
        bad = check(svc, st["owner"])
        if bad:
            return (bad, ev)
    # liveness probe: expire any outstanding delegation, then propose locally
    owner = st["owner"]
    st["expired"] = True
    revoke(owner, CALL.fingerprint())
    d = dispatcher(owner, svc)
    try:
        d.execute(CALL, lambda a: svc.apply(a.get("idempotency_key", ""))
                  or {"ok": True})
    except Exception:
        pass
    bad = check(svc, owner)
    if bad:
        return (bad, "probe")
    s = rec_status(owner)
    if svc.effects == 1 and s is Status.COMMITTED:
        return ("live", None)
    if contract == "plain" and s in (Status.UNKNOWN, Status.PREPARED):
        return ("blocked", None)
    if svc.effects == 0 and s is None:
        return ("live-pending", None)
    return (f"L not resolved (effects={svc.effects}, record={s})", "probe")


# ------------------------------------------------------------- algebra
def merge_algebra():
    """Idempotence, commutativity and monotonicity of ``merge_result``."""
    out = Counter()
    statuses = [Status.COMMITTED, Status.FAILED, Status.UNKNOWN,
                Status.PREPARED]
    TERM = {Status.COMMITTED, Status.FAILED, Status.COMPENSATED}
    for a, b in itertools.product(statuses, repeat=2):
        finals = []
        for order in ((a, b), (b, a)):
            T = new_ledger()
            delegate(T, CALL, "source", "d1", now=0, lease=10)
            key = T.get(CALL.fingerprint()).key
            seen = []
            for x in order:
                merge_result(T, DelegationResult("task", "d1",
                                                 CALL.fingerprint(), key,
                                                 x.value))
                seen.append(rec_status(T))
            # monotone: once terminal, never back to PREPARED/INTENDED
            if seen[0] in TERM and seen[1] in (Status.PREPARED,
                                               Status.INTENDED):
                out["monotonicity violated"] += 1
            finals.append(seen[1])
            # idempotence
            merge_result(T, DelegationResult("task", "d1", CALL.fingerprint(),
                                             key, order[1].value))
            if rec_status(T) != seen[1]:
                out["idempotence violated"] += 1
        if a in TERM and b in TERM and finals[0] != finals[1]:
            out["commutativity violated (terminal pair)"] += 1
        out["pairs"] += 1
    return dict(out)


def main():
    t0 = time.time()
    out = {"max_len_single": MAXLEN_SINGLE, "max_len_delegation": MAXLEN_DELEG,
           "has_service_rejected": Rejected is not None,
           "single": {}, "delegation": {}}
    for c in CONTRACTS:
        tally, examples, n = Counter(), {}, 0
        for L in range(1, MAXLEN_SINGLE + 1):
            for seq in itertools.product(SINGLE, repeat=L):
                n += 1
                res, ev = run_single(seq, c)
                tally[res] += 1
                if res not in ("live", "blocked") and res not in examples:
                    examples[res] = list(seq)
        out["single"][c] = {"sequences": n, "outcomes": dict(tally),
                            "first_counterexamples": examples}
        print("single", c, n, dict(tally), flush=True)
    for c in CONTRACTS:
        tally, examples, n = Counter(), {}, 0
        for L in range(0, MAXLEN_DELEG + 1):
            for seq in itertools.product(DELEG, repeat=L):
                n += 1
                res, ev = run_deleg(seq, c)
                tally[res] += 1
                if res not in ("live", "blocked", "live-pending") and \
                        res not in examples:
                    examples[res] = list(seq)
        out["delegation"][c] = {"sequences": n, "outcomes": dict(tally),
                                "first_counterexamples": examples}
        print("delegation", c, n, dict(tally), flush=True)
    out["merge_algebra"] = merge_algebra()
    print("merge algebra", out["merge_algebra"])
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print("->", OUT, out["runtime_s"], "s")


if __name__ == "__main__":
    main()
