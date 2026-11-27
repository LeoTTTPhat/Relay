"""E22: delegation state merge under loss, crashes, races and a second handoff.

A DELEGATE verdict runs a step on the source (or a cloud continuation) while
the target continues.  Reviewers asked how the two ledgers merge.  The merge
rules are specified in ``runtime/delegation.py`` (R1-R6); this experiment
tests them with a randomized discrete-event simulation in which every actor
uses the implemented code -- ``SQLiteLedger`` stores (in-memory databases),
``TrustedDispatcher``, ``delegate``/``merge_result``/``revoke`` -- and only the
network, the clock and the failures are simulated.

Per trial, one action is delegated by the target T to the source S.  Faults
are drawn independently: ticket lost; ticket delivered late (after its lease);
S crashes before the effect, or after the effect but before recording/replying
(and later restarts and resumes); result lost or duplicated; T's model
proposes the same action locally at a random time; a second handoff moves the
task from T to T2 (epoch +1) while the delegation is outstanding, with the
result routed to either the new or the stale owner.  At the end the current
owner proposes the action once more (the resumption conditional).

Three designs:
  ``relay``             rules R1-R6;
  ``naive_optimistic``  T records the action COMMITTED when the ticket is sent;
  ``naive_unrecorded``  T records nothing for a delegated step (deterministic
                        keys are still used, so keyed services can dedup).
Three service contracts as in E20: ``plain``, ``keyed``, ``queryable``.

Outcomes, from the service's effect count and the owner's final record:
  ``correct``   one effect and the owner's record says COMMITTED;
  ``duplicate`` two or more effects;
  ``lost``      no effect but the record says COMMITTED (false suppression);
  ``blocked``   the owner's record is unresolved (surfaced to the user);
  ``pending``   no effect and nothing recorded as done (the action simply has
                not happened yet and nothing claims it has).

This is a simulation of message delivery and timing, not a distributed
deployment; process-crash durability is E20's job.
"""
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from runtime.dispatcher import (TrustedDispatcher, ToolCall,        # noqa: E402
                                EndpointContract, Verdict)
from runtime.durable import SQLiteLedger, load_snapshot             # noqa: E402
from runtime.delegation import (delegate, merge_result, revoke,     # noqa: E402
                                DelegationResult)
from state.protocol import Status, derive_key                       # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RES, "e22_delegation.json")
N = int(os.environ.get("RELAY_E22_N", "2000"))
LEASE = 10
P = {"ticket_lost": 0.1, "ticket_late": 0.1, "s_crash_before": 0.1,
     "s_crash_after": 0.1, "result_lost": 0.1, "result_dup": 0.1,
     "local_propose": 0.6, "second_handoff": 0.3, "route_stale": 0.5}


class World:
    def __init__(self, contract):
        self.contract = contract
        self.effects = 0
        self.done_keys = set()

    def act(self, key):
        if self.contract != "plain" and key in self.done_keys:
            return {"ok": True, "replayed": True}
        self.effects += 1
        if self.contract != "plain":
            self.done_keys.add(key)
        return {"ok": True}

    def status(self, key):
        return key in self.done_keys


def _disp(ledger, world):
    c = world.contract
    if c == "queryable":
        return TrustedDispatcher(ledger, reconciler=lambda r: world.status(r.key),
                                 contract=EndpointContract(True, True))
    if c == "keyed":
        return TrustedDispatcher(ledger, contract=EndpointContract(True, False))
    return TrustedDispatcher(ledger)


def _tool(world):
    return lambda args: world.act(args.get("idempotency_key", ""))


def trial(design, contract, rng):
    world = World(contract)
    call = ToolCall("purchase", {"item": "usb-c dock", "amount": "42.00"})
    T = SQLiteLedger(":memory:", task_secret="task", fullfsync=False)
    S = SQLiteLedger(":memory:", task_secret="task", fullfsync=False)
    f = {k: rng.random() < p for k, p in P.items()}
    key = derive_key("task", call.signature())      # what every design sends
    if design == "relay":
        ticket = delegate(T, call, "source", "d1", now=0, lease=LEASE)
        assert ticket.key == key
    elif design == "naive_optimistic":
        T.prepare(call.fingerprint(), call.signature())
        T.commit(call.fingerprint(), "assumed done at delegation")
    events = []
    if not f["ticket_lost"]:
        arrive = rng.randint(1, 4) + (LEASE + 2 if f["ticket_late"] else 0)
        events.append((arrive, 0, "ticket"))
    if f["local_propose"]:
        events.append((rng.randint(1, 25), 1, "local"))
    if f["second_handoff"]:
        events.append((rng.randint(2, 20), 2, "handoff"))
    events.append((LEASE + 1, 3, "lease_check"))
    st = {"owner": T, "T2": None}

    def deliver(res):
        if design != "relay":
            return        # optimistic already says done; unrecorded has no slot
        if st["T2"] is not None and f["route_stale"]:
            merge_result(T, res)       # accepted by the stale owner only
        else:
            merge_result(st["owner"], res)

    def s_execute():
        c2 = ToolCall(call.tool, call.args, reported_key=key)
        _disp(S, world).execute(c2, _tool(world))
        rec = S.get(c2.fingerprint())
        return rec.status if rec else Status.UNKNOWN

    while events:
        events.sort()
        t, _, kind = events.pop(0)
        if kind == "ticket":
            if design == "relay" and t > LEASE:
                continue                     # R3: expired ticket refused
            if f["s_crash_before"]:
                continue                     # S dies before doing anything
            if f["s_crash_after"]:
                # the effect happens; S dies before recording or replying,
                # then restarts and resumes the ticket it holds
                c2 = ToolCall(call.tool, call.args, reported_key=key)
                S.prepare(c2.fingerprint(), c2.signature(), reported_key=key)
                world.act(key)
                events.append((t + rng.randint(3, 12), 4, "s_resume"))
                continue
            status = s_execute()
            if not f["result_lost"]:
                r = DelegationResult("task", "d1", call.fingerprint(), key,
                                     status.value)
                deliver(r)
                if f["result_dup"]:
                    deliver(r)
        elif kind == "s_resume":
            S.recover()
            status = s_execute()
            if not f["result_lost"]:
                deliver(DelegationResult("task", "d1", call.fingerprint(), key,
                                         status.value))
        elif kind == "local":
            _disp(st["owner"], world).execute(call, _tool(world))
        elif kind == "handoff":
            old = st["owner"]
            old.bump_epoch()                 # D3: fence before the snapshot
            snap = old.snapshot_for_handoff()
            T2 = SQLiteLedger(":memory:", task_secret="task",
                              epoch=old.epoch, fullfsync=False)
            load_snapshot(T2, snap)
            st["T2"], st["owner"] = T2, T2
        elif kind == "lease_check" and design == "relay":
            revoke(st["owner"], call.fingerprint())   # no-op once merged
    # resumption conditional: the current owner proposes the action again
    owner = st["owner"]
    dec, _ = _disp(owner, world).execute(call, _tool(world))
    rec = owner.get(call.fingerprint())
    s_ = rec.status if rec else None
    if world.effects >= 2:
        return "duplicate"
    if world.effects == 1 and s_ is Status.COMMITTED:
        return "correct"
    if world.effects == 0 and s_ is Status.COMMITTED:
        return "lost"
    if s_ in (Status.UNKNOWN, Status.PREPARED) or \
            dec.verdict is Verdict.RECONCILE:
        return "blocked"
    return "pending" if world.effects == 0 else "unrecorded_effect"


def run():
    rng = random.Random(22)
    out = {"n_per_cell": N, "fault_probabilities": P, "lease": LEASE}
    cells = {}
    for design in ("naive_optimistic", "naive_unrecorded", "relay"):
        for contract in ("plain", "keyed", "queryable"):
            tally = defaultdict(int)
            for _ in range(N):
                tally[trial(design, contract, rng)] += 1
            cells[f"{design}|{contract}"] = {
                "counts": dict(tally),
                "rates": {k: round(v / N, 4) for k, v in sorted(tally.items())}}
            print(f"{design:18s} {contract:10s} {cells[f'{design}|{contract}']['rates']}",
                  flush=True)
    out["cells"] = cells
    json.dump(out, open(OUT, "w"), indent=1)
    print("->", OUT)


if __name__ == "__main__":
    run()
