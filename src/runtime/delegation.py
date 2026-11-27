"""Delegation: executing a step elsewhere without a second source of truth.

Capability repair can DELEGATE a pending step back to the source (or a cloud
continuation) while the target keeps working.  That creates a consistency
surface the single-dispatcher design did not have: two runtimes, two ledgers,
one action.  The rules below are what E22 tests.

  R1 **write-ahead delegation.**  Before a ticket leaves the target, the
     target durably records the action PREPARED with ``owner=delegate:<dev>``
     and the ticket id.  The ticket carries the action's deterministic key.
  R2 **one executor at a time.**  While that record is outstanding, the
     target's dispatcher answers RECONCILE ("awaiting delegate") to a local
     proposal of the same action; it never races the delegate.
  R3 **the delegate executes under the ticket's key**, through its own
     dispatcher and durable ledger, and refuses a ticket whose lease has
     expired (this assumes bounded clock skew between the two runtimes).
  R4 **results are facts, merged monotonically.**  A result is accepted by
     whichever runtime currently owns the task, whatever its epoch (epochs
     fence *commands*, not reports of what happened), provided its ticket id
     and key match.  Merge is by rank (INTENDED < PREPARED < UNKNOWN <
     COMMITTED/FAILED < COMPENSATED): a report never lowers a record; an
     equal status is an idempotent duplicate; COMMITTED against FAILED is a
     conflict and becomes UNKNOWN, which stays UNKNOWN whatever report
     arrives next (surfaced for reconciliation, never last-writer-wins).
  R5 **lease expiry revokes, it does not assume.**  A ticket with no result
     by its deadline turns the record UNKNOWN, and the service contract
     decides: query it, retry under the same key, or fail closed.
  R6 **handoff carries outstanding delegations** as delegated records (owner
     and ticket kept), so the next owner can still accept the result.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

from state.protocol import ActionRecord, ProtocolLedger, Status


@dataclass
class DelegationTicket:
    task_id: str
    deleg_id: str
    fp: str
    family: str
    key: str
    epoch: int
    tool: str
    args: Dict = field(default_factory=dict)
    lease_until: float = 0.0
    delegate: str = "source"


@dataclass
class DelegationResult:
    task_id: str
    deleg_id: str
    fp: str
    key: str
    status: str
    evidence: str = ""


def _persist(ledger: ProtocolLedger, rec: ActionRecord, op: str):
    put = getattr(ledger, "put_record", None)
    if put is not None:
        put(rec, op)
    else:
        ledger.records[rec.fp] = rec


def delegate(ledger: ProtocolLedger, call, delegate_name: str, deleg_id: str,
             now: float, lease: float, task_id: str = "task"
             ) -> DelegationTicket:
    """R1: durable delegated intent, then the ticket."""
    rec = ledger.prepare(call.fingerprint(), call.signature(),
                         family=call.family(),
                         schema_version=call.schema_version or "",
                         owner=f"delegate:{delegate_name}", deleg_id=deleg_id)
    return DelegationTicket(task_id, deleg_id, rec.fp, rec.family, rec.key,
                            ledger.epoch, call.tool, dict(call.args),
                            now + lease, delegate_name)


#: Information order for merging reports: a report can only raise a record.
RANK = {Status.INTENDED: 0, Status.PREPARED: 1, Status.UNKNOWN: 2,
        Status.COMMITTED: 3, Status.FAILED: 3, Status.COMPENSATED: 4}
CONFLICT = "conflicting delegate reports"


def merge_result(ledger: ProtocolLedger, res: DelegationResult) -> str:
    """R4: monotone merge of a delegate's report into the owner's ledger.

    A report never lowers a record (UNKNOWN is not turned back into PREPARED,
    a terminal status is not turned back into anything weaker).  Equal
    reports are idempotent.  COMMITTED against FAILED is a conflict: the record
    becomes UNKNOWN and stays so for any later report -- only reconciliation
    with the service (``TrustedDispatcher``) can resolve it, so a duplicate of
    either report cannot silently pick a winner.
    """
    rec = ledger.get(res.fp)
    new = Status(res.status)
    if rec is None:
        rec = ActionRecord(fp=res.fp, key=res.key, status=new,
                           epoch=ledger.epoch, evidence="delegated")
        _persist(ledger, rec, "merge-install")
        return "installed"
    if rec.key != res.key:
        return "rejected-key-mismatch"
    if rec.owner and rec.deleg_id and rec.deleg_id != res.deleg_id:
        return "rejected-ticket-mismatch"
    if rec.status is Status.UNKNOWN and rec.evidence == CONFLICT:
        return "ignored-conflict"
    if new is rec.status:
        return "duplicate-result"
    if {rec.status, new} == {Status.COMMITTED, Status.FAILED}:
        rec.status = Status.UNKNOWN
        rec.owner, rec.deleg_id = "", ""
        rec.evidence = CONFLICT
        _persist(ledger, rec, "merge-conflict")
        return "conflict"
    if RANK[new] < RANK[rec.status]:
        return "ignored-weaker"
    rec.status = new
    if new is not Status.PREPARED:
        rec.owner, rec.deleg_id = "", ""
    rec.evidence = (res.evidence or "delegated")[:120]
    _persist(ledger, rec, "merge")
    return "merged"


def revoke(ledger: ProtocolLedger, fp: str) -> bool:
    """R5: an expired delegation becomes an ordinary undetermined action."""
    rec = ledger.get(fp)
    if rec is None or not rec.owner or rec.status is not Status.PREPARED:
        return False
    rec.owner, rec.deleg_id = "", ""
    rec.status = Status.UNKNOWN
    _persist(ledger, rec, "revoke")
    return True
