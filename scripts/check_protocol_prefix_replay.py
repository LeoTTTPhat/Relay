"""Replay the bounded exhaustive check against the protocol code as it was
before the two fixes it motivated (AUDIT H.4), with the final harness.

The pre-fix ``TrustedDispatcher.execute`` recorded FAILED for *any* tool
exception, and the pre-fix ``merge_result`` let a later report overwrite a
conflict.  Both are reproduced verbatim below and monkeypatched in, so the
counterexamples are attributable to the code, not to the harness.
Output: ``results/protocol_exhaustive_prefix.json`` (length 4).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("RELAY_EXH_SINGLE", "4")
os.environ.setdefault("RELAY_EXH_DELEG", "4")
os.environ["RELAY_EXH_OUT"] = "protocol_exhaustive_prefix.json"
import check_protocol_exhaustive as C                               # noqa: E402
from runtime.dispatcher import TrustedDispatcher                    # noqa: E402
from runtime.delegation import _persist                             # noqa: E402
from state.protocol import ActionRecord, Status                     # noqa: E402


def old_execute(self, call, tool_impl, observed_epoch=None, hook=None):
    h = hook or (lambda _p: None)
    d = self.check(call, observed_epoch)
    if not d.dispatched:
        return d, None
    h("c0_before_prepare")
    rec = self.ledger.prepare(d.fp, call.signature(),
                              reported_key=call.reported_key,
                              family=call.family(),
                              schema_version=call.schema_version or "")
    h("c1_after_prepare")
    try:
        result = tool_impl({**call.args, "idempotency_key": rec.key})
    except Exception as exc:                      # observed failure
        self.ledger.fail(d.fp, f"exception: {type(exc).__name__}")
        raise
    h("c3_after_response")
    self.ledger.commit(d.fp, str(result)[:120])
    h("c4_after_record")
    return d, result


def old_merge_result(ledger, res):
    rec = ledger.get(res.fp)
    new = Status(res.status)
    if rec is None:
        rec = ActionRecord(fp=res.fp, key=res.key, status=new,
                           epoch=ledger.epoch, evidence="delegated")
        _persist(ledger, rec, "merge-install")
        return "installed"
    if rec.key != res.key:
        return "rejected-key-mismatch"
    if rec.status in (Status.INTENDED, Status.PREPARED, Status.UNKNOWN):
        if rec.owner and rec.deleg_id and rec.deleg_id != res.deleg_id:
            return "rejected-ticket-mismatch"
        rec.status = new
        if new is not Status.PREPARED:
            rec.owner, rec.deleg_id = "", ""
        rec.evidence = (res.evidence or "delegated")[:120]
        _persist(ledger, rec, "merge")
        return "merged"
    if rec.status is new:
        return "duplicate-result"
    if {rec.status, new} == {Status.COMMITTED, Status.FAILED}:
        rec.status = Status.UNKNOWN
        rec.owner, rec.deleg_id = "", ""
        rec.evidence = "conflicting delegate result"
        _persist(ledger, rec, "merge-conflict")
        return "conflict"
    return "ignored"


TrustedDispatcher.execute = old_execute
C.merge_result = old_merge_result
C.Rejected = None            # pre-fix code had no ServiceRejected distinction

if __name__ == "__main__":
    C.main()
