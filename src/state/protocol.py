"""Side-effect commit protocol: action status, write-ahead records, and the
crash/race semantics the ledger actually provides.

The first version of this system recorded ``status="done"`` for every
side-effect-like step it found in the transcript and asked the resumed model
to skip matching actions.  That is *advisory duplicate suppression under a
single-writer assumption*, not exactly-once execution, and three windows
were unmodelled:

  1. the external action succeeds but the source dies before recording it
     (-> the record is lost, the resumed agent repeats the action);
  2. the record says ``done`` but the service actually failed or rolled back
     (-> the resumed agent skips an action that never happened);
  3. two resumptions race and both execute (-> duplicate).

This module makes those windows explicit and closes what can be closed
without a distributed coordinator.

**Status machine.**  An action moves

    INTENDED -> PREPARED -> COMMITTED
                        \-> FAILED
                        \-> UNKNOWN -> {COMMITTED, FAILED, COMPENSATED}

``PREPARED`` is a *write-ahead intent* recorded durably **before** the tool
is dispatched.  It is the entry that closes window 1: a crash between
dispatch and response leaves a ``PREPARED`` record, and recovery reads that
as ``UNKNOWN`` -- an action that must be *reconciled* (queried against the
service) rather than blindly skipped or blindly retried.  Only a response
observed from the service produces ``COMMITTED`` or ``FAILED``.

**Atomicity.**  We do not have two-phase commit with the external service;
almost no real tool endpoint offers one.  What we have is the same
discipline payment APIs use: a durable intent written before the call, a
*deterministic* idempotency key carried on the call, and reconciliation on
recovery.  The pair (write-ahead intent, deterministic key) is what turns an
unknown outcome into a resolvable one rather than a coin flip.

**Deterministic keys and the race.**  ``derive_key`` mints the idempotency
key as ``HMAC(task_secret, action_signature)`` -- a pure function of the
task and the action, not of the device, the clock or the resumption.  Two
resumptions that race on the same pending action therefore present the
*same* key to the service, and the service's own idempotency dedups them.
Split-brain is not eliminated (nothing local can eliminate it); it is
reduced to the service-side guarantee that idempotency keys already carry,
and E13 measures the rate at which racing resumptions agree on the key.
When the transcript reports a key the service itself minted, that key wins
(``adopt_upstream_key``): a key we invented is one the service will not
recognise.

**Epochs.**  Every migration increments a handoff epoch carried in the
block.  A dispatcher holding epoch $e$ refuses to commit if it observes a
record from epoch $> e$: the stale writer (a source whose acknowledgment was
lost, or a user who walked back to the old device) fails closed instead of
executing.  This is a local safety check, not consensus.
"""
import hashlib
import hmac
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional


class Status(str, Enum):
    """Lifecycle of one side-effectful action."""
    INTENDED = "intended"       # planned, nothing written, nothing dispatched
    PREPARED = "prepared"       # write-ahead intent durable; may be in flight
    COMMITTED = "committed"     # service response observed, action happened
    FAILED = "failed"           # service response observed, action did not happen
    UNKNOWN = "unknown"         # crashed in the window; outcome undetermined
    COMPENSATED = "compensated"  # happened, then undone by a compensating action


#: Statuses whose actions are known to have taken effect in the world.
EFFECTED = {Status.COMMITTED}
#: Statuses that require a reconciliation query before any further action.
NEEDS_RECONCILE = {Status.PREPARED, Status.UNKNOWN}
#: Statuses after which a retry is safe (carrying the same key).
RETRYABLE = {Status.FAILED, Status.INTENDED}


@dataclass
class ActionRecord:
    """One ledger entry under the commit protocol.

    ``fp`` is the action identity (see ``fingerprint.py``); ``key`` is the
    idempotency key presented to the service; ``epoch`` is the handoff
    generation that wrote the record; ``attempts`` counts dispatches, which
    is how a reconciliation distinguishes "never sent" from "sent, outcome
    unobserved".
    """
    fp: str
    status: Status = Status.INTENDED
    key: str = ""
    epoch: int = 0
    attempts: int = 0
    head: str = ""
    step: int = -1
    agent: str = ""
    upstream_key: bool = False      # key came from the service, not from us
    evidence: str = ""              # what the service returned, bounded
    # --- added for E20-E22 (all default to the pre-revision behaviour) ---
    family: str = ""                # version-free identity (fp_typed_family)
    schema_version: str = ""        # tool API/schema version identity used
    owner: str = ""                 # "" = this runtime; "delegate:<dev>" while
                                    # a delegated execution is outstanding
    deleg_id: str = ""              # delegation ticket this record waits on

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @staticmethod
    def from_dict(d: Dict) -> "ActionRecord":
        d = dict(d)
        d["status"] = Status(d.get("status", "intended"))
        return ActionRecord(**d)


# ---------------------------------------------------------------------------
# Deterministic idempotency keys
# ---------------------------------------------------------------------------
def derive_key(task_secret: str, action_signature: str, nbits: int = 20) -> str:
    """Idempotency key as a pure function of (task, action).

    Determinism is the point: two devices that resume the same trajectory and
    reach the same pending action mint the *same* key without talking to each
    other, so a service that honours idempotency keys collapses the racing
    retries into one execution.  Keys derived from a device id, a timestamp
    or a random nonce -- the obvious implementations -- do the opposite.
    """
    mac = hmac.new(task_secret.encode(), action_signature.encode(),
                   hashlib.sha256).hexdigest()
    return "relay-" + mac[:nbits]


def adopt_upstream_key(reported: Optional[str], derived: str) -> "tuple":
    """Prefer a key the service minted over one we derived.

    Returns ``(key, is_upstream)``.  A retry presented with a key we invented
    is, to the service, a *new* request; only the key it issued makes the
    retry idempotent.
    """
    if reported:
        return reported, True
    return derived, False


# ---------------------------------------------------------------------------
# The ledger under the protocol
# ---------------------------------------------------------------------------
class ProtocolLedger:
    """Durable-by-contract set of ``ActionRecord``s, keyed by fingerprint.

    The "durability" here is the *interface*: ``prepare`` must return only
    after the intent is stable (an fsync'd append, a KV put with
    write-concern), because every guarantee below assumes a crash after
    ``prepare`` leaves the record readable.  The crash-injection study (E13)
    is written against exactly the windows this interface defines.
    """

    def __init__(self, task_secret: str = "task", epoch: int = 0):
        self.task_secret = task_secret
        self.epoch = epoch
        self.records: Dict[str, ActionRecord] = {}
        self.wal: List[tuple] = []       # (op, fp, status) -- replayable

    # -- write path -------------------------------------------------------
    def prepare(self, fp: str, signature: str, head: str = "",
                step: int = -1, agent: str = "",
                reported_key: Optional[str] = None, family: str = "",
                schema_version: str = "", owner: str = "",
                deleg_id: str = "") -> ActionRecord:
        """Durably record the intent to perform ``fp``, *before* dispatch."""
        derived = derive_key(self.task_secret, signature)
        key, upstream = adopt_upstream_key(reported_key, derived)
        r = self.records.get(fp)
        if r is None:
            r = ActionRecord(fp=fp, key=key, epoch=self.epoch, head=head,
                             step=step, agent=agent, upstream_key=upstream,
                             family=family, schema_version=schema_version)
            self.records[fp] = r
        r.status = Status.PREPARED
        r.attempts += 1
        r.owner, r.deleg_id = owner, deleg_id
        self.wal.append(("prepare", fp, Status.PREPARED.value))
        return r

    def find_family(self, family: str) -> List[ActionRecord]:
        """Records sharing a version-free identity (see ``fp_typed_family``)."""
        if not family:
            return []
        return [r for r in self.records.values() if r.family == family]

    def commit(self, fp: str, evidence: str = "") -> None:
        """Record an observed success."""
        r = self.records.get(fp)
        if r is None:
            return
        r.status = Status.COMMITTED
        r.evidence = evidence[:120]
        self.wal.append(("commit", fp, Status.COMMITTED.value))

    def fail(self, fp: str, evidence: str = "") -> None:
        """Record an observed failure: the action did *not* take effect."""
        r = self.records.get(fp)
        if r is None:
            return
        r.status = Status.FAILED
        r.evidence = evidence[:120]
        self.wal.append(("fail", fp, Status.FAILED.value))

    def mark_unknown(self, fp: str, evidence: str = "") -> None:
        """The request was sent but its outcome was not observed."""
        r = self.records.get(fp)
        if r is None or r.status is not Status.PREPARED:
            return
        r.status = Status.UNKNOWN
        r.evidence = evidence[:120]
        self.wal.append(("unknown", fp, Status.UNKNOWN.value))

    def compensate(self, fp: str, evidence: str = "") -> None:
        r = self.records.get(fp)
        if r is None:
            return
        r.status = Status.COMPENSATED
        r.evidence = evidence[:120]
        self.wal.append(("compensate", fp, Status.COMPENSATED.value))

    # -- recovery ---------------------------------------------------------
    def recover(self) -> List[ActionRecord]:
        """Promote every in-flight intent to UNKNOWN after a crash.

        This is the step that makes window 1 survivable: an action dispatched
        but never resolved is *not* silently absent from the ledger and *not*
        reported as done.  It is reported as undetermined, which is a state
        the resumed runtime can act on.
        """
        out = []
        for r in self.records.values():
            if r.status is Status.PREPARED:
                r.status = Status.UNKNOWN
                out.append(r)
        return out

    def bump_epoch(self) -> int:
        self.epoch += 1
        for r in self.records.values():
            pass                      # records keep the epoch that wrote them
        return self.epoch

    # -- read path --------------------------------------------------------
    def get(self, fp: str) -> Optional[ActionRecord]:
        return self.records.get(fp)

    def entries(self) -> List[ActionRecord]:
        return sorted(self.records.values(), key=lambda r: (r.step, r.fp))

    def stale_writer(self, observed_epoch: int) -> bool:
        """True if a record from a later epoch exists: we are the stale copy."""
        return observed_epoch > self.epoch


#: Crash points, named for the window they fall in.  E13 injects each.
CRASH_POINTS = [
    "c0_before_prepare",     # nothing written, nothing dispatched
    "c1_after_prepare",      # intent durable, tool not yet dispatched
    "c2_in_flight",          # dispatched, response not observed
    "c3_after_response",     # response observed, outcome not yet recorded
    "c4_after_record",       # fully recorded
]

#: What each crash point leaves behind, and what a *correct* recovery owes.
#: ``effected`` = did the world change?  ``ledger`` = what recovery reads.
CRASH_SEMANTICS = {
    "c0_before_prepare":  {"effected": False, "ledger": Status.INTENDED},
    "c1_after_prepare":   {"effected": False, "ledger": Status.UNKNOWN},
    "c2_in_flight":       {"effected": None,  "ledger": Status.UNKNOWN},
    "c3_after_response":  {"effected": None,  "ledger": Status.UNKNOWN},
    "c4_after_record":    {"effected": None,  "ledger": None},   # as recorded
}
