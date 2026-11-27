"""Trusted tool dispatcher: where duplicate suppression is actually enforced.

The first version of this system shipped the ledger into the prompt under a
``do NOT re-execute`` header and relied on the resumed model to consult it.
That makes the model part of the trusted computing base for a safety
property, which is the wrong place for it: a model can misread the ledger
(our own resumed-model study measures a recall below 1), can be talked out
of it by injected text in a tool result, or can simply be a different model
than the one we tested.

``TrustedDispatcher`` moves the check *below* the model.  The resumed agent
proposes a typed tool call; the dispatcher -- not the model -- canonicalises
it, fingerprints it, consults the ledger, and returns a verdict.  A call the
ledger says already happened never reaches the tool, whatever the model
intended.  The model's comprehension of the ledger still matters, but for
*plan quality* (not wasting a turn proposing a suppressed action, explaining
to the user what was already done), not for safety.

This is interposition, not consensus.  What it guarantees:

  * an action whose record says COMMITTED is not dispatched again;
  * an action whose outcome is undetermined (PREPARED/UNKNOWN after a crash)
    is *not* dispatched blind and *not* silently skipped -- it is routed to
    reconciliation, which is a first-class verdict rather than a fallback;
  * an action from a superseded epoch fails closed;
  * every dispatch carries the deterministic idempotency key, so a duplicate
    that escapes us (a racing resumption, a divergent fingerprint) still
    meets the service's own idempotency check.

What it does not guarantee: that the service honours idempotency keys, that
the fingerprint never diverges, or that a compromised target runs our
dispatcher at all.  Those are stated as assumptions, not covered by a claim.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

from state.fingerprint import fp_typed, fp_typed_family, typed_signature
from state.protocol import (ActionRecord, ProtocolLedger, Status,
                            EFFECTED, NEEDS_RECONCILE, RETRYABLE)


class ServiceRejected(Exception):
    """The service answered and the action did not take effect.

    Only this exception records FAILED.  Any other exception from a tool (a
    timeout, a reset connection, a crash of the client library) means the
    outcome was *not observed*, so the record becomes UNKNOWN: a request that
    may have taken effect must be reconciled, never retried blind.
    """


class Verdict(str, Enum):
    EXECUTE = "execute"                  # no record: dispatch normally
    SKIP_DUPLICATE = "skip_duplicate"    # COMMITTED: the world already has it
    RECONCILE = "reconcile"              # UNKNOWN/PREPARED: query, then decide
    RETRY_WITH_KEY = "retry_with_key"    # FAILED: safe to retry, same key
    BLOCK_STALE_EPOCH = "block_stale"    # we are a superseded writer
    RE_EXECUTE_COMPENSATED = "re_execute_compensated"  # undone; may redo


@dataclass
class ToolCall:
    """A typed call as the runtime sees it *before* dispatch."""
    tool: str
    args: Dict
    side_effectful: bool = True
    reported_key: Optional[str] = None
    #: tool API/schema version from the tool registry; part of identity when
    #: set (E21).  ``None`` keeps the unversioned identity.
    schema_version: Optional[str] = None

    def signature(self) -> str:
        return typed_signature(self.tool, self.args,
                               schema_version=self.schema_version)

    def fingerprint(self) -> str:
        return fp_typed(self.tool, self.args,
                        schema_version=self.schema_version)

    def family(self) -> str:
        return fp_typed_family(self.tool, self.args)


@dataclass
class EndpointContract:
    """What the service behind a tool guarantees, declared in the manifest.

    The guarantee Relay can give after a crash depends on this and on nothing
    else, so it is a typed declaration rather than an assumption buried in
    prose:

    * ``queryable``: the service answers "did the request bearing key k take
      effect?" -> an undetermined action is reconciled by asking;
    * ``idempotent_keys``: the service executes at most once per key and
      returns the stored result on a retry -> re-sending with the *same* key
      is itself the reconciliation;
    * neither: nothing can resolve the outcome, so the dispatcher fails
      closed and surfaces the action to the user.
    """
    idempotent_keys: bool = False
    queryable: bool = False


@dataclass
class Decision:
    verdict: Verdict
    fp: str
    key: str = ""
    record: Optional[ActionRecord] = None
    reason: str = ""

    @property
    def dispatched(self) -> bool:
        """Does the tool actually get called?"""
        return self.verdict in (Verdict.EXECUTE, Verdict.RETRY_WITH_KEY,
                                Verdict.RE_EXECUTE_COMPENSATED)


class TrustedDispatcher:
    """Interposes on every side-effectful call the resumed agent proposes.

    ``reconciler`` is the hook a deployment supplies to resolve an
    undetermined action against the service ("does an order with this
    idempotency key exist?").  It returns ``True`` (the action did happen),
    ``False`` (it did not), or ``None`` (the service cannot say), and the
    dispatcher records the outcome.  When no reconciler is available the
    dispatcher fails *closed*: an undetermined action is not dispatched, and
    the ambiguity is surfaced to the user rather than resolved by guessing.
    """

    def __init__(self, ledger: ProtocolLedger,
                 reconciler: Optional[Callable[[ActionRecord], Optional[bool]]] = None,
                 fail_closed: bool = True,
                 contract: Optional[EndpointContract] = None):
        self.ledger = ledger
        self.reconciler = reconciler
        self.fail_closed = fail_closed
        self.contract = contract
        self.log: List[Decision] = []

    # ------------------------------------------------------------------
    def check(self, call: ToolCall, observed_epoch: Optional[int] = None) -> Decision:
        """Decide what happens to ``call``.  The model does not get a vote."""
        if not call.side_effectful:
            d = Decision(Verdict.EXECUTE, "", reason="read-only")
            self.log.append(d)
            return d

        if observed_epoch is not None and self.ledger.stale_writer(observed_epoch):
            d = Decision(Verdict.BLOCK_STALE_EPOCH, call.fingerprint(),
                         reason=f"epoch {self.ledger.epoch} superseded by "
                                f"{observed_epoch}")
            self.log.append(d)
            return d

        fp = call.fingerprint()
        rec = self.ledger.get(fp)

        if rec is None and call.schema_version:
            # E21: the versioned identity is unrecorded, but the same call was
            # recorded under another schema version.  Suppressing it would
            # treat a possibly changed action as done; executing it would
            # repeat a possibly unchanged one.  Neither guess is safe.
            other = [r for r in self.ledger.find_family(call.family())
                     if r.schema_version != (call.schema_version or "")
                     and (r.status in EFFECTED or r.status in NEEDS_RECONCILE)]
            if other:
                d = Decision(Verdict.RECONCILE, fp, other[0].key, other[0],
                             f"schema version changed "
                             f"({other[0].schema_version} -> "
                             f"{call.schema_version})")
                self.log.append(d)
                return d

        if rec is None:
            d = Decision(Verdict.EXECUTE, fp, reason="no record")
        elif rec.status in EFFECTED:
            d = Decision(Verdict.SKIP_DUPLICATE, fp, rec.key, rec,
                         "committed record")
        elif rec.status is Status.COMPENSATED:
            d = Decision(Verdict.RE_EXECUTE_COMPENSATED, fp, rec.key, rec,
                         "previously undone")
        elif rec.status in NEEDS_RECONCILE:
            d = self._reconcile(rec, fp)
        elif rec.status in RETRYABLE:
            d = Decision(Verdict.RETRY_WITH_KEY, fp, rec.key, rec,
                         "recorded failure")
        else:
            d = Decision(Verdict.RECONCILE, fp, rec.key, rec, "unhandled status")

        self.log.append(d)
        return d

    def _reconcile(self, rec: ActionRecord, fp: str) -> Decision:
        """Resolve an undetermined action before letting anything through."""
        if rec.owner and rec.status is Status.PREPARED:
            # E22: a delegated execution is outstanding.  The local runtime
            # must not race the delegate; it waits for the delegate's result
            # (merged by ``delegation.merge_result``) or for the lease to be
            # revoked, which turns the record into an ordinary UNKNOWN.
            return Decision(Verdict.RECONCILE, fp, rec.key, rec,
                            f"delegated to {rec.owner}; awaiting result")
        if self.reconciler is None and self.contract is not None \
                and self.contract.idempotent_keys:
            # The service collapses retries that carry the same key, so a
            # same-key retry is safe whatever happened before the crash.
            return Decision(Verdict.RETRY_WITH_KEY, fp, rec.key, rec,
                            "idempotent endpoint: same-key retry reconciles")
        if self.reconciler is None:
            return Decision(Verdict.RECONCILE, fp, rec.key, rec,
                            "undetermined, no reconciler: fail closed"
                            if self.fail_closed else "undetermined")
        outcome = self.reconciler(rec)
        if outcome is True:
            self.ledger.commit(fp, "reconciled: service reports done")
            return Decision(Verdict.SKIP_DUPLICATE, fp, rec.key, rec,
                            "reconciled to committed")
        if outcome is False:
            self.ledger.fail(fp, "reconciled: service reports not done")
            return Decision(Verdict.RETRY_WITH_KEY, fp, rec.key, rec,
                            "reconciled to failed")
        return Decision(Verdict.RECONCILE, fp, rec.key, rec,
                        "service could not determine outcome")

    # ------------------------------------------------------------------
    def execute(self, call: ToolCall, tool_impl: Callable[[Dict], object],
                observed_epoch: Optional[int] = None,
                hook: Optional[Callable[[str], None]] = None):
        """Full guarded path: check, write-ahead, dispatch, record.

        The ordering is the protocol: the intent is durable *before* the tool
        runs, so a crash in flight is recoverable as UNKNOWN rather than
        invisible.  Returns ``(Decision, result)``; ``result`` is ``None``
        whenever the call was not dispatched.

        ``hook`` (E20 fault injection only) is called with the name of each
        crash point as the path crosses it; ``tool_impl`` crosses
        ``c2_in_flight`` itself, after the request has left the process.
        """
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
        except ServiceRejected as exc:                # observed failure
            self.ledger.fail(d.fp, f"rejected: {exc}"[:120])
            raise
        except Exception as exc:                      # outcome not observed
            self.ledger.mark_unknown(d.fp,
                                     f"indeterminate: {type(exc).__name__}")
            raise
        h("c3_after_response")
        self.ledger.commit(d.fp, str(result)[:120])
        h("c4_after_record")
        return d, result
