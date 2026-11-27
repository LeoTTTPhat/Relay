"""E21: tool API/schema version as part of action identity.

A reviewer asked whether the same call sent to a semantically changed tool
version could be falsely identified with the old one.  Before this revision it
could: ``fp_typed`` hashed tool name and canonical arguments only, so
``purchase(item=dock, amount=5)`` under a v1 schema (amount in dollars) and
under a v2 schema (amount in cents) had one identity, and the dispatcher would
suppress the second as a duplicate of the first.

The fix has two parts (``state/fingerprint.py``, ``runtime/dispatcher.py``):
the declared schema version enters the typed signature, and a proposed call
whose *versioned* identity is unrecorded while its version-free *family* was
recorded under another version is routed to RECONCILE -- because a version
bump may or may not change semantics, neither suppressing nor executing is a
safe guess.

Tests, over every typed side effect in the 200 synthetic episodes, each first
recorded COMMITTED under version ``2025-01``:

  **same version, semantics-preserving rewrite** (the six E12 typed
  perturbations): the call must still be recognised -> SKIP_DUPLICATE;
  **changed version, same arguments** and **changed version + rewrite**: must
  not be silently skipped (the old identity's false-identity rate) and must not
  be silently executed -> RECONCILE;
  **changed version, a genuinely different call** (one semantic argument
  changed): nothing matches -> EXECUTE, and no spurious RECONCILE.

Rates are reported for the unversioned identity (pre-revision) and the
versioned identity plus family check (shipped).
"""
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, RES                                   # noqa: E402
from run_e12_adversarial import (TYPED_PERTURBATIONS,              # noqa: E402
                                 _parse_args)
from runtime.dispatcher import TrustedDispatcher, ToolCall, Verdict  # noqa: E402
from state.protocol import ProtocolLedger                          # noqa: E402

OUT = os.path.join(RES, "e21_schema_version.json")
V_OLD, V_NEW = "2025-01", "2026-03"


def _record(ledger, call):
    ledger.prepare(call.fingerprint(), call.signature(),
                   family=call.family(),
                   schema_version=call.schema_version or "")
    ledger.commit(call.fingerprint(), "ok")


def run():
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    acts = [(se["action"], _parse_args(se["args"]))
            for e in eps for se in e.states[-1].side_effects]
    rng = random.Random(21)
    tally = {m: defaultdict(lambda: defaultdict(int))
             for m in ("unversioned", "versioned")}
    for tool, args in acts:
        for mode in ("unversioned", "versioned"):
            ver = (lambda v: v) if mode == "versioned" else (lambda v: None)
            base = ToolCall(tool, dict(args), schema_version=ver(V_OLD))
            cases = []
            for name, fn in TYPED_PERTURBATIONS.items():
                t2, a2 = fn(tool, dict(args), rng)
                cases.append(("same_version_rewrite",
                              ToolCall(t2, a2, schema_version=ver(V_OLD))))
            cases.append(("new_version_same_args",
                          ToolCall(tool, dict(args), schema_version=V_NEW
                                   if mode == "versioned" else None)))
            for name, fn in TYPED_PERTURBATIONS.items():
                t2, a2 = fn(tool, dict(args), rng)
                cases.append(("new_version_rewrite",
                              ToolCall(t2, a2, schema_version=V_NEW
                                       if mode == "versioned" else None)))
            if args:
                k = rng.choice(list(args))
                a3 = dict(args)
                a3[k] = str(args[k]) + "_other"
                cases.append(("new_version_different_call",
                              ToolCall(tool, a3, schema_version=V_NEW
                                       if mode == "versioned" else None)))
            for label, call in cases:
                led = ProtocolLedger(task_secret="t")
                _record(led, base)
                v = TrustedDispatcher(led).check(call).verdict
                tally[mode][label][v.value] += 1
    out = {"n_actions": len(acts), "versions": [V_OLD, V_NEW]}
    for mode, d in tally.items():
        out[mode] = {}
        for label, v in d.items():
            n = sum(v.values())
            out[mode][label] = {"n": n, **{k: round(c / n, 4)
                                           for k, c in sorted(v.items())}}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("->", OUT)


if __name__ == "__main__":
    run()
