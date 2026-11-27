"""E13: what the ledger actually guarantees -- enforcement, crashes, races.

E4 measured *record loss*: whether the record of a completed action survives
the target's budget.  That is a necessary condition for safety and nowhere
near a sufficient one.  Three things stand between a surviving record and an
action that does not happen twice, and this experiment measures each.

  **E13a enforcement.**  A record only prevents a duplicate if something acts
  on it.  In the first version of this system that something was the resumed
  *model*, prompted with a "do NOT re-execute" header -- and our own E10
  measured its compliance at 0.925 for \\sys{} and 0.775 for a recency window,
  not 1.0.  We compare three regimes on the same actions: no ledger, the
  advisory ledger (model decides), and the trusted dispatcher (the runtime
  decides, below the model).  The duplicate rate composes three *measured*
  quantities -- record survival (E4), model compliance (E10) and fingerprint
  divergence (E12) -- rather than assuming any of them.

  **E13b crash consistency.**  Five crash points are injected around each
  action.  ``status_only`` is the original design: a ledger entry appears only
  once the transcript reports the action done, so a crash between the external
  effect and the record leaves no trace of it.  ``wal`` is the commit protocol:
  a durable intent *before* dispatch, an observed outcome after, and recovery
  that promotes an unresolved intent to UNKNOWN instead of to absent.  We count
  duplicates, lost executions and -- the price of the protocol -- actions
  blocked pending reconciliation.

  **E13c split brain.**  Two resumptions racing on the same pending action.
  Nothing local can prevent both from acting, so the question is whether they
  present the service the *same* idempotency key, which is what lets the
  service collapse them.  Deterministic keys are compared against the two
  obvious implementations (device-scoped and random), and against a source
  whose acknowledgment was lost and which resumes with a stale epoch.

Every rate here is conditional on the resumed agent *proposing* the action
again; we do not model how often it does, because that is a property of the
agent, not of Relay.  The conditional rate is the quantity the mechanism
controls, and stating it that way is more honest than multiplying by a guess.

Two things this experiment does **not** show, stated here so its numbers are
not overread:

  * E13b's ``wal/reconciler`` column is correct *by construction*: it assumes
    a service that can answer "did the request with this idempotency key take
    effect?".  The contribution being measured is not that reconciliation
    succeeds -- it is that the protocol leaves a *resolvable* state where the
    status-only design leaves an unresolvable one.  The ``no-reconciler``
    column is the honest floor: without a service that can answer, the
    protocol converts silent duplicates into visible stalls, and 59% of
    crash-point cases stall.
  * E13a grants the window baselines more than they have.  A record counts as
    visible if the action's idempotency key appears anywhere in the text the
    target holds, which assumes a dispatcher that can recover action identity
    from raw prose.  Policies that ship a raw window carry no *structured*
    ledger for a dispatcher to consult at all; charging them for that would
    only widen the gap reported here.
"""
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import (load_all, sample_positions, boot_ci,           # noqa: E402
                    policy_budget, RES)
from state.protocol import (ProtocolLedger, derive_key,            # noqa: E402
                            CRASH_POINTS)
from state.schema import Distiller                                 # noqa: E402
from state.integrity import TaskKeys                               # noqa: E402
from runtime.dispatcher import TrustedDispatcher, ToolCall         # noqa: E402
from policies.policies import POLICIES, relay_budget               # noqa: E402
from traces.anchors import norm_text                               # noqa: E402

OUT = os.path.join(RES, "e13_protocol.json")
SEEDS = [0, 1, 2, 3, 4]

#: Model compliance with an advisory ledger, measured in E10 (recognised_done).
#: Read from the results file so this experiment cannot drift from the study
#: it depends on.
def _compliance():
    p = os.path.join(RES, "e10_llm.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))["E10b_ledger"]["per_policy"]
    return {k: v["recognised_done"] for k, v in d.items()}


#: Fingerprint divergence, measured in E12.  The dispatcher computes identity
#: from the typed call (v3); a text-derived identity is reported alongside so
#: the cost of *not* having typed calls is visible.
def _divergence():
    p = os.path.join(RES, "e12_adversarial.json")
    if not os.path.exists(p):
        return {"v3-typed": 0.0, "v2-action": 0.0}
    d = json.load(open(p))["e12a_divergence"]
    return {"v3-typed": d["v3-typed_overall"],
            "v2-action": d["v2-action_overall"]}


def _parse_args(argstr):
    import re
    out = {}
    for part in re.split(r",\s*(?=[a-z_]+=)", argstr):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out or {"arg": argstr}


# ---------------------------------------------------------------------------
def e13a_enforcement(eps, dist_s, comply, diverge, dst="watch"):
    """Duplicate rate given the action is proposed again after resumption."""
    budget = relay_budget(dst)
    policies = ["full", "recency", "typed-recency", "relay", "relay-lite",
                "trunc25", "randdrop", "restart"]
    rows = {p: {"none": [], "advisory": [], "enforced": []} for p in policies}

    for e in eps:
        traj = e.to_trajectory()
        for seed in SEEDS:
            for k in sample_positions(e.n, e.eid, seed):
                if k >= e.n:
                    continue
                executed = e.states[k].side_effects
                if not executed:
                    continue
                for pname in policies:
                    out = POLICIES[pname](traj, k, dst, distiller=dist_s,
                                          seed=seed,
                                          budget=policy_budget(pname, dst))
                    h = norm_text(out.resume_text)
                    rng = random.Random(f"{seed}|{e.eid}|{k}|{pname}")
                    cp = comply.get(pname, comply.get("relay", 0.925))
                    n_none = n_adv = n_enf = 0
                    for se in executed:
                        # record survival: is this action's entry in what the
                        # target actually holds?  (E4's measurement, per action)
                        visible = se["key"].lower() in h
                        # 1. no ledger at all
                        n_none += 1
                        # 2. advisory: the model must both see it and honour it
                        if not (visible and rng.random() < cp):
                            n_adv += 1
                        # 3. enforced: the dispatcher must see it and the
                        #    fingerprint must not have diverged
                        if not (visible and rng.random() >= diverge["v3-typed"]):
                            n_enf += 1
                    n = len(executed)
                    rows[pname]["none"].append(n_none / n)
                    rows[pname]["advisory"].append(n_adv / n)
                    rows[pname]["enforced"].append(n_enf / n)

    out = {}
    for p, d in rows.items():
        out[p] = {r: dict(zip(("mean", "lo", "hi"), boot_ci(v)))
                  for r, v in d.items()}
        out[p]["n"] = len(d["none"])
    return {"per_policy": out, "compliance_used": comply,
            "divergence_used": diverge}


# ---------------------------------------------------------------------------
#: Ground truth per crash point: did the world change?  ``None`` means the
#: outcome is genuinely undetermined at the crash and is drawn per trial.
_EFFECTED = {"c0_before_prepare": False, "c1_after_prepare": False,
             "c2_in_flight": None, "c3_after_response": True,
             "c4_after_record": True}


def _simulate_crash(design, crash, effected, reconciler_available):
    """Return one of correct / duplicate / lost / blocked.

    ``design`` is ``status_only`` (record written only when the transcript
    reports the action done) or ``wal`` (durable intent before dispatch).
    The resumed agent is assumed to propose the action again -- that is the
    conditional this experiment is defined on.
    """
    if design == "status_only":
        # a record exists only if we got all the way to recording it
        recorded = (crash == "c4_after_record")
        if recorded:
            return "correct"                     # skipped, and it did happen
        # no record -> the action is re-issued
        return "duplicate" if effected else "correct"

    # --- wal ---
    if crash == "c0_before_prepare":
        return "correct"                         # nothing written, nothing done
    if crash == "c4_after_record":
        return "correct"                         # resolved record, skipped
    # c1/c2/c3 leave a PREPARED intent -> recovery reads UNKNOWN
    if not reconciler_available:
        # fail closed: not dispatched, surfaced to the user.  This is the
        # protocol's cost, and it is a *visible* stall rather than a silent
        # duplicate.
        return "blocked"
    return "correct"                             # reconciled to the truth


def e13b_crash(n_actions=4000, seed=3, p_effect_inflight=0.5):
    rng = random.Random(seed)
    out = {}
    for design in ("status_only", "wal"):
        for recon in (True, False):
            key = f"{design}/{'reconciler' if recon else 'no-reconciler'}"
            tally = {c: defaultdict(int) for c in CRASH_POINTS}
            for _ in range(n_actions):
                c = rng.choice(CRASH_POINTS)
                eff = _EFFECTED[c]
                if eff is None:
                    eff = rng.random() < p_effect_inflight
                tally[c][_simulate_crash(design, c, eff, recon)] += 1
            agg = defaultdict(int)
            for c in CRASH_POINTS:
                for k, v in tally[c].items():
                    agg[k] += v
            tot = sum(agg.values())
            out[key] = {
                "per_crash_point": {c: dict(v) for c, v in tally.items()},
                "overall": {k: round(v / tot, 4) for k, v in agg.items()},
                "n": tot,
            }
    return out


# ---------------------------------------------------------------------------
def e13c_split_brain(eps, seed=5):
    """Do two racing resumptions present the service the same key?"""
    rng = random.Random(seed)
    keys = TaskKeys.derive(b"paired-device-task-secret")
    schemes = {"deterministic": None, "device_scoped": None, "random_nonce": None}
    agree = {s: {"n": 0, "same": 0} for s in schemes}

    for e in eps:
        for se in e.states[-1].side_effects:
            tool, args = se["action"], _parse_args(se["args"])
            # the two devices independently reach the same pending action;
            # each computes identity from its own typed call
            from state.fingerprint import typed_signature
            sig_a = typed_signature(tool, args)
            items = list(args.items())
            rng.shuffle(items)                    # different rendering order
            sig_b = typed_signature(tool, dict(items))

            for scheme in schemes:
                agree[scheme]["n"] += 1
                if scheme == "deterministic":
                    ka = derive_key(keys.idem_secret(), sig_a)
                    kb = derive_key(keys.idem_secret(), sig_b)
                elif scheme == "device_scoped":
                    ka = derive_key(keys.idem_secret() + "|watch", sig_a)
                    kb = derive_key(keys.idem_secret() + "|phone", sig_b)
                else:
                    ka = f"k-{rng.randrange(10**12)}"
                    kb = f"k-{rng.randrange(10**12)}"
                agree[scheme]["same"] += int(ka == kb)

    for s, d in agree.items():
        d["agreement"] = round(d["same"] / d["n"], 4) if d["n"] else 0.0
        # a service that honours idempotency keys executes once iff the keys
        # agree; otherwise the race produces two executions
        d["executions_per_race"] = round(2 - d["agreement"], 3)

    # --- stale-epoch fencing: the source whose ack was lost tries to act ---
    led = ProtocolLedger(task_secret=keys.idem_secret(), epoch=0)
    disp = TrustedDispatcher(led)
    call = ToolCall("purchase", {"item": "dock", "amount": 5})
    blocked = disp.check(call, observed_epoch=1).verdict
    allowed = disp.check(call, observed_epoch=0).verdict
    return {"key_agreement": agree,
            "stale_epoch_verdict": blocked.value,
            "current_epoch_verdict": allowed.value}


# ---------------------------------------------------------------------------
def e13d_integrity(eps):
    """Does sealing actually reject forgery, misdirection and replay?"""
    from state.integrity import (TaskKeys as TK, seal, unseal, ReplayWindow,
                                 IntegrityError)
    keys = TK.derive(b"paired-device-task-secret")
    blob = b'{"goal":"x","side_fx":[{"fp":"abc","status":"committed"}]}'
    env = seal(blob, keys, "task-1", epoch=2, counter=7, src="phone", dst="watch")
    wire = env.to_wire()
    win = ReplayWindow()
    res = {}

    res["honest_roundtrip"] = (unseal(wire, keys, "task-1", "watch", win) == blob)

    def _rejects(fn):
        try:
            fn()
            return False
        except IntegrityError:
            return True

    res["rejects_replay"] = _rejects(
        lambda: unseal(wire, keys, "task-1", "watch", win))
    res["rejects_wrong_destination"] = _rejects(
        lambda: unseal(wire, keys, "task-1", "speaker", ReplayWindow()))
    res["rejects_wrong_task"] = _rejects(
        lambda: unseal(wire, keys, "task-2", "watch", ReplayWindow()))
    res["rejects_wrong_key"] = _rejects(
        lambda: unseal(wire, TK.derive(b"other"), "task-1", "watch",
                       ReplayWindow()))
    tampered = bytearray(wire)
    tampered[-1] ^= 0x01
    res["rejects_tampered_ciphertext"] = _rejects(
        lambda: unseal(bytes(tampered), keys, "task-1", "watch", ReplayWindow()))
    # a stale block (older epoch) presented after a newer one
    w2 = ReplayWindow()
    newer = seal(blob, keys, "task-1", 3, 1, "phone", "watch").to_wire()
    unseal(newer, keys, "task-1", "watch", w2)
    older = seal(blob, keys, "task-1", 2, 99, "phone", "watch").to_wire()
    res["rejects_stale_epoch"] = _rejects(
        lambda: unseal(older, keys, "task-1", "watch", w2))
    res["overhead_bytes"] = len(wire) - len(blob)
    return res


def run():
    import time
    t0 = time.time()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    dist_s = Distiller(idx_s)
    comply, diverge = _compliance(), _divergence()
    out = {
        "e13a_enforcement": e13a_enforcement(eps, dist_s, comply, diverge),
        "e13b_crash": e13b_crash(),
        "e13c_split_brain": e13c_split_brain(eps),
        "e13d_integrity": e13d_integrity(eps),
        "runtime_s": round(time.time() - t0, 1),
    }
    json.dump(out, open(OUT, "w"), indent=1)

    a = out["e13a_enforcement"]["per_policy"]
    print("E13a duplicate rate | proposed again (watch target)")
    print(f"{'policy':14s} {'no ledger':>10s} {'advisory':>10s} {'enforced':>10s}")
    for p, d in a.items():
        print(f"{p:14s} {d['none']['mean']:10.3f} {d['advisory']['mean']:10.3f} "
              f"{d['enforced']['mean']:10.3f}")
    print("\nE13b crash outcomes:")
    for k, v in out["e13b_crash"].items():
        print(f"  {k:34s} {v['overall']}")
    print("\nE13c", json.dumps(out["e13c_split_brain"], indent=1)[:600])
    print("\nE13d", json.dumps(out["e13d_integrity"]))
    print("->", OUT)


if __name__ == "__main__":
    run()
