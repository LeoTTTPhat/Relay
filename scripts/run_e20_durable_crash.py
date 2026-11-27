"""E20: the write-ahead protocol against a real durable store and real crashes.

E13b was a simulation over the action state machine.  Reviewers asked for the
protocol to be fault-injected against a real store.  This experiment does
that on one host, with no simulation in the loop:

  * each agent runtime is a separate OS process (``e20_worker.py``) whose
    ledger is a SQLite file in WAL mode with ``synchronous=FULL`` and
    ``fullfsync`` (``runtime/durable.py``);
  * the side-effectful service is another OS process (``e20_endpoint.py``)
    speaking HTTP over loopback and logging every executed effect durably --
    the ground truth, which the runtime never reads;
  * a crash is ``SIGKILL`` delivered by this parent process at the moment the
    worker announces it has reached a protocol boundary: before the intent
    (c0), after the durable intent (c1), with the request in flight (c2),
    after the effect's response arrived but before COMMITTED was written (c3),
    and after the record (c4), plus a no-crash control;
  * recovery is a fresh process over the same file, which proposes the same
    action again, and the outcome is read from the service's effect log.

Designs: ``wal`` (Relay) and ``status_only`` (record written only when the
action is reported done), both with the same deterministic key.  Service
contracts: ``plain`` (no key, no query), ``keyed`` (idempotency key, no
query) and ``queryable`` (key + status query).  For c2, half the trials ask
the service to abandon a request whose client disconnected, so an in-flight
crash has a genuinely undetermined outcome.

Outcome per trial, from the effect log after recovery:
  ``correct``    exactly one effect and the runtime did not block;
  ``duplicate``  two or more effects;
  ``blocked``    the dispatcher failed closed (RECONCILE surfaced to the user);
                 split by whether the effect had in fact happened;
  ``lost``       no effect and no block (the runtime wrongly skipped).

Also measured: (E20b) two racing runtimes with separate stores and the same
deterministic key vs device-scoped keys; (E20c) the latency of a durable
intent on this host's SSD under FULL+fullfsync vs NORMAL -- a workstation
measurement, not a wearable one; (E20d) the anti-replay window fix.

Scope: process crashes only.  Power loss and kernel crashes are not tested;
reconciliation queries are issued only after the service's in-flight
processing window has elapsed (a status query is conclusive only once the
original request can no longer take effect).
"""
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from explib import RES                                              # noqa: E402

OUT = os.path.join(RES, "e20_durable_crash.json")
PY = sys.executable
N_PER_CELL = int(os.environ.get("RELAY_E20_N", "30"))
CRASHES = ["none", "c0_before_prepare", "c1_after_prepare", "c2_in_flight",
           "c3_after_response", "c4_after_record"]
SETTLE_S = 0.35          # > endpoint PRE+RESP delay: in-flight work finished


def _actions():
    """Typed side effects from the synthetic corpus (as E12/E13 use)."""
    from synth.generator import generate
    from run_e12_adversarial import _parse_args
    out = []
    for e in generate(200, seed=0):
        for se in e.states[-1].side_effects:
            out.append((se["action"], _parse_args(se["args"])))
    return out


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                timeout=30) as r:
        return json.loads(r.read().decode())


def one_trial(port, tmp, design, contract, crash, i, action, rng_abort):
    uid = f"{design}-{contract}-{crash}-{i}"
    tool, args = action
    args = dict(args, trial=uid)
    act = json.dumps({"tool": tool, "args": args, "action_id": uid})
    db = os.path.join(tmp, uid + ".db")
    base = [PY, os.path.join(HERE, "e20_worker.py"), "--db", db,
            "--port", str(port), "--contract", contract, "--design", design,
            "--action", act]
    if crash == "c2_in_flight" and rng_abort:
        base.append("--abort-on-disconnect")
    p = subprocess.Popen(base + ["--phase", "run", "--crash", crash],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    killed, run_ms = False, None
    for line in p.stdout:
        if line.startswith("AT "):
            os.kill(p.pid, signal.SIGKILL)
            killed = True
            break
        if line.startswith("DONE"):
            run_ms = json.loads(line[5:])["elapsed_ms"]
            break
    p.wait()
    time.sleep(SETTLE_S)
    eff_before = _get(port, f"/effects?action_id={uid}")["count"]
    r = subprocess.run(base + ["--phase", "resume"], capture_output=True,
                       text=True, timeout=120)
    try:
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        res = {"error": r.stderr[-400:]}
    time.sleep(0.02)
    eff_after = _get(port, f"/effects?action_id={uid}")["count"]
    blocked = (res.get("verdict") == "reconcile" and not res.get("dispatched"))
    if eff_after >= 2:
        outcome = "duplicate"
    elif blocked:
        outcome = "blocked_effected" if eff_after == 1 else "blocked_not_effected"
    elif eff_after == 1:
        outcome = "correct"
    else:
        outcome = "lost"
    return {"design": design, "contract": contract, "crash": crash, "i": i,
            "killed": killed, "abort_on_disconnect": bool(
                crash == "c2_in_flight" and rng_abort),
            "effects_before_resume": eff_before, "effects_after": eff_after,
            "resume": res, "outcome": outcome, "run_ms": run_ms}


def race_trial(port, tmp, contract, scheme, i, action):
    """Two runtimes, separate stores, reach the same pending action at once."""
    uid = f"race-{contract}-{scheme}-{i}"
    tool, args = action
    act = json.dumps({"tool": tool, "args": dict(args, trial=uid),
                      "action_id": uid})
    procs = []
    for dev in ("watch", "phone"):
        secret = "paired-task-secret" + (f"|{dev}" if scheme == "device_scoped"
                                         else "")
        db = os.path.join(tmp, f"{uid}-{dev}.db")
        procs.append(subprocess.Popen(
            [PY, os.path.join(HERE, "e20_worker.py"), "--db", db, "--port",
             str(port), "--contract", contract, "--design", "wal", "--action",
             act, "--phase", "run", "--secret", secret],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True))
    for p in procs:
        p.wait()
    time.sleep(SETTLE_S)
    return {"contract": contract, "scheme": scheme, "i": i,
            "effects": _get(port, f"/effects?action_id={uid}")["count"]}


def latency_bench(tmp, n=200):
    from runtime.durable import SQLiteLedger
    out = {}
    for label, sync, ff in (("FULL+fullfsync", "FULL", True),
                            ("NORMAL", "NORMAL", False)):
        path = os.path.join(tmp, f"bench-{label}.db")
        led = SQLiteLedger(path, synchronous=sync, fullfsync=ff)
        ts = []
        for j in range(n):
            t0 = time.perf_counter()
            led.prepare(f"fp{j:05d}", f"sig{j}")
            ts.append(1000 * (time.perf_counter() - t0))
        led.close()
        ts.sort()
        out[label] = {"n": n, "median_ms": round(ts[n // 2], 3),
                      "p95_ms": round(ts[int(n * 0.95)], 3),
                      "config": led.config}
    return out


def replay_window_check():
    """E20d: an unauthenticated envelope must not advance the window."""
    from state.integrity import (TaskKeys, seal, unseal, ReplayWindow,
                                 Envelope, IntegrityError)
    keys = TaskKeys.derive(b"paired-device-task-secret")
    good = seal(b'{"ledger":[]}', keys, "task-1", 2, 1, "phone", "watch")
    forged = Envelope(kid=keys.kid, task_id="task-1", epoch=10 ** 9,
                      counter=1, src="phone", dst="watch",
                      nonce=os.urandom(12), ct=os.urandom(40)).to_wire()
    # pre-revision ordering, reproduced explicitly: update, then authenticate
    old = ReplayWindow()
    old.check_and_update(10 ** 9, 1)          # what the old unseal did first
    try:
        old.check_and_update(2, 1)
        old_dos = False
    except IntegrityError:
        old_dos = True
    # shipped ordering
    new = ReplayWindow()
    try:
        unseal(forged, keys, "task-1", "watch", new)
    except IntegrityError:
        pass
    try:
        unseal(good.to_wire(), keys, "task-1", "watch", new)
        new_dos = False
    except IntegrityError:
        new_dos = True
    return {"pre_fix_forged_envelope_blocks_legitimate": old_dos,
            "post_fix_forged_envelope_blocks_legitimate": new_dos}


def summarise(rows):
    cells = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cells[(r["design"], r["contract"], r["crash"])][r["outcome"]] += 1
    per_cell = {"|".join(k): dict(v) for k, v in sorted(cells.items())}
    agg = defaultdict(lambda: defaultdict(int))
    for (d, c, cr), v in cells.items():
        if cr == "none":
            continue
        for o, n in v.items():
            agg[f"{d}|{c}"][o] += n
    overall = {}
    for k, v in agg.items():
        tot = sum(v.values())
        overall[k] = {"n": tot, **{o: round(n / tot, 4) for o, n in v.items()}}
    c2 = [r for r in rows if r["crash"] == "c2_in_flight"]
    return {"per_cell_counts": per_cell, "overall_crash_trials": overall,
            "c2_effected_before_resume": sum(
                1 for r in c2 if r["effects_before_resume"] >= 1),
            "c2_trials": len(c2)}


def run():
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="relay-e20-")
    world = os.path.join(tmp, "world.db")
    srv = subprocess.Popen([PY, os.path.join(HERE, "e20_endpoint.py"),
                            "--db", world], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True)
    port = int(srv.stdout.readline().split()[1])
    acts = _actions()
    rng = random.Random(20)
    jobs = []
    i = 0
    for design in ("status_only", "wal"):
        for contract in ("plain", "keyed", "queryable"):
            for crash in CRASHES:
                for j in range(N_PER_CELL):
                    jobs.append((design, contract, crash, j,
                                 acts[i % len(acts)], rng.random() < 0.5))
                    i += 1
    rows = []
    lock = threading.Lock()
    try:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(one_trial, port, tmp, *j) for j in jobs]
            for n, f in enumerate(futs):
                r = f.result()
                with lock:
                    rows.append(r)
                if n % 120 == 119:
                    print(f"{n+1}/{len(jobs)} trials, "
                          f"{time.time()-t0:.0f}s", flush=True)
        races = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = []
            k = 0
            for contract in ("plain", "keyed"):
                for scheme in ("deterministic", "device_scoped"):
                    for j in range(N_PER_CELL):
                        futs.append(ex.submit(race_trial, port, tmp, contract,
                                              scheme, j, acts[k % len(acts)]))
                        k += 1
            races = [f.result() for f in futs]
        lat = latency_bench(tmp)
    finally:
        srv.kill()
        srv.wait()
    race_sum = defaultdict(lambda: defaultdict(int))
    for r in races:
        race_sum[f"{r['contract']}|{r['scheme']}"][str(r["effects"])] += 1
    out = {"n_per_cell": N_PER_CELL, "n_trials": len(rows),
           "summary": summarise(rows),
           "race_effect_counts": {k: dict(v) for k, v in race_sum.items()},
           "durable_prepare_latency": lat,
           "replay_window": replay_window_check(),
           "store_config": rows[0]["resume"].get("store") if rows else None,
           "runtime_s": round(time.time() - t0, 1),
           "rows": rows}
    json.dump(out, open(OUT, "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    s = out["summary"]
    for k, v in sorted(s["overall_crash_trials"].items()):
        print(f"{k:28s} {v}")
    print("c2 effected before resume:", s["c2_effected_before_resume"], "/",
          s["c2_trials"])
    print("races:", json.dumps(out["race_effect_counts"]))
    print("latency:", json.dumps(lat))
    print("replay window:", out["replay_window"])
    print("->", OUT, out["runtime_s"], "s")


if __name__ == "__main__":
    run()
