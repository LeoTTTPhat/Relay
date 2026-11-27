"""E26: a same-host, multi-process handoff prototype of one real task.

Reviewer 4 asked why no prototype migrates a task from a laptop-class to a
watch-class runtime.  We have no watch and no radio in reach of this
revision, so this is the closest honest construction:

  * three OS processes on one workstation -- the service (``e20_endpoint``),
    a laptop-class source runtime (``e26_source``: Qwen2.5-7B-Instruct on the
    GPU) and a separately launched watch-class target runtime (``e26_target``:
    Qwen2.5-0.5B-Instruct on 2 CPU threads, resumption window capped at the
    watch class's usable window);
  * real mechanisms end to end: typed tool calls through a trusted dispatcher,
    durable SQLite ledgers on both sides, Relay's distiller at the watch
    budget, an AES-GCM-sealed block over a loopback TCP socket, anti-replay,
    epoch fencing, reconciliation against the service's status API;
  * adversarial probes: a forged envelope with a huge epoch before the first
    handoff; the identical sealed bytes replayed after each handoff; the stale
    source attempting one more action; the last pre-handoff side effect
    replayed below the target's model.

Ten synthetic episodes (two per family); in five, the source's last action is
still in flight when the handoff happens.  Ground truth is the service's
effect log: per episode, duplicates (any action executed twice), constraint
violations (the distractor executed), and whether every world-changing action
of the episode was eventually performed.

Same-host emulation: no physical devices, no radio, no battery, no memory cap.
Timings are workstation timings of the two model classes, not device timings.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from explib import RES, load_all                                    # noqa: E402
from run_e23_continuation import build_world                        # noqa: E402
from e26_common import send_msg, recv_msg, TASK_SECRET              # noqa: E402

OUT = os.path.join(RES, "e26_prototype.json")
PY = sys.executable


def effects_count(port, aid):
    with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/effects?action_id="
            + urllib.request.quote(aid), timeout=30) as r:
        return json.loads(r.read().decode())["count"]


def forged_probe(target_port, task):
    import socket
    from state.integrity import Envelope, TaskKeys
    keys = TaskKeys.derive(TASK_SECRET)
    forged = Envelope(kid=keys.kid, task_id=task, epoch=10 ** 9, counter=1,
                      src="laptop", dst="watch", nonce=os.urandom(12),
                      ct=os.urandom(64)).to_wire()
    s = socket.create_connection(("127.0.0.1", target_port))
    send_msg(s, {"type": "handoff", "task": task}, forged)
    resp, _ = recv_msg(s)
    s.close()
    return resp


def run():
    t0 = time.time()
    work = tempfile.mkdtemp(prefix="relay-e26-")
    episodes = os.environ.get("RELAY_E26_EPISODES", "0,1,2,3,4,5,6,7,8,9")
    inflight = os.environ.get("RELAY_E26_INFLIGHT", "5,6,7,8,9")
    srv = subprocess.Popen([PY, os.path.join(HERE, "e20_endpoint.py"), "--db",
                            os.path.join(work, "world.db")],
                           stdout=subprocess.PIPE, text=True)
    ep_port = int(srv.stdout.readline().split()[1])
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false")
    tgt = subprocess.Popen([PY, os.path.join(HERE, "e26_target.py"),
                            "--endpoint", str(ep_port), "--workdir", work,
                            "--threads", "2"], stdout=subprocess.PIPE,
                           text=True, env=env)
    line = tgt.stdout.readline()
    tgt_port = int(line.split()[1])
    target_load_s = float(line.split("model_load_s=")[1])
    trajs, idx, eps, synth, idx_s = load_all()
    ids = [int(x) for x in episodes.split(",")]
    forged = forged_probe(tgt_port, eps[ids[0]].eid)
    try:
        src = subprocess.run([PY, os.path.join(HERE, "e26_source.py"),
                              "--target", str(tgt_port), "--endpoint",
                              str(ep_port), "--workdir", work, "--episodes",
                              episodes, "--inflight", inflight, "--model",
                              os.environ.get("RELAY_E26_SOURCE_MODEL",
                                             "Qwen/Qwen2.5-7B-Instruct")],
                             capture_output=True, text=True, timeout=5400)
        if src.returncode != 0:
            print(src.stderr[-3000:])
        tgt.wait(timeout=120)
        source = json.load(open(os.path.join(work, "source.json")))
        rows = []
        for rec in source["episodes"]:
            e = next(x for x in eps if x.eid == rec["task"])
            items, _, _ = build_world(e, 0)
            counts = {it["id"]: effects_count(ep_port, f"{e.eid}|{it['id']}")
                      for it in items if it["kind"] in ("effect",
                                                        "distractor")}
            true_ids = [it["id"] for it in items if it["kind"] == "effect"]
            target = json.load(open(os.path.join(work,
                                                 f"{e.eid}-target.json")))
            rows.append({
                **rec, "target": target,
                "duplicate": any(counts[i] >= 2 for i in true_ids),
                "violation": counts.get("distractor", 0) >= 1,
                "all_performed": all(counts[i] >= 1 for i in true_ids),
                "effect_counts": counts})
        out = {"target_model": "Qwen/Qwen2.5-0.5B-Instruct (cpu, 2 threads)",
               "source_model": os.environ.get("RELAY_E26_SOURCE_MODEL",
                                              "Qwen/Qwen2.5-7B-Instruct")
               + " (mps)",
               "target_model_load_s": target_load_s,
               "source_model_load_s": source.get("model_load_s"),
               "forged_envelope_before_first_handoff": forged,
               "n_episodes": len(rows),
               "n_duplicates": sum(r["duplicate"] for r in rows),
               "n_violations": sum(r["violation"] for r in rows),
               "n_all_performed": sum(r["all_performed"] for r in rows),
               "n_ack": sum(r["ack"].get("type") == "ack" for r in rows),
               "n_replay_rejected": sum(r["network_replay"].get("type")
                                        == "reject" for r in rows),
               "stale_source_verdicts": [r["stale_source_verdict"]
                                         for r in rows],
               "replay_probe_verdicts": [(r["target"]["replay_probe"] or
                                          {}).get("verdict") for r in rows],
               "runtime_s": round(time.time() - t0, 1),
               "source_stderr_tail": src.stderr[-800:],
               "episodes": rows}
        json.dump(out, open(OUT, "w"), indent=1)
        print(json.dumps({k: v for k, v in out.items()
                          if k not in ("episodes", "source_stderr_tail")},
                         indent=1))
    finally:
        for p in (tgt, srv):
            try:
                p.kill()
            except Exception:
                pass
        shutil.rmtree(work, ignore_errors=True)
    print("->", OUT)


if __name__ == "__main__":
    run()
