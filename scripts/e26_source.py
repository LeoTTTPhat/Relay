"""E26 source: a laptop-class runtime that starts a task and hands it off.

Qwen2.5-7B-Instruct on the GPU (MPS), full context, the same typed-action
menu protocol as E23, a trusted dispatcher over its own durable SQLite ledger
and the shared endpoint.  After it has performed about half of the episode's
world-changing actions it hands the task off:

  1. fence: durably record the new epoch (D3), so this runtime can no longer
     dispatch;
  2. snapshot its ledger (D4: an action still in flight ships as UNKNOWN);
  3. distil its own trajectory -- rendered in the synthetic-episode step
     format -- with Relay's distiller at the watch budget, and serialise it;
  4. seal (AES-GCM, task/epoch/counter/src/dst bound) and send over a TCP
     socket to the separately launched target; wait for the ACK;
  5. re-send the identical sealed bytes (network replay), and attempt one
     more dispatch itself (stale source); both must be refused;
  6. wait for the target to finish, then move to the next episode.

In "in-flight" episodes the last source action is still awaiting its
response (the endpoint delays it) when the user walks away, so the snapshot
carries it as UNKNOWN and the source never observes the outcome.
"""
import argparse
import base64
import json
import math
import os
import socket
import threading
import time

from e26_common import (send_msg, recv_msg, LM, MENU_PROMPT, TURN_PROMPT,
                        parse_call, effect_tool, reconciler, choice_of,
                        TASK_SECRET)
from run_e23_continuation import build_world
from explib import load_all
from runtime.dispatcher import (TrustedDispatcher, ToolCall,
                                EndpointContract, Verdict)
from runtime.durable import SQLiteLedger
from state.integrity import TaskKeys, seal
from state.schema import Distiller
from state.protocol import Status
from traces.parser import Step, Trajectory, est_tokens
from policies.policies import relay_budget


def step(i, agent, text, se=False, tool=False):
    return Step(idx=i, role="assistant", agent=agent, content=text,
                side_effect_like=se, is_tool_result=tool,
                tokens=est_tokens(text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--endpoint", type=int, required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--episodes", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--inflight", default="5,6,7,8,9")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    a = ap.parse_args()

    trajs, idx, eps, synth, idx_s = load_all()
    dist = Distiller(idx_s)
    t0 = time.perf_counter()
    lm = LM(a.model, device="mps")
    load_s = time.perf_counter() - t0
    keys = TaskKeys.derive(TASK_SECRET)
    inflight = {int(x) for x in a.inflight.split(",") if x}
    summary = {"model_load_s": round(load_s, 1), "episodes": []}

    for ei in [int(x) for x in a.episodes.split(",")]:
        e = eps[ei]
        task = e.eid
        items, _, _ = build_world(e, 0)
        effects = [it for it in items if it["kind"] in ("effect",
                                                        "distractor")]
        n_true = sum(1 for it in items if it["kind"] == "effect")
        h = max(1, math.ceil(n_true / 2))
        for it in items:
            it["aid"] = f"{task}|{it['id']}"
        menu = "\n".join(f"{it['mid']}: {it['label']}" for it in items)
        by_mid = {it["mid"]: it for it in items}
        led = SQLiteLedger(os.path.join(a.workdir, f"{task}-source.db"),
                           task_secret=keys.idem_secret())
        disp = TrustedDispatcher(led, reconciler=reconciler(a.endpoint),
                                 contract=EndpointContract(True, True))
        steps = [e.steps[0]]                   # the task's plan step
        history, turns, executed = [], [], []
        committed_labels = []
        inflight_label = None
        for turn in range(n_true + 4):
            if len(executed) >= h:
                break
            traj_text = "\n".join(f"[step {s.idx} {s.agent}] {s.content}"
                                  for s in steps)
            prefix = MENU_PROMPT.format(
                intro="The goal and everything done so far on this device is "
                      "below.", title="TASK LOG",
                state=e.goal + "\n" + traj_text, menu=menu)
            pre, post = lm.split(prefix)
            pid, cache, pf = lm.prefix_cache(pre)
            tg = time.perf_counter()
            out = lm.gen(pid, cache, TURN_PROMPT.format(
                history="\n".join(history) or "(none yet)") + post)
            turns.append({"choice": choice_of(out),
                          "prefill_s": round(pf, 3),
                          "gen_s": round(time.perf_counter() - tg, 3)})
            ch = choice_of(out)
            it = by_mid.get(ch)
            if ch == "D0" or it is None:
                history.append(f"Turn {turn + 1}: "
                               + ("DONE refused: task not finished"
                                  if ch == "D0" else "invalid reply"))
                continue
            if it["kind"] == "lookup":
                obs = f"RESULT {it['result']}"
                steps.append(step(len(steps), "tool-runner",
                                  f"TOOL {it['label'].replace('(', '(', 1)} "
                                  f"-> {it['result']}. ", tool=True))
                history.append(f"Turn {turn + 1}: {ch} {it['label']} -> {obs}")
                continue
            tool, args = parse_call(it["label"])
            call = ToolCall(tool, args)
            last = (len(executed) == h - 1) and ei in inflight
            if last:
                # the user walks away while this request is in flight
                d = disp.check(call)
                if d.dispatched:
                    rec = led.prepare(d.fp, call.signature(),
                                      family=call.family())
                    th = threading.Thread(
                        target=lambda: _swallow(effect_tool(
                            a.endpoint, it["aid"], delay_ms=5000),
                            {**call.args, "idempotency_key": rec.key}),
                        daemon=True)
                    th.start()
                    time.sleep(0.5)            # request sent; effect applied
                    inflight_label = it["label"]
                    executed.append(it["mid"])
                    history.append(f"Turn {turn + 1}: {ch} {it['label']} -> "
                                   f"(in flight)")
                continue
            d, res = disp.execute(call, effect_tool(a.endpoint, it["aid"]))
            if d.dispatched:
                rec = led.get(d.fp)
                steps.append(step(len(steps), "executor",
                                  f"EXECUTED {it['label']} [idempotency-key "
                                  f"{rec.key}] status=done.", se=True))
                executed.append(it["mid"])
                committed_labels.append((it["label"], it["aid"]))
                obs = "OK, performed"
            else:
                obs = "SKIPPED by the runtime: already performed"
            history.append(f"Turn {turn + 1}: {ch} {it['label']} -> {obs}")

        # ---------------- handoff ----------------
        th0 = time.perf_counter()
        new_epoch = led.fence(led.epoch + 1)                    # D3
        snap = led.snapshot_for_handoff()                       # D4
        t_snap = time.perf_counter()
        traj = Trajectory(tid=task, source="synthetic", question=e.goal,
                          ground_truth="", steps=steps)
        block = dist.distill(traj, len(steps) - 1, relay_budget("watch"))
        blob = block.serialize()
        t_dist = time.perf_counter()
        payload = json.dumps({
            "epoch": new_epoch, "ledger": snap,
            "block": base64.b64encode(blob).decode(),
            "menu": [{"mid": it["mid"], "label": it["label"],
                      "kind": "lookup" if it["kind"] == "lookup" else "effect",
                      "result": it.get("result", ""), "aid": it["aid"]}
                     for it in items],
            "turn_cap": n_true - len(committed_labels) + 3,
            "replay_label": committed_labels[-1][0] if committed_labels
            else None,
            "replay_aid": committed_labels[-1][1] if committed_labels
            else None}).encode()
        env = seal(payload, keys, task, new_epoch, 1, "laptop", "watch")
        wire = env.to_wire()
        t_seal = time.perf_counter()
        sock = socket.create_connection(("127.0.0.1", a.target))
        send_msg(sock, {"type": "handoff", "task": task}, wire)
        ack, _ = recv_msg(sock)
        t_ack = time.perf_counter()
        # network replay of the identical sealed bytes on a new connection
        s2 = socket.create_connection(("127.0.0.1", a.target))
        send_msg(s2, {"type": "handoff", "task": task}, wire)
        replay, _ = recv_msg(s2)
        s2.close()
        # the stale source tries to act once more
        nxt = [it for it in effects if it["mid"] not in executed]
        stale = None
        if nxt:
            tool, args = parse_call(nxt[0]["label"])
            stale = disp.check(ToolCall(tool, args),
                               observed_epoch=led.fenced_epoch()).verdict.value
        done, _ = recv_msg(sock)                 # target finished the task
        sock.close()
        rec = {"task": task, "family": e.family, "inflight": ei in inflight,
               "n_true_effects": n_true, "handoff_after": h,
               "source_turns": turns, "source_executed_mids": executed,
               "inflight_label": inflight_label,
               "snapshot_statuses": [r["status"] for r in snap],
               "block_tokens_est": block.tokens(), "block_bytes": len(blob),
               "wire_bytes": len(wire),
               "ms": {"fence_snapshot": round(1000 * (t_snap - th0), 2),
                      "distil_serialise": round(1000 * (t_dist - t_snap), 2),
                      "seal": round(1000 * (t_seal - t_dist), 2),
                      "send_verify_install_ack": round(1000 * (t_ack - t_seal),
                                                       2)},
               "ack": ack, "network_replay": replay,
               "stale_source_verdict": stale}
        summary["episodes"].append(rec)
        json.dump(summary, open(os.path.join(a.workdir, "source.json"), "w"),
                  indent=1)
        print(f"{task}: ack={ack.get('type')} replay={replay.get('type')} "
              f"stale={stale}", flush=True)
        led.close()
    s = socket.create_connection(("127.0.0.1", a.target))
    send_msg(s, {"type": "quit"})
    s.close()


def _swallow(fn, args):
    try:
        fn(args)
    except Exception:
        pass


if __name__ == "__main__":
    main()
