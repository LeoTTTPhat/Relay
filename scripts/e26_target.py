"""E26 target: a separately launched, resource-capped watch-class runtime.

CPU only, ``--threads`` torch threads (default 2), a Qwen2.5-0.5B-Instruct
model, and a hard resumption window equal to the watch class's usable window
(the block is capped with the same drop-oldest rule as every experiment).  It
is not memory-capped (macOS does not enforce RLIMIT_AS) and it is not a watch.

For each connection: receive a sealed handoff, verify it (AES-GCM, task,
destination, anti-replay window), install the ledger snapshot into its own
durable SQLite store, acknowledge, then continue the task through its own
trusted dispatcher against the shared endpoint, and report.  A connection
carrying ``replay`` or ``forged`` bytes exercises the rejection paths.
"""
import argparse
import base64
import gzip
import json
import os
import socket
import sys
import time

from e26_common import (send_msg, recv_msg, LM, MENU_PROMPT, TURN_PROMPT,
                        parse_call, effect_tool, reconciler, choice_of,
                        TASK_SECRET)
from runtime.dispatcher import TrustedDispatcher, ToolCall, EndpointContract
from runtime.durable import SQLiteLedger, load_snapshot
from state.integrity import TaskKeys, unseal, ReplayWindow, IntegrityError
from state.schema import StateBlock
from state.protocol import Status
from tok.tokenizer import cap_suffix
from policies.policies import usable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=0)
    ap.add_argument("--endpoint", type=int, required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    t0 = time.perf_counter()
    lm = LM(a.model, device="cpu", threads=a.threads, dtype="float32")
    load_s = time.perf_counter() - t0
    keys = TaskKeys.derive(TASK_SECRET)
    windows = {}
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", a.listen))
    srv.listen(4)
    print(f"LISTEN {srv.getsockname()[1]} model_load_s={load_s:.1f}",
          flush=True)
    while True:
        conn, _ = srv.accept()
        hdr, wire = recv_msg(conn)
        if hdr.get("type") == "quit":
            conn.close()
            break
        task = hdr["task"]
        win = windows.setdefault(task, ReplayWindow())
        tv = time.perf_counter()
        try:
            payload = unseal(wire, keys, task, "watch", win)
        except IntegrityError as exc:
            send_msg(conn, {"type": "reject", "reason": str(exc)})
            conn.close()
            continue
        verify_ms = 1000 * (time.perf_counter() - tv)
        p = json.loads(payload.decode())
        tl = time.perf_counter()
        db = os.path.join(a.workdir, f"{task}-target.db")
        led = SQLiteLedger(db, task_secret=keys.idem_secret())
        led.set_epoch(p["epoch"])
        n_installed = load_snapshot(led, p["ledger"])
        recovered = len(led.recover())
        load_ms = 1000 * (time.perf_counter() - tl)
        send_msg(conn, {"type": "ack", "epoch": p["epoch"],
                        "verify_ms": round(verify_ms, 2),
                        "load_ms": round(load_ms, 2),
                        "records": n_installed})

        # ---- continue the task ----
        block = StateBlock.deserialize(base64.b64decode(p["block"]))
        text, full_tok, truncated = cap_suffix(block.render(), usable("watch"))
        menu_items = p["menu"]
        menu = "\n".join(f"{it['mid']}: {it['label']}" for it in menu_items)
        by_mid = {it["mid"]: it for it in menu_items}
        prefix = MENU_PROMPT.format(
            intro="You are resuming it on this device; the state transferred "
                  "from the other device is below.",
            title="TRANSFERRED STATE", state=text, menu=menu)
        pre, post = lm.split(prefix)
        pid, cache, prefill_s = lm.prefix_cache(pre)
        disp = TrustedDispatcher(led, reconciler=reconciler(a.endpoint),
                                 contract=EndpointContract(True, True))
        history, turns, verdicts = [], [], []
        replay_probe = None
        if p.get("replay_label"):
            # a replayed side effect arriving below the model: the runtime is
            # handed the last pre-handoff action again before the model acts
            rt, ra = parse_call(p["replay_label"])
            rd, _ = disp.execute(ToolCall(rt, ra),
                                 effect_tool(a.endpoint, p["replay_aid"]))
            replay_probe = {"verdict": rd.verdict.value,
                            "dispatched": rd.dispatched, "reason": rd.reason}
        for turn in range(p["turn_cap"]):
            tg = time.perf_counter()
            out = lm.gen(pid, cache, TURN_PROMPT.format(
                history="\n".join(history) or "(none yet)") + post)
            gen_s = time.perf_counter() - tg
            ch = choice_of(out)
            turns.append({"choice": ch, "gen_s": round(gen_s, 3)})
            if ch == "D0":
                break
            it = by_mid.get(ch)
            if it is None:
                history.append(f"Turn {turn + 1}: invalid reply")
                continue
            if it["kind"] == "lookup":
                history.append(f"Turn {turn + 1}: {ch} {it['label']} -> "
                               f"RESULT {it['result']}")
                continue
            tool, args = parse_call(it["label"])
            call = ToolCall(tool, args)
            d, _ = disp.execute(call, effect_tool(a.endpoint, it["aid"]))
            verdicts.append({"mid": ch, "verdict": d.verdict.value,
                             "dispatched": d.dispatched, "reason": d.reason})
            obs = ("OK, performed" if d.dispatched else
                   "SKIPPED by the runtime: already performed"
                   if d.verdict.value == "skip_duplicate" else
                   f"NOT PERFORMED: {d.reason}")
            history.append(f"Turn {turn + 1}: {ch} {it['label']} -> {obs}")
        res = {"task": task, "block_tokens": full_tok,
               "truncated_at_target": truncated,
               "prefill_tokens": int(pid.shape[1]),
               "prefill_s": round(prefill_s, 3),
               "prefill_tok_per_s": round(pid.shape[1] / prefill_s, 1),
               "turns": turns, "verdicts": verdicts,
               "replay_probe": replay_probe, "recovered_unknown": recovered,
               "unknown_in_snapshot": sum(1 for r in p["ledger"]
                                          if r["status"] == "unknown"),
               "final_statuses": {r.fp: r.status.value
                                  for r in led.entries()}}
        json.dump(res, open(os.path.join(a.workdir, f"{task}-target.json"),
                            "w"), indent=1)
        send_msg(conn, {"type": "done"})
        conn.close()
        led.close()


if __name__ == "__main__":
    main()
