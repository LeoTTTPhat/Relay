"""E20 worker: one agent runtime process that dispatches one side effect.

``--phase run``    executes the action through the guarded path and, when
                   ``--crash`` names a protocol boundary, announces reaching it
                   on stdout and blocks until the parent SIGKILLs it;
``--phase resume`` is a *fresh* process over the same durable ledger file: it
                   runs recovery and then proposes the same action again (the
                   conditional every E13/E20 rate is defined on).

``--design wal``          intent committed durably before dispatch (Relay);
``--design status_only``  a record is written only once the action is
                          reported done (the original design), with the same
                          deterministic key so the comparison isolates the
                          write-ahead record.
"""
import argparse
import http.client
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from runtime.dispatcher import (TrustedDispatcher, ToolCall,        # noqa: E402
                                EndpointContract)
from runtime.durable import SQLiteLedger                            # noqa: E402
from state.protocol import ActionRecord, Status, derive_key         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--contract", choices=["plain", "keyed", "queryable"])
    ap.add_argument("--design", choices=["wal", "status_only"], default="wal")
    ap.add_argument("--action", required=True)       # JSON {tool,args,action_id}
    ap.add_argument("--phase", choices=["run", "resume"], required=True)
    ap.add_argument("--crash", default="none")
    ap.add_argument("--abort-on-disconnect", action="store_true")
    ap.add_argument("--secret", default="paired-task-secret")
    a = ap.parse_args()

    act = json.loads(a.action)
    call = ToolCall(act["tool"], act["args"])
    ledger = SQLiteLedger(a.db, task_secret=a.secret)

    def hook(point):
        if a.phase == "run" and point == a.crash:
            print(f"AT {point}", flush=True)
            sys.stdin.read()                  # wait for SIGKILL
            os._exit(3)                       # parent died: never continue

    def tool_impl(args):
        key = args.pop("idempotency_key", "")
        body = json.dumps({"action_id": act["action_id"], "tool": act["tool"],
                           "args": args})
        path = f"/{a.contract}/act" + ("?abort_on_disconnect=1"
                                        if a.abort_on_disconnect else "")
        conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)
        conn.request("POST", path, body=body,
                     headers={"Content-Type": "application/json",
                              "Idempotency-Key": key})
        hook("c2_in_flight")                  # request bytes have left
        resp = conn.getresponse()
        data = json.loads(resp.read().decode())
        conn.close()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}")
        return data

    def reconciler(rec):
        conn = http.client.HTTPConnection("127.0.0.1", a.port, timeout=30)
        conn.request("GET", f"/{a.contract}/status?key={rec.key}")
        resp = conn.getresponse()
        data = json.loads(resp.read().decode())
        conn.close()
        if resp.status != 200:
            return None
        return bool(data.get("found"))

    if a.contract == "queryable":
        disp = TrustedDispatcher(ledger, reconciler=reconciler,
                                 contract=EndpointContract(True, True))
    elif a.contract == "keyed":
        disp = TrustedDispatcher(ledger, contract=EndpointContract(True, False))
    else:
        disp = TrustedDispatcher(ledger)          # fail closed

    if a.phase == "run":
        t0 = time.perf_counter()
        if a.design == "wal":
            d, res = disp.execute(call, tool_impl, hook=hook)
        else:
            # status-only: nothing is written until the action is done
            hook("c0_before_prepare")
            hook("c1_after_prepare")
            key = derive_key(a.secret, call.signature())
            res = tool_impl({**call.args, "idempotency_key": key})
            hook("c3_after_response")
            ledger.put_record(ActionRecord(fp=call.fingerprint(), key=key,
                                           status=Status.COMMITTED,
                                           family=call.family()), "record")
            hook("c4_after_record")
        print("DONE " + json.dumps(
            {"elapsed_ms": round(1000 * (time.perf_counter() - t0), 2)}),
            flush=True)
        return

    # ---- resume: fresh process, same durable file ----
    recovered = [r.fp for r in ledger.recover()]
    before = ledger.get(call.fingerprint())
    d, res = disp.execute(call, tool_impl)
    after = ledger.get(call.fingerprint())
    print(json.dumps({
        "verdict": d.verdict.value, "dispatched": d.dispatched,
        "reason": d.reason, "recovered_unknown": len(recovered),
        "status_before": before.status.value if before else None,
        "status_after": after.status.value if after else None,
        "store": ledger.config}), flush=True)


if __name__ == "__main__":
    main()
