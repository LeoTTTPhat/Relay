"""E20 endpoint: a local HTTP service standing in for a side-effectful API.

Runs as its own OS process, so killing an agent runtime never touches it.
Every executed effect is committed to its own SQLite file before the response
is sent; that log is the ground truth E20 scores against, and it is exposed
only on an inspection route the runtime under test never calls.

Three service contracts, chosen per request by URL prefix:

  /queryable/act   honours ``Idempotency-Key`` (at most one execution per key,
                   stored response replayed on retry) *and* answers
                   ``GET /queryable/status?key=`` ("did key k take effect?");
  /keyed/act       honours ``Idempotency-Key`` but has no status query;
  /plain/act       executes every request; no key, no query.

``?abort_on_disconnect=1`` makes the service check, after a short pre-effect
delay, whether the client is still connected and abandon the request if it is
not -- so a crash while the request is in flight has a genuinely undetermined
outcome.  Without it the service completes the effect regardless.
"""
import argparse
import json
import select
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

PRE_DELAY_S = 0.03      # request received -> effect applied
RESP_DELAY_S = 0.05     # effect applied   -> response sent

LOCK = threading.Lock()
DB = None


def _db(path):
    db = sqlite3.connect(path, isolation_level=None, check_same_thread=False,
                         timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS effects (seq INTEGER PRIMARY KEY AUTOINCREMENT,
        action_id TEXT, key TEXT, contract TEXT, t_ns INTEGER);
      CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT);
      CREATE INDEX IF NOT EXISTS eff_action ON effects(action_id);
    """)
    return db


def _client_gone(sock):
    """True if the peer closed the connection (readable with EOF)."""
    try:
        r, _, _ = select.select([sock], [], [], 0)
        if not r:
            return False
        import socket
        data = sock.recv(1, socket.MSG_PEEK)
        return data == b""
    except Exception:
        return True


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj, extra=None):
        body = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        u = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = u.path.strip("/").split("/")
        if parts == ["effects"]:
            with LOCK:
                n = DB.execute("SELECT COUNT(*) FROM effects WHERE action_id=?",
                               (q.get("action_id", ""),)).fetchone()[0]
            return self._send(200, {"count": n})
        if len(parts) == 2 and parts[1] == "status":
            if parts[0] != "queryable":
                return self._send(404, {"error": "no status API"})
            with LOCK:
                row = DB.execute("SELECT body FROM responses WHERE key=?",
                                 (q.get("key", ""),)).fetchone()
            return self._send(200, {"found": row is not None})
        return self._send(404, {"error": "unknown route"})

    def do_POST(self):
        u = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = u.path.strip("/").split("/")
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n).decode() or "{}")
        if len(parts) != 2 or parts[1] != "act" or \
                parts[0] not in ("queryable", "keyed", "plain"):
            return self._send(404, {"error": "unknown route"})
        contract = parts[0]
        key = self.headers.get("Idempotency-Key", "")
        if contract != "plain" and key:
            with LOCK:
                row = DB.execute("SELECT body FROM responses WHERE key=?",
                                 (key,)).fetchone()
            if row is not None:
                return self._send(200, json.loads(row[0]),
                                  {"Idempotent-Replayed": "true"})
        time.sleep(PRE_DELAY_S)
        if q.get("abort_on_disconnect") == "1" and \
                _client_gone(self.connection):
            return                              # abandoned: no effect
        body = {"ok": True, "action_id": req.get("action_id"),
                "executed_at_ns": time.time_ns()}
        with LOCK:
            if contract != "plain" and key:
                row = DB.execute("SELECT body FROM responses WHERE key=?",
                                 (key,)).fetchone()
                if row is not None:           # lost a same-key race
                    return self._send(200, json.loads(row[0]),
                                      {"Idempotent-Replayed": "true"})
            DB.execute("BEGIN IMMEDIATE")
            DB.execute("INSERT INTO effects(action_id, key, contract, t_ns) "
                       "VALUES (?, ?, ?, ?)",
                       (req.get("action_id"), key, contract, time.time_ns()))
            if contract != "plain" and key:
                DB.execute("INSERT INTO responses(key, body) VALUES (?, ?)",
                           (key, json.dumps(body)))
            DB.execute("COMMIT")
        # E26 only: ``?delay_ms=`` lengthens the response delay so a handoff
        # can be taken while this request is still in flight (E20 never
        # passes it, so its timing is unchanged).
        time.sleep(float(q.get("delay_ms", RESP_DELAY_S * 1000)) / 1000.0)
        return self._send(200, body)


def main():
    global DB
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args()
    DB = _db(a.db)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    srv.daemon_threads = True
    print(f"PORT {srv.server_address[1]}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
