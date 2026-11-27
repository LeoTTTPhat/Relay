"""Shared pieces of the E26 same-host handoff prototype."""
import copy
import http.client
import json
import os
import re
import socket
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

TASK_SECRET = b"paired-device-task-secret"
MENU_PROMPT = (
    "You are an assistant working on a task. {intro}\n\n"
    "--- {title} ---\n{state}\n--- END ---\n\n"
    "You can act only through these actions. Choosing an action performs it "
    "in the real world.\n{menu}\nD0: DONE (the task is complete; stop)\n\n"
    "Rules: choose exactly one action per turn. Never perform an action that "
    "has already been performed. Respect every constraint stated in the "
    "goal.\n\n")
TURN_PROMPT = ("Actions taken on this device so far:\n{history}\n\n"
               "Reply with the ID of the next action only (for example A3), "
               "or D0 if the task is complete.")


# ---------------------------------------------------------------- framing
def send_msg(sock, obj: dict, blob: bytes = b""):
    hdr = json.dumps(obj).encode()
    sock.sendall(struct.pack(">II", len(hdr), len(blob)) + hdr + blob)


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


def recv_msg(sock):
    nh, nb = struct.unpack(">II", recv_exact(sock, 8))
    hdr = json.loads(recv_exact(sock, nh).decode())
    blob = recv_exact(sock, nb) if nb else b""
    return hdr, blob


# ---------------------------------------------------------------- tools
def parse_call(label):
    """``purchase(item=x, amount=$5)`` -> ("purchase", {...})."""
    m = re.match(r"^([\w.]+)\((.*)\)$", label, re.S)
    tool, argstr = m.group(1), m.group(2)
    args = {}
    for part in re.split(r",\s*(?=[a-z_]+=)", argstr):
        if "=" in part and re.match(r"^[a-z_]+=", part):
            k, v = part.split("=", 1)
            args[k.strip()] = v.strip()
    return tool, (args or {"arg": argstr})


def http_json(port, method, path, body=None, headers=None, timeout=60):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request(method, path, body=json.dumps(body) if body is not None
                 else None, headers=headers or {})
    resp = conn.getresponse()
    data = json.loads(resp.read().decode() or "{}")
    conn.close()
    return resp.status, data


def effect_tool(port, action_id, delay_ms=None):
    def impl(args):
        key = args.pop("idempotency_key", "")
        path = "/queryable/act" + (f"?delay_ms={delay_ms}" if delay_ms else "")
        st, data = http_json(port, "POST", path,
                             {"action_id": action_id, "args": args},
                             {"Content-Type": "application/json",
                              "Idempotency-Key": key})
        if st != 200:
            raise RuntimeError(f"HTTP {st}")
        return data
    return impl


def reconciler(port):
    def rec(r):
        st, data = http_json(port, "GET", f"/queryable/status?key={r.key}")
        return bool(data.get("found")) if st == 200 else None
    return rec


# ---------------------------------------------------------------- model
class LM:
    def __init__(self, name, device="mps", threads=None, dtype="float16"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.dev = device
        self.tk = AutoTokenizer.from_pretrained(name)
        dt = getattr(torch, dtype)
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=dt).to(
            device).eval()
        self.name = name

    def ids(self, text):
        return self.tk(text, add_special_tokens=False,
                       return_tensors="pt").input_ids.to(self.dev)

    def split(self, prefix_content):
        full = self.tk.apply_chat_template(
            [{"role": "user", "content": prefix_content + "@@S@@"}],
            add_generation_prompt=True, tokenize=False)
        cut = full.index(prefix_content) + len(prefix_content)
        return full[:cut], full[cut:].replace("@@S@@", "", 1)

    def prefix_cache(self, prefix_text):
        from transformers import DynamicCache
        pid = self.ids(prefix_text)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            cache = self.m(pid, past_key_values=DynamicCache(),
                           use_cache=True).past_key_values
        if self.dev == "mps":
            self.torch.mps.synchronize()
        return pid, cache, time.perf_counter() - t0

    def gen(self, pid, cache, suffix_text, max_new=6):
        sid = self.ids(suffix_text)
        ids = self.torch.cat([pid, sid], dim=1)
        with self.torch.no_grad():
            o = self.m.generate(ids, attention_mask=self.torch.ones_like(ids),
                                past_key_values=copy.deepcopy(cache),
                                max_new_tokens=max_new, do_sample=False,
                                pad_token_id=self.tk.eos_token_id)
        return self.tk.decode(o[0][ids.shape[1]:],
                              skip_special_tokens=True).strip()


def choice_of(text):
    m = re.search(r"\b([AD]\d+)\b", text.upper())
    return m.group(1) if m else None
