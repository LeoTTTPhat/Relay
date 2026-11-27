"""Action fingerprints for the side-effect ledger.

The ledger's contract is exactly-once: before issuing a side-effectful
action the resumed agent fingerprints it and skips the action if the
fingerprint is already recorded.  A fingerprint function therefore has two
error directions, and a ledger design is only evaluated if both are
measured:

  * **divergence** (the same action fingerprints differently after
    resumption) -> the recorded action is not recognised -> *double
    execution*, the hazard the paper's E4 measures;
  * **collision** (two genuinely different actions share a fingerprint)
    -> the resumed agent believes a not-yet-performed action is done ->
    *false suppression*, a hazard the ledger itself introduces.

``fp_head`` (v1) is the head-hash used in the first version of this system:
``sha1(agent | first 200 normalised chars)``.  It is retained so E8 can
quantify what the redesign buys.

``fp_action`` (v2) hashes an *action signature* instead: the acting agent,
the side-effect verb class, and canonicalised argument evidence extracted
from the whole step (code-block bodies, shell command lines, URLs, file
paths, ``k=v`` pairs, quoted strings), with volatile spans (timestamps,
durations, exit codes, run ids, byte counts) removed.  Excluding volatile
output lowers divergence; using whole-step argument evidence instead of a
fixed-length head lowers collision.
"""
import hashlib
import re
from typing import List, Tuple

# --- verb classes: which kind of world-affecting action the step reports ---
VERB_CLASSES: List[Tuple[str, re.Pattern]] = [
    ("exec",     re.compile(r"exitcode:\s*\d+|\bexecution succeeded\b|"
                            r"\bexecuting (the )?(code|script)\b|"
                            r"\bcode output\b", re.I)),
    ("submit",   re.compile(r"\bsubmitted (the )?(form|query|request)\b|"
                            r"\bclicked (the )?['\"]?submit\b|"
                            r"\bI (typed|clicked|pressed|submitted)\b", re.I)),
    ("send",     re.compile(r"\b(sent|sending) (an? )?(email|message|request)\b|"
                            r"\bposted\b|\bmessage_send\b", re.I)),
    ("write",    re.compile(r"\bwrote (the )?(file|output|results?) to\b|"
                            r"\bsaved (the )?(file|output|results?)\b|"
                            r"\bfile (was |has been )?(written|saved|created)\b",
                            re.I)),
    ("download", re.compile(r"\bdownloaded\b", re.I)),
    ("book",     re.compile(r"\bbooked\b|\bbooking\b", re.I)),
    ("purchase", re.compile(r"\bpurchased\b|\bplaced (the |an? )?order\b|"
                            r"\bpurchase\b", re.I)),
    ("actuate",  re.compile(r"\bdevice_actuation\b|\bactuated\b", re.I)),
    ("reserve",  re.compile(r"\breserve_item\b|\breserved\b", re.I)),
]

CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.S)
SHELL_RE = re.compile(r"^\s*(?:\$|>>>|#)\s*(\S.*)$", re.M)
URL_RE = re.compile(r"https?://[^\s\)\]\"'>]+", re.I)
PATH_RE = re.compile(r"(?:[\w.\-]+/)+[\w.\-]+\.[A-Za-z0-9]{1,5}")
KV_RE = re.compile(r"\b([a-z_][a-z0-9_]{1,20})\s*=\s*([^,;)\n]{1,80})", re.I)
# Quoted *values*, not JSON keys: a key is followed by a colon, and every
# object in a transcript shares its keys.
QUOTED_RE = re.compile(r"[\"']([^\"'\n]{4,80})[\"'](?!\s*:)")
CALL_RE = re.compile(r"\b([A-Z_][A-Z0-9_]{2,})\s+([a-z_][a-z0-9_]*)\s*\(([^)]{0,200})\)")
# The *action clause*: the leading sentence of the step, which on agent
# transcripts states the action itself ("I clicked 'Credits'.", "EXECUTED
# purchase(item=..)"), before the observation the action produced.
# A sentence boundary is preferred over a line break, and at least 40
# characters must precede it: steps that open with a JSON brace ("Updated
# Ledger: {\n  \"is_request_satisfied\": {\n") would otherwise be summarised
# by their structural preamble, which every such step shares.
CLAUSE_SENT_RE = re.compile(r"^.{40,240}?[.!?](?:\s|$)", re.S)
CLAUSE_LINE_RE = re.compile(r"^.{40,240}?\n", re.S)

# volatile spans: differ between two executions of the *same* action and so
# must not enter the fingerprint
VOLATILE_RE = re.compile(
    r"exitcode:\s*\d+"                                   # exit status
    r"|\b\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}(:\d{2})?\b"    # ISO timestamps
    r"|\b\d{1,2}:\d{2}:\d{2}\b"                          # clock times w/ secs
    r"|\b\d+(\.\d+)?\s?(ms|s|sec|secs|seconds|minutes)\b"  # durations
    r"|\b(0x)?[0-9a-f]{12,}\b"                           # hex ids / hashes
    r"|\bidempotency-key\s+\S+"                          # our own key echo
    r"|\b\d+\s?(bytes|kb|mb|gb)\b",                      # sizes
    re.I)

WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return WS_RE.sub(" ", VOLATILE_RE.sub(" ", s.lower())).strip()


def verb_class(content: str) -> str:
    for name, rx in VERB_CLASSES:
        if rx.search(content):
            return name
    return "other"


def action_clause(content: str) -> str:
    """De-volatilised leading clause: what the step says it *did*."""
    body = content.strip()
    m = CLAUSE_SENT_RE.match(body) or CLAUSE_LINE_RE.match(body)
    head = m.group(0) if m else body[:240]
    return _norm(head)[:200]


def action_signature(content: str, max_args: int = 12) -> str:
    """Canonical argument evidence for the action reported by ``content``.

    The action clause and any explicitly quoted target (the clicked label,
    the typed string) are always included: on browser transcripts the step
    body is dominated by the *observation* the action produced, and two
    different clicks that land on the same page share every URL and path in
    that observation.  Fingerprinting the observation instead of the action
    is what makes head- and body-hashes collide.
    """
    ev: List[str] = ["act:" + action_clause(content)]
    for m in list(QUOTED_RE.finditer(content))[:4]:
        ev.append("q:" + _norm(m.group(1))[:120])
    for m in CODE_BLOCK_RE.finditer(content):
        ev.append("code:" + _norm(m.group(1))[:400])
    if not ev:
        for m in SHELL_RE.finditer(content):
            ev.append("cmd:" + _norm(m.group(1))[:200])
    for m in CALL_RE.finditer(content):
        ev.append("call:" + _norm(m.group(2) + "(" + m.group(3) + ")")[:200])
    for m in URL_RE.finditer(content):
        ev.append("url:" + m.group(0).lower().rstrip(".,);"))
    for m in PATH_RE.finditer(content):
        ev.append("path:" + m.group(0).lower())
    for m in KV_RE.finditer(content):
        ev.append("kv:" + _norm(m.group(1) + "=" + m.group(2))[:120])
    # de-duplicate, keep deterministic order
    seen, out = set(), []
    for e in ev:
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
        if len(out) >= max_args:
            break
    if len(out) <= 1:
        # only the clause survived: back it with the whole de-volatilised
        # body (not a fixed-length head -- that is what collides)
        out.append("body:" + _norm(content)[:1000])
    return "|".join(sorted(out))


def fp_action(agent: str, content: str, nbits: int = 16) -> str:
    """v2 fingerprint: agent + verb class + canonical argument evidence."""
    sig = f"{agent.lower()}|{verb_class(content)}|{action_signature(content)}"
    return hashlib.sha1(sig.encode()).hexdigest()[:nbits]


def fp_head(agent: str, content: str, nbits: int = 16) -> str:
    """v1 fingerprint (retained for the E8 comparison): agent + 200-char head."""
    head = WS_RE.sub(" ", content[:200].lower()).strip()
    return hashlib.sha1((agent + "|" + head).encode()).hexdigest()[:nbits]


# ---------------------------------------------------------------------------
# Typed action identity
# ---------------------------------------------------------------------------
# ``fp_action`` recovers identity from transcript *text* because that is all a
# post-hoc trace corpus offers.  A deployed runtime does not have that
# problem: it sees the typed tool call before it dispatches it, and identity
# should be computed there.  ``fp_typed`` is the canonical identity function
# for a typed call, and the dispatcher (``runtime/dispatcher.py``) uses it in
# preference to the text signature whenever a typed call is available.
#
# Canonicalisation is what makes it an identity function rather than a hash of
# an arbitrary rendering.  Two calls denote the same action iff they agree on
# tool, on the resolved alias of every argument name, and on the normalised
# value of every *semantic* argument.  Argument order, whitespace, numeric
# formatting, URL query order, quoting style and any argument declared
# volatile are all rendering, not identity, and are normalised away.

#: Argument-name aliases that denote the same parameter.  A real deployment
#: reads these from the tool schema; the table here covers the aliases our
#: corpora and adversarial suite exercise.
ARG_ALIASES = {
    "recipient": "to", "dest": "to", "destination": "to", "addressee": "to",
    "body": "text", "message": "text", "content": "text", "msg": "text",
    "amount_usd": "amount", "price": "amount", "total": "amount",
    "sku": "item", "product": "item", "item_id": "item",
    "path": "file", "filename": "file", "filepath": "file", "target": "file",
    "device_id": "device", "unit": "device",
    "qty": "quantity", "count": "quantity", "n": "quantity",
}

#: Arguments that vary between two executions of the *same* action and so
#: must not enter identity (the typed analogue of ``VOLATILE_RE``).
VOLATILE_ARGS = {
    "timestamp", "ts", "time", "date", "nonce", "request_id", "trace_id",
    "run_id", "attempt", "retry", "session", "client_id", "user_agent",
    "idempotency_key", "correlation_id", "sent_at", "elapsed", "duration_ms",
}

_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_TRAILING_ZEROS = re.compile(r"\.?0+$")


def canon_arg_name(name: str) -> str:
    n = re.sub(r"[^a-z0-9_]", "", str(name).strip().lower())
    return ARG_ALIASES.get(n, n)


def canon_arg_value(v) -> str:
    """Normalise a value to its semantic content.

    Numbers compare numerically (``5``, ``5.0``, ``$5.00`` are one value),
    URLs compare with sorted query parameters and no trailing slash, and text
    compares case- and whitespace-insensitively.  Everything else about how a
    value was rendered is discarded.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        s = f"{float(v):.6f}"
        return _TRAILING_ZEROS.sub("", s) or "0"
    if isinstance(v, (list, tuple)):
        # order-insensitive: a set of recipients is the same action however
        # the caller happened to order it
        return "[" + ",".join(sorted(canon_arg_value(x) for x in v)) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(
            f"{canon_arg_name(k)}={canon_arg_value(x)}"
            for k, x in sorted(v.items(), key=lambda kv: canon_arg_name(kv[0]))
        ) + "}"
    s = str(v).strip()
    money = re.fullmatch(r"[$€£]\s*(-?[\d,]+(?:\.\d+)?)", s)
    if money:
        s = money.group(1).replace(",", "")
    if _NUM_RE.fullmatch(s.replace(",", "")):
        f = float(s.replace(",", ""))
        return _TRAILING_ZEROS.sub("", f"{f:.6f}") or "0"
    if s.lower().startswith(("http://", "https://")):
        from urllib.parse import urlsplit, parse_qsl, urlunsplit, urlencode
        u = urlsplit(s.lower())
        q = urlencode(sorted(parse_qsl(u.query)))
        return urlunsplit((u.scheme, u.netloc, u.path.rstrip("/"), q, ""))
    return WS_RE.sub(" ", s.lower()).strip()


def typed_signature(tool: str, args: dict,
                    volatile: set = None,
                    schema_version: str = None) -> str:
    """Canonical signature of a typed call: tool + sorted semantic args.

    ``schema_version`` is the version of the *tool's* API/schema as declared
    by the tool registry (e.g. ``payments.create@2024-06``).  When given, it
    is part of identity: the same argument dict sent to a semantically changed
    tool version is a different action, and must not be suppressed as a
    duplicate of the old one.  ``None`` reproduces the unversioned signature
    used by every experiment before E21, byte for byte.
    """
    vol = VOLATILE_ARGS if volatile is None else (VOLATILE_ARGS | set(volatile))
    items = []
    for k, v in (args or {}).items():
        name = canon_arg_name(k)
        if name in vol:
            continue
        items.append(f"{name}={canon_arg_value(v)}")
    t = WS_RE.sub('', str(tool).lower())
    if schema_version:
        t += "@" + WS_RE.sub('', str(schema_version).lower())
    return f"{t}({','.join(sorted(items))})"


def fp_typed(tool: str, args: dict, nbits: int = 16,
             volatile: set = None, schema_version: str = None) -> str:
    """v3 fingerprint: identity computed from the typed call, not from text."""
    return hashlib.sha1(typed_signature(
        tool, args, volatile, schema_version).encode()).hexdigest()[:nbits]


def fp_typed_family(tool: str, args: dict, nbits: int = 16,
                    volatile: set = None) -> str:
    """Version-free identity of a typed call.

    Used only as a *secondary* index: a proposed call whose versioned identity
    is unrecorded but whose family matches a record written under another
    schema version is neither a known duplicate nor a known new action, and
    the dispatcher routes it to reconciliation instead of guessing.
    """
    return fp_typed(tool, args, nbits, volatile, None)
