"""E12: is the action fingerprint an identity function, or a hash of a rendering?

E8 measured the fingerprint on the two error directions that arise *naturally*
in the corpora.  It did not ask the adversarial question: given an action that
has already been performed and recorded, how hard is it for a semantically
identical retry to be rendered differently enough to miss its own ledger entry?
Every such miss is a double execution, and a text-derived identity function has
many ways to be nudged.

Two studies:

  **E12a divergence under semantics-preserving perturbation.**  Each recorded
  action is re-rendered by a transformation that provably does not change what
  the action *does* -- argument order, argument aliasing, numeric and currency
  formatting, URL query order, quoting style, whitespace and case, a changed
  timestamp or run id, an injected decoy, a paraphrased explanation around the
  call, a command split across lines.  A fingerprint that changes under any of
  these has failed to be an identity function.  Rates are reported per family,
  because the aggregate hides which constructions are actually dangerous.

  **E12b canonicalisation collision.**  R4's distinction between a
  *cryptographic* collision (SHA-1 on distinct inputs -- not the relevant
  property) and a *canonicalisation* collision (two genuinely different actions
  mapped to the same canonical form before hashing -- the property that
  matters).  We construct minimal action pairs that differ in exactly one
  semantic argument and count how often each identity function erases that
  difference.

The comparison is three-way: ``v1`` head hash, ``v2`` text action-signature
(what the trace corpora allow), and ``v3`` typed-call identity (what a deployed
runtime can compute, since it sees the call before dispatching it).  The point
of the experiment is to establish where identity should be computed, not to
declare the text signature adequate.
"""
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, RES                                   # noqa: E402
from state.fingerprint import fp_action, fp_head, fp_typed         # noqa: E402

OUT = os.path.join(RES, "e12_adversarial.json")


# ---------------------------------------------------------------------------
# Typed-call perturbations (apply to (tool, args) pairs)
# ---------------------------------------------------------------------------
def _p_reorder(tool, args, rng):
    items = list(args.items())
    rng.shuffle(items)
    return tool, dict(items)


_ALIAS_BACK = {"to": "recipient", "text": "body", "amount": "price",
               "item": "sku", "file": "path", "device": "unit",
               "quantity": "qty"}


def _p_alias(tool, args, rng):
    return tool, {_ALIAS_BACK.get(k, k): v for k, v in args.items()}


def _p_numfmt(tool, args, rng):
    out = {}
    for k, v in args.items():
        s = str(v)
        m = re.fullmatch(r"\$?(\d+)(?:\.(\d+))?", s)
        if m:
            out[k] = f"${float(s.lstrip('$')):.2f}"
        else:
            out[k] = v
    return tool, out


def _p_ws_case(tool, args, rng):
    return tool, {k: (f"  {v}  ".upper() if isinstance(v, str) else v)
                  for k, v in args.items()}


def _p_volatile(tool, args, rng):
    out = dict(args)
    out["timestamp"] = f"2026-0{rng.randrange(1, 9)}-1{rng.randrange(0, 9)}T09:00"
    out["request_id"] = f"req-{rng.randrange(10**6)}"
    return tool, out


def _p_urlorder(tool, args, rng):
    out = {}
    for k, v in args.items():
        s = str(v)
        if s.lower().startswith("http") and "?" in s:
            base, q = s.split("?", 1)
            ps = q.split("&")
            rng.shuffle(ps)
            out[k] = base + "?" + "&".join(ps)
        else:
            out[k] = v
    return tool, out


TYPED_PERTURBATIONS = {
    "arg_reorder": _p_reorder,
    "arg_alias": _p_alias,
    "numeric_format": _p_numfmt,
    "whitespace_case": _p_ws_case,
    "volatile_field": _p_volatile,
    "url_query_order": _p_urlorder,
}


# ---------------------------------------------------------------------------
# Text perturbations (apply to a step's rendered content)
# ---------------------------------------------------------------------------
def _t_reorder_kv(text, rng):
    """Reorder `k=v` arguments inside the first parenthesised call."""
    def sub(m):
        parts = [p.strip() for p in m.group(1).split(",")]
        rng.shuffle(parts)
        return "(" + ", ".join(parts) + ")"
    return re.sub(r"\(([^()]{5,200})\)", sub, text, count=1)


def _t_alias_kv(text, rng):
    for a, b in (("item=", "sku="), ("to=", "recipient="), ("text=", "body="),
                 ("amount=", "price="), ("store=", "vendor=")):
        text = text.replace(a, b)
    return text


def _t_whitespace(text, rng):
    return re.sub(r" ", "  ", text).replace("\n", " \n ")


def _t_quotes(text, rng):
    return text.replace("'", '"')


def _t_volatile(text, rng):
    text = re.sub(r"exitcode:\s*\d+", f"exitcode: {rng.randrange(2, 9)}", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "2026-07-01", text)
    return text + f"\n[run {rng.randrange(10**8)}] elapsed {rng.randrange(99)}s"


def _t_decoy(text, rng):
    """An attacker (or a chatty tool) appends unrelated evidence.

    This is the perturbation the text signature is most exposed to: its
    argument evidence is *extracted* from the step, so anything appended to
    the step is a candidate argument.
    """
    return text + ("\nSee also https://example.org/help?ref=42 and "
                   "/tmp/cache/note.txt (retry_hint=true) \"click here\"")


_PARA = [(r"\bI typed\b", "I entered"), (r"\bI clicked\b", "I selected"),
         (r"\bHere is a screenshot of\b", "Below is a capture of"),
         (r"\bEXECUTED\b", "PERFORMED"), (r"\bstatus=done\b", "status: complete"),
         (r"\bReserved\b", "Held"), (r"\bNotified owner\b", "Owner notified")]


def _t_paraphrase(text, rng):
    for pat, rep in _PARA:
        text = re.sub(pat, rep, text)
    return text


def _t_split_cmd(text, rng):
    """Split a command line across a continuation, as a shell user would."""
    return re.sub(r"(\S+) (--?\w+)", r"\1 \\\n  \2", text, count=2)


TEXT_PERTURBATIONS = {
    "arg_reorder": _t_reorder_kv,
    "arg_alias": _t_alias_kv,
    "whitespace_case": _t_whitespace,
    "quote_style": _t_quotes,
    "volatile_field": _t_volatile,
    "decoy_injection": _t_decoy,
    "explanation_paraphrase": _t_paraphrase,
    "command_split": _t_split_cmd,
}


# ---------------------------------------------------------------------------
def _parse_args(argstr):
    """`item=x, store=y` -> dict.  Synthetic actions are typed by construction."""
    out = {}
    for part in re.split(r",\s*(?=[a-z_]+=)", argstr):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out or {"arg": argstr}


def e12a_divergence(eps, trajs, seed=0):
    """Divergence rate per perturbation family, for v1/v2/v3."""
    rng = random.Random(seed)

    # ---- typed actions from the synthetic episodes ----
    typed = []
    for e in eps:
        for se in e.states[-1].side_effects:
            typed.append((se["action"], _parse_args(se["args"])))
    typed_res = defaultdict(lambda: {"n": 0, "diverged": 0})
    for tool, args in typed:
        base = fp_typed(tool, args)
        for name, fn in TYPED_PERTURBATIONS.items():
            t2, a2 = fn(tool, dict(args), rng)
            d = typed_res[name]
            d["n"] += 1
            d["diverged"] += int(fp_typed(t2, a2) != base)

    # ---- text actions: synthetic (rendered) and real (transcript) ----
    text_items = []
    for e in eps:
        for s in e.steps:
            if s.side_effect_like:
                text_items.append(("synthetic", s.agent, s.content))
    real_items = []
    for t in trajs:
        for s in t.steps:
            if s.side_effect_like:
                real_items.append(("real", s.agent, s.content))
    rng.shuffle(real_items)
    text_items += real_items[:600]          # bounded, deterministic subsample

    text_res = {v: defaultdict(lambda: {"n": 0, "diverged": 0})
                for v in ("v1-head", "v2-action")}
    for corpus, agent, content in text_items:
        b1, b2 = fp_head(agent, content), fp_action(agent, content)
        for name, fn in TEXT_PERTURBATIONS.items():
            c2 = fn(content, rng)
            for ver, base, fp in (("v1-head", b1, fp_head),
                                  ("v2-action", b2, fp_action)):
                d = text_res[ver][name]
                d["n"] += 1
                d["diverged"] += int(fp(agent, c2) != base)

    def _rate(d):
        return {k: {"n": v["n"], "diverged": v["diverged"],
                    "rate": round(v["diverged"] / v["n"], 4) if v["n"] else 0.0}
                for k, v in sorted(d.items())}

    out = {
        "v3-typed": _rate(typed_res),
        "v1-head": _rate(text_res["v1-head"]),
        "v2-action": _rate(text_res["v2-action"]),
        "n_typed_actions": len(typed),
        "n_text_actions": len(text_items),
    }
    for ver in ("v1-head", "v2-action", "v3-typed"):
        tot = sum(v["n"] for v in out[ver].values())
        div = sum(v["diverged"] for v in out[ver].values())
        out[ver + "_overall"] = round(div / tot, 4) if tot else 0.0
    return out


def e12b_canon_collision(eps, n_pairs=4000, seed=1):
    """Do two actions differing in exactly one semantic argument collide?

    This is *canonicalisation* collision, not cryptographic collision: the
    question is whether the identity function erases a difference that matters
    before it ever reaches the hash.
    """
    rng = random.Random(seed)
    acts = []
    for e in eps:
        for se in e.states[-1].side_effects:
            acts.append((se["action"], _parse_args(se["args"])))
    res = {v: {"n": 0, "collided": 0} for v in ("v2-action", "v3-typed")}
    for _ in range(n_pairs):
        tool, args = acts[rng.randrange(len(acts))]
        if not args:
            continue
        k = rng.choice(list(args))
        a2 = dict(args)
        a2[k] = str(args[k]) + rng.choice(["_alt", " 2", "-b", "X"])
        # typed identity
        res["v3-typed"]["n"] += 1
        res["v3-typed"]["collided"] += int(fp_typed(tool, args) ==
                                           fp_typed(tool, a2))
        # text identity, on the same pair rendered as the generator renders it
        def render(a):
            return (f"EXECUTED {tool}(" +
                    ", ".join(f"{kk}={vv}" for kk, vv in a.items()) +
                    ") [idempotency-key idem-000] status=done.")
        res["v2-action"]["n"] += 1
        res["v2-action"]["collided"] += int(
            fp_action("executor", render(args)) ==
            fp_action("executor", render(a2)))
    for v in res.values():
        v["rate"] = round(v["collided"] / v["n"], 4) if v["n"] else 0.0
    return res


def run():
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    out = {"e12a_divergence": e12a_divergence(eps, trajs),
           "e12b_canon_collision": e12b_canon_collision(eps)}
    json.dump(out, open(OUT, "w"), indent=1)
    a = out["e12a_divergence"]
    print("overall divergence: v1=%.3f v2=%.3f v3=%.3f"
          % (a["v1-head_overall"], a["v2-action_overall"], a["v3-typed_overall"]))
    for ver in ("v1-head", "v2-action", "v3-typed"):
        print(f"\n{ver}:")
        for k, v in a[ver].items():
            print(f"   {k:24s} {v['rate']:.3f}  ({v['diverged']}/{v['n']})")
    print("\ncanonicalisation collision:", json.dumps(out["e12b_canon_collision"]))
    print("->", OUT)


if __name__ == "__main__":
    run()
