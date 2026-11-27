"""E27 (generation step): a HELD-OUT LLM-written paraphrase family.

The gemma3:27b family (E25) exposed the revision parser's dependence on
template cue words and was then used to *develop* the conservative fallback
(``state/schema.py: conservative=True``), so it is a development set.  This
script writes a fresh, held-out family with a different model family
(``mistral-small3.1:24b``), different registers and different seeds, using the
same validation rules.  It was started before the fallback was frozen and its
outputs are not scored until the fallback code is frozen (AUDIT H.3).

Everything below the next paragraph is inherited from the E25 generator.

E25 (generation step): an LLM-written paraphrase family for the
out-of-schema corpus.

Why this exists.  The out-of-schema corpus (``synth/oos.py``) exposed a real
schema bug, the bug was fixed, and the fix was then evaluated on the *same*
template generator.  Its revision, correction and open-question sentences are
written with a handful of fixed cue phrases ("change of plan", "CORRECTION",
"supersedes", "OPEN QUESTION", "Awaiting reply"), and the revision parser in
``state/schema.py`` recognises explicit correction language.  A reviewer
correctly pointed out that this is corrective validation, not evidence that
the parser generalises to other authors.

We cannot recruit independent human authors before the deadline.  The closest
honest proxy we can build is a *held-out paraphrase family written by a
different generator*: a local LLM (``gemma3:27b`` via ollama, a model family
used nowhere else in this project) rewrites each out-of-schema sentence in one
of three registers.  Two conditions are generated:

  ``natural``   rewrite freely, keep the meaning and the required tokens;
  ``disjoint``  additionally avoid every cue word the template generator uses,
                i.e. what an author who never saw our template would plausibly
                write.

The parser is NOT changed after seeing these outputs; this script only writes
``results/e25_paraphrases.json`` and never scores anything.  Every output is
validated mechanically (required tokens present; for ``disjoint`` no banned cue
word present); failures are retried with a new seed up to ``MAX_TRIES`` times
and otherwise recorded as failures, never replaced by the template sentence.

These are LLM-generated paraphrases, not independent human authors, and the
paper labels them that way.
"""
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from synth.oos import generate_oos                                  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RES, "e27_heldout_paraphrases.json")
MODEL = os.environ.get("RELAY_E27_MODEL", "mistral-small3.1:24b")
URL = "http://localhost:11434/api/generate"
MAX_TRIES = 4

STYLES = [
    "a voice-assistant transcript of what the user said out loud",
    "a text message the user sends to a friend who is helping with the task",
    "one bullet point in hastily written meeting notes",
]

#: Cue words/phrases used by the template generator (oos.py) for the
#: out-of-schema events.  ``disjoint`` paraphrases must avoid all of them.
BANNED = ["update from", "change of plan", "correction", "corrected",
          "supersede", "supersedes", "no longer applies", "ignore the earlier",
          "was wrong", "stale", "open question", "awaiting reply",
          "awaiting a reply", "awaiting confirmation", "has not said",
          "has not confirmed", "do not purchase until", "earlier"]


def _ollama(prompt, seed, temperature=0.8, num_predict=120):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": temperature, "seed": seed,
                                   "num_predict": num_predict}}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["response"]


def _clean(s):
    s = s.strip().strip('"').strip("'").strip()
    s = re.sub(r"^\s*(here is|here's|rewritten|paraphrase)[^:\n]*:\s*", "",
               s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    return s.strip().strip('"').strip()


def _valid(text, required, condition):
    low = text.lower()
    missing = [t for t in required if t.lower() not in low]
    banned = []
    if condition == "disjoint":
        banned = [b for b in BANNED
                  if re.search(r"\b" + re.escape(b) + r"\b", low)]
    return (not missing and not banned), missing, banned


PROMPT = (
    "Rewrite the following message from an assistant's task log as {style}. "
    "Keep exactly the same meaning and facts. Write one or two sentences, "
    "plain text, no quotation marks, no preamble.\n"
    "You MUST keep these exact strings verbatim somewhere in your rewrite: "
    "{required}.\n{extra}"
    "Message: {text}\nRewrite:")


def _paraphrase(text, required, condition, style, seed0):
    extra = ""
    if condition == "disjoint":
        extra = ("Do NOT use any of these words or phrases: "
                 + ", ".join(f"'{b}'" for b in BANNED) + ".\n")
    tries = []
    for t in range(MAX_TRIES):
        out = _clean(_ollama(PROMPT.format(
            style=style, required=", ".join(f"'{r}'" for r in required),
            extra=extra, text=text), seed=seed0 + 1000 * t))
        ok, miss, ban = _valid(out, required, condition)
        tries.append({"text": out, "missing": miss, "banned": ban})
        if ok:
            return out, tries
    return None, tries


def targets(e):
    """The out-of-schema sentences of one episode and their required tokens.

    Required tokens are the named entities and numbers the *template* sentence
    already contains, never the probe keys it lacks, so a paraphrase cannot be
    easier to score than the original.
    """
    o, out = e.oos, []
    for s in e.steps:
        c = s.content
        if c.startswith("UPDATE FROM"):
            m = re.search(r"destination is now (.+?), not (.+?), and the total "
                          r"must stay under \$(\d+)", c)
            out.append({"slot": "goal_revision", "step": s.idx, "text": c,
                        "replace": c,
                        "required": [m.group(1), m.group(2), m.group(3)]})
        elif c.startswith("CORRECTION:"):
            req = re.findall(r"\d{1,2}:00", c)
            out.append({"slot": "constraint_correction", "step": s.idx,
                        "text": c, "replace": c,
                        "required": sorted(set(req))})
        elif c.startswith("OPEN QUESTION:"):
            if "ship to the" in c:
                m = re.search(r"OPEN QUESTION: (\w+) has not said whether the "
                              r"(.+?) should ship to the (.+?) hotel", c)
                req = [m.group(1), m.group(2), m.group(3), "hotel", "home"]
            else:
                m = re.search(r"OPEN QUESTION: (\w+) has not confirmed whether "
                              r"the (.+?) should hold (\d+) overnight", c)
                req = [m.group(1), m.group(2), m.group(3), "overnight"]
            out.append({"slot": "open_question", "step": s.idx, "text": c,
                        "replace": c, "required": req})
        elif c.startswith("Remaining: buy the"):
            m = re.search(r"Remaining: buy the (.+?) once", c)
            out.append({"slot": "remaining", "step": s.idx, "text": c,
                        "replace": c,
                        "required": [m.group(1), "shipping address"]})
        elif " Supersedes the earlier $" in c:
            note = c[c.index("Supersedes the earlier $"):]
            m = re.search(r"earlier \$(\d+) quote, which was for (.+?)\.", note)
            out.append({"slot": "value_correction", "step": s.idx, "text": note,
                        "replace": note,
                        "required": ["$" + m.group(1), m.group(2)]})
        elif " Earlier reading of " in c and c.rstrip().endswith("was stale."):
            note = c[c.index("Earlier reading of "):]
            m = re.search(r"Earlier reading of (\d+) was stale", note)
            out.append({"slot": "value_correction", "step": s.idx, "text": note,
                        "replace": note, "required": [m.group(1)]})
    return out


def main():
    eps = generate_oos(120, seed=7)
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res = prev.get("episodes", {})
    t0 = time.time()
    n_calls = 0
    for i, e in enumerate(eps):
        style = STYLES[i % len(STYLES)]
        for cond in ("natural", "disjoint"):
            key = f"{e.eid}|{cond}"
            if key in res:
                continue
            items = []
            for j, tg in enumerate(targets(e)):
                seed = 100003 + 131 * i + 17 * j + (0 if cond == "natural"
                                                     else 11)
                para, tries = _paraphrase(tg["text"], tg["required"], cond,
                                          style, seed)
                n_calls += len(tries)
                items.append({**tg, "paraphrase": para, "tries": tries})
            res[key] = {"style": style, "condition": cond, "items": items}
        if i % 10 == 9:
            json.dump({"model": MODEL, "styles": STYLES, "banned": BANNED,
                       "episodes": res}, open(OUT, "w"), indent=1)
            print(f"{i+1}/120 episodes, {n_calls} calls, "
                  f"{time.time()-t0:.0f}s", flush=True)
    fails = {c: sum(1 for k, v in res.items() if v["condition"] == c
                    for it in v["items"] if it["paraphrase"] is None)
             for c in ("natural", "disjoint")}
    n_items = {c: sum(len(v["items"]) for v in res.values()
                      if v["condition"] == c) for c in ("natural", "disjoint")}
    json.dump({"model": MODEL, "styles": STYLES, "banned": BANNED,
               "n_items": n_items, "n_failed": fails,
               "generation_s": round(time.time() - t0, 1),
               "episodes": res}, open(OUT, "w"), indent=1)
    print("done", n_items, "failed", fails, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
