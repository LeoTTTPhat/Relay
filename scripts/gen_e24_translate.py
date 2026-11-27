"""E24 (translation step): machine-translate a subset of real trajectories.

A reviewer asked how the deterministic extractor and the value-density
allocator behave on non-English traces, where anchor and cue patterns differ.
No multilingual agent-trace corpus with this structure is available to us, so
we build the closest honest proxy: the same Who&When trajectories, machine-
translated from English into German, Spanish and French with the local
Helsinki-NLP opus-mt models.  These are **machine-translated traces**, not
native multilingual agent logs, and every result derived from them is labelled
that way.

What is translated.  The task statement and every step, line by line, split
into sentences.  Kept verbatim, because a multilingual deployment would not
translate them either: fenced code blocks, URLs, lines with fewer than two
alphabetic words (data rows, numbers, identifiers), and sentences longer than
``MAX_CHARS`` characters (logged; the rate is reported).  Numbers inside
translated prose are left to the translation model, which may localise them
(e.g. a decimal comma) -- that is part of what non-English text does to a
string-matching extractor.  The ground-truth answer is not translated: it is
task metadata, not trace text.

Translations are cached per language in ``results/e24_mt/`` so the scoring
step (``run_e24_multilingual.py``) is deterministic and does not need the
models.  Greedy decoding (num_beams=1).
"""
import gzip
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from traces.parser import load_trajectories                          # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
MTDIR = os.path.join(RES, "e24_mt")
LANGS = os.environ.get("RELAY_E24_LANGS", "de,es,fr").split(",")
STRIDE = int(os.environ.get("RELAY_E24_STRIDE", "4"))
MAX_CHARS = 600
CODE_RE = re.compile(r"(```.*?```)", re.S)
URL_RE = re.compile(r"(https?://\S+)")
SENT_RE = re.compile(r"(?<=[.!?])\s+")


def subset():
    ts = load_trajectories()
    return ts[::STRIDE]


#: Status lines emitted by the code-execution *tool* rather than written by an
#: agent.  Scoring variant ``tool_status_verbatim`` keeps them untranslated,
#: because a non-English deployment's executor still prints them in English.
#: This variant was added after inspecting the first scoring run (the
#: translation models had rendered ``exitcode:`` as e.g. ``codigo de salida``);
#: both variants are reported.
TOOL_STATUS_RE = re.compile(r"^\s*(exitcode:|code output:)", re.I)


def segments(text, keep_tool_status=False):
    """Yield (kind, piece): kind 'mt' pieces are translated, 'keep' are not."""
    for block in CODE_RE.split(text):
        if block.startswith("```"):
            yield ("keep", block)
            continue
        lines = block.split("\n")
        for li, line in enumerate(lines):
            if len(re.findall(r"[A-Za-z]{3,}", line)) < 2 or (
                    keep_tool_status and TOOL_STATUS_RE.match(line)):
                yield ("keep", line)
            else:
                for pi, part in enumerate(URL_RE.split(line)):
                    if URL_RE.fullmatch(part) or \
                            len(re.findall(r"[A-Za-z]{3,}", part)) < 2:
                        yield ("keep", part)
                        continue
                    lead = part[:len(part) - len(part.lstrip())]
                    trail = part[len(part.rstrip()):]
                    sents = SENT_RE.split(part.strip())
                    if lead:
                        yield ("keep", lead)
                    for si, s in enumerate(sents):
                        if len(s) > MAX_CHARS:
                            yield ("long", s)
                        else:
                            yield ("mt", s)
                        if si < len(sents) - 1:
                            yield ("keep", " ")
                    if trail:
                        yield ("keep", trail)
            if li < len(lines) - 1:
                yield ("keep", "\n")


def translate_text(text, table, keep_tool_status=False):
    out = []
    for kind, piece in segments(text, keep_tool_status):
        out.append(table.get(piece, piece) if kind == "mt" else piece)
    return "".join(out)


def main():
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    torch.set_num_threads(8)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.makedirs(MTDIR, exist_ok=True)
    trajs = subset()
    todo, n_long, n_mt = set(), 0, 0
    for t in trajs:
        for text in [t.question] + [s.content for s in t.steps]:
            for kind, piece in segments(text):
                if kind == "mt":
                    todo.add(piece)
                    n_mt += 1
                elif kind == "long":
                    n_long += 1
    todo = sorted(todo, key=len)
    dev = "mps" if (torch.backends.mps.is_available()
                    and os.environ.get("RELAY_E24_MPS")) else "cpu"
    meta = {"n_traces": len(trajs), "stride": STRIDE,
            "tids": [t.tid for t in trajs], "n_unique_sentences": len(todo),
            "n_sentence_occurrences": n_mt, "n_long_kept_verbatim": n_long,
            "max_chars": MAX_CHARS, "device": dev, "langs": {}}
    for lang in LANGS:
        path = os.path.join(MTDIR, f"table_{lang}.json.gz")
        table = {}
        if os.path.exists(path):
            table = json.load(gzip.open(path, "rt"))
        name = f"Helsinki-NLP/opus-mt-en-{lang}"
        tok = MarianTokenizer.from_pretrained(name)
        model = MarianMTModel.from_pretrained(name).to(dev).eval()
        t0 = time.time()
        pending = [s for s in todo if s not in table]
        B = 24
        for i in range(0, len(pending), B):
            batch = pending[i:i + B]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=400).to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, num_beams=1, max_new_tokens=512)
            for src, out in zip(batch, tok.batch_decode(
                    gen, skip_special_tokens=True)):
                table[src] = out
            if (i // B) % 20 == 19:
                json.dump(table, gzip.open(path, "wt"))
                print(f"[{lang}] {i + len(batch)}/{len(pending)} "
                      f"{time.time() - t0:.0f}s", flush=True)
        json.dump(table, gzip.open(path, "wt"))
        meta["langs"][lang] = {"model": name, "seconds": round(time.time() - t0, 1),
                               "n_translated": len(table)}
        print(f"[{lang}] done {len(table)} in {time.time() - t0:.0f}s",
              flush=True)
        del model
    json.dump(meta, open(os.path.join(MTDIR, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
