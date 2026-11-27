"""Token accounting.

The evaluation reports two token counts and is explicit about which is used
where:

* ``exact_tokens`` -- a real BPE count from the Qwen2.5 tokenizer (the
  tokenizer of a model family that plausibly runs on the phone/laptop
  classes we model, and whose 0.5B/1.5B members are the wearable- and
  speaker-class candidates).  Every *cost* number in the paper (prefill
  time, target-window occupancy, truncation decisions) uses this.
* ``est_tokens`` -- the cheap ``ceil(chars / 3.5)`` heuristic used *inside*
  the distiller's fill loop, where thousands of increments per block make
  exact tokenization wasteful.  3.5 is deliberately below the measured
  corpus mean (3.63 chars/token) so the estimate is conservative: the fill
  loop can only under-fill, never overrun the budget.  Every block is then
  verified and trimmed against ``exact_tokens`` before it is shipped.

If the tokenizer is unavailable (no local model cache) the module falls
back to the heuristic everywhere and sets ``EXACT_AVAILABLE = False``; all
experiment drivers record which mode produced their numbers.
"""
import functools
import os

_TOKENIZER = None
EXACT_AVAILABLE = False
_BUNDLED_TOKENIZER = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "tokenizer"))
TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Measured corpus mean is 3.63 chars/token on raw Who&When prefixes; the fill
# loop uses a conservative 3.5 so it under-fills rather than overruns.
CHARS_PER_TOKEN_FALLBACK = 3.5


def _load():
    global _TOKENIZER, EXACT_AVAILABLE
    if _TOKENIZER is not None:
        return _TOKENIZER
    if os.environ.get("RELAY_NO_TOKENIZER"):
        return None
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(
            _BUNDLED_TOKENIZER if os.path.isdir(_BUNDLED_TOKENIZER)
            else TOKENIZER_NAME)
        EXACT_AVAILABLE = True
    except Exception:                                   # pragma: no cover
        _TOKENIZER = None
    return _TOKENIZER


def est_tokens(text: str) -> int:
    """Fast conservative estimate used inside budget fill loops."""
    if not text:
        return 1
    return max(1, int(len(text) / CHARS_PER_TOKEN_FALLBACK) + 1)


def exact_tokens(text: str) -> int:
    """Real BPE token count (Qwen2.5); falls back to the estimate."""
    tk = _load()
    if tk is None:
        return est_tokens(text)
    if not text:
        return 0
    return len(tk(text, add_special_tokens=False).input_ids)


@functools.lru_cache(maxsize=200_000)
def _exact_cached(text: str) -> int:
    return exact_tokens(text)


def exact_tokens_cached(text: str) -> int:
    if len(text) > 200_000:          # do not cache very large prefixes
        return exact_tokens(text)
    return _exact_cached(text)


def cap_suffix(text: str, max_tokens: int):
    """Longest suffix of ``text`` holding at most ``max_tokens`` BPE tokens.

    Returns ``(suffix, total_tokens, truncated)``.  Done with a single
    tokenisation using character offsets rather than a binary search over
    substrings, which would re-tokenise a 300 KB prefix a dozen times per
    handoff.
    """
    tk = _load()
    if tk is None:
        total = est_tokens(text)
        if total <= max_tokens:
            return text, total, False
        keep = int(max_tokens * CHARS_PER_TOKEN_FALLBACK)
        return text[-keep:], total, True
    enc = tk(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc.input_ids
    total = len(ids)
    if total <= max_tokens:
        return text, total, False
    start_char = enc.offset_mapping[total - max_tokens][0]
    return text[start_char:], total, True


def available() -> bool:
    _load()
    return EXACT_AVAILABLE
