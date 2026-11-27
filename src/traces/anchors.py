"""Anchor extraction and future-relevance recall -- the paper's headline
fidelity metric.

For a handoff at step k of a real trajectory, the *future-relevant anchor set*
A(T,k) is the set of anchors that (i) originate in the prefix (steps <= k,
question included) and (ii) are referenced again, verbatim after
normalization, in the real suffix (steps > k). Fidelity of a handoff policy
is the fraction of A(T,k) recoverable (normalized containment with word
boundaries) from the state block the policy actually shipped.

Anchor types:
  url     -- http(s) URLs (normalized: scheme/www stripped, trailing punct cut)
  file    -- file names with a known extension
  number  -- numeric literals with >= 2 significant digits (comma/decimal
             forms canonicalized); trivial small integers excluded
  entity  -- capitalized n-grams (1..4 words), stop-listed, mid-sentence
             evidence required for unigrams
  quoted  -- single- or double-quoted spans of 3..60 chars
  answer  -- the ground-truth answer string when it originates in the prefix

Precision safeguards (developed over two audit iterations; see
scripts/audit_extractor.py, results/anchor_audit_iter1.csv and
results/anchor_audit.csv):
  * cross-trace document frequency filter: anchors occurring in > DF_MAX of
    all 184 trajectories are scaffold vocabulary (agent names, "FINAL ANSWER",
    "exitcode 0", bing.com boilerplate) and are dropped;
  * a hand-maintained stoplist of framework/scaffold terms plus a common-
    English-word list: unigram entities and single-word quoted spans must not
    be common words; multi-word entities/quotes need >= 1 non-common word
    (iteration 1 showed generic unigrams were the largest false-positive
    family at 0.42 entity precision);
  * scaffold-context suppression: an anchor is kept only if at least one of
    its occurrences lies outside boilerplate contexts (og:/twitter: metadata,
    site-packages tracebacks, bing search chrome, orchestrator-ledger JSON);
  * numbers: >= 2 digits, no leading zero, not adjacent to '-' (kills UUID /
    phone-number / ISO-date fragments), stop-listed round values excluded;
  * quoted spans: no code identifiers (snake_case), no template fragments
    ('{'/'}'), no metadata keys (namespace:key shapes);
  * entity sub-span dedup: an entity that is a substring of another extracted
    entity is dropped (keeps 'bbc earth youtube', drops 'earth youtube');
  * unigram entities must occur at least once in a non-sentence-initial
    position and be >= 4 chars.

Everything is deterministic; no LLM anywhere.
"""
import re
import os
import json
from functools import lru_cache
from collections import Counter
from typing import Dict, List, Set, Tuple

# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def norm_text(text: str) -> str:
    """Lowercase, collapse whitespace; keep punctuation (word-boundary
    matching handles the rest)."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def norm_anchor(a: str) -> str:
    a = a.strip().strip(".,;:!?()[]{}<>\"'`")
    a = re.sub(r"\s+", " ", a.lower())
    return a


def norm_url(u: str) -> str:
    u = u.strip().rstrip(".,;:!?)\"'")
    u = re.sub(r"^https?://", "", u, flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I)
    return u.rstrip("/").lower()


def norm_number(n: str) -> str:
    n = n.replace(",", "").rstrip(".")
    # canonical float-ish form: strip trailing zeros after decimal point
    if "." in n:
        n = n.rstrip("0").rstrip(".")
    return n


# --------------------------------------------------------------------------
# Extraction regexes
# --------------------------------------------------------------------------
URL_RE = re.compile(r"https?://[^\s<>\"')\]}]+", re.I)
FILE_RE = re.compile(
    r"\b[\w][\w.\-]{0,60}\.(?:csv|xlsx?|pdf|txt|json|py|zip|docx?|pptx?|png|"
    r"jpe?g|mp3|mp4|wav|html?|md|jsonld|tsv|xml)\b", re.I)
NUM_RE = re.compile(
    r"(?<![\w.\-])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{2,})(?![\w.\-])")
CAP_SEQ_RE = re.compile(
    r"\b([A-Z][a-zA-Z'&\-]+(?:\s+(?:of|the|de|for|and|&)?\s*[A-Z][a-zA-Z'&\-]+){0,5})\b")
QUOTE_RE = re.compile(r"[\"“']([^\"“”'\n]{3,60})[\"”']")

# Scaffold-context markers: an occurrence whose +/-30-char window contains one
# of these is boilerplate (web-page metadata, framework tracebacks, bing
# chrome, orchestrator ledger JSON); anchors need >= 1 occurrence outside.
SCAFFOLD_CTX_RE = re.compile(
    r"og:|fb:app_id|twitter:|meta_tags|site-packages|userwarning|"
    r"warnings\.warn|traceback \(most recent|stream_executor|cuda|"
    r"model mismatch|form=qblh|cvid=|characters out of 2000|_ver\"|"
    r"viewport shows|address bar|ledger_response|is_request_satisfied|"
    r"next_speaker|is_progress_being_made|is_in_loop|instruction_or_question|"
    r"jsonld|@context|schema\.org|handheldfriendly|msapplication|"
    r"/home/|groupchat|tmp_code|openstreetmap|©|copyright|"
    r"command line skills|ai assistant|device-width|initial-scale|"
    r"systementropy|breadcrumblist|itemlistelement|epoch|apple-itunes|"
    r"msvalidate|be:api|be:mod|citation_|dc\.|gs_meta|facts to derive|"
    r"educated guesses|helpful assistant|og:site_name|app-id", re.I)
SCAFFOLD_CTX_WINDOW = 60   # iteration 2: 30 was too small for metadata keys

# Common-English words (plus web/agent-domain generics). Unigram entities and
# single-word quoted spans in this set are dropped; multi-word spans need at
# least one word NOT in this set. Kept moderate so task-specific unigrams
# (opencv, tripadvisor, kipchoge) survive.
COMMON_WORDS = set("""
the of and a to in is was for on as with by at from it an be this that or
which are were has have had not but his her its their they them then than
there here when where what who whom whose why how all any both each few more
most other some such only own same so no nor too very can will just should
now also after before during between into through above below again further
once out off over under about against because until while although though
if else unless since like unlike within without along across behind beyond
near around among toward towards upon per via
one two three four five six seven eight nine ten first second third fourth
fifth last next previous new old early late young
time year month week day date hour minute number part form place case point
work world life hand eye way side kind head house service friend father
mother question answer problem fact idea water money story result issue
area word business study book job word night point home water room mother
example paper group music person thing man woman child people family student
country city state school
january february march april may june july august september october november
december monday tuesday wednesday thursday friday saturday sunday
get make go know take see come think look want give use find tell ask seem
feel try leave call keep let begin help talk turn start show hear play run
move live believe hold bring happen write provide sit stand lose pay meet
include continue set learn change lead understand watch follow stop create
speak read allow add spend grow open walk win offer remember love consider
appear buy wait serve die send expect build stay fall cut reach kill remain
suggest raise pass sell require report decide pull return explain hope
develop carry break receive agree support hit produce eat cover catch draw
choose cause listen
good great high small large big long little own other old right different
important public bad same able best better sure free low late hard major
possible whole real certain current available likely single true full
special easy clear recent strong entire final exact careful ready simple
total general common poor natural significant similar detailed direct
various actual specific particular relevant additional necessary correct
accurate complete initial original main further several many much
also very often however almost always usually together already still yet
ever never again once perhaps probably actually really quite rather enough
well far even back down up then now here there just only please thank
being been am does did doing done having going saying said says knew known
made took saw came thought looked wanted gave used found told asked seemed
felt tried left called kept began helped talked turned started showed heard
played ran moved lived believed held brought happened written provided sat
stood lost paid met included continued learned changed led understood
watched followed stopped created spoke spoken read allowed added spent grew
grown opened walked won offered remembered loved considered appeared bought
waited served died sent expected built stayed fell fallen cut reached
killed remained suggested raised passed sold required reported decided
pulled returned explained hoped developed carried broke broken received
agreed supported hit produced ate eaten covered caught drew drawn chose
chosen caused listened
name names list lists item items type types value values text texts line
lines page pages file files data set sets link links image images video
videos site sites web website term terms detail details info information
content contents section sections table tables row rows column columns
field fields format formats order orders search searches query queries
review reviews rating ratings count counts total totals sum average min max
found using used need needs needed require verify verified check checked
confirm confirmed identify identified determine determined locate located
obtain obtained perform performed execute executed analyze analyzed
calculate calculated compute computed process processed extract extracted
display displayed print printed load loaded save saved download downloaded
upload uploaded click clicked scroll scrolled type typed enter entered
select selected result results step steps plan plans goal goals note notes
error errors warning warnings message messages output outputs input inputs
code codes script scripts function functions method methods variable
variables object objects class classes module modules library libraries
package packages import imports version versions update updates updated
source sources title titles author authors article articles report reports
women men children athletes people persons
distance distances pace color colors colour colours platform platforms bed
beds bath baths seafood thousand thousands million billion difference
differences probability probabilities language languages verification
dictionary dictionaries header headers footer transcription transcribe
transcribed proceed proposed proposal manually approximately button buttons
option options direction directions sign signs least filter filters
filtered replace replaced genome genomes historical weather html player
players success successful successfully station stations category
categories restaurant restaurants museum museums hotel hotels store stores
compare compared amount amounts consistent traveled travel retrieve
retrieved submit submitted entertainment annual cost costs daily saving
savings member members validation must rule rules access rewards install
installed installing fetch fetched useful driving derive derived
gym gyms bird birds pitcher pitchers mock gluten membership marathon
sort sorted convert converted explore explored draft drafted post posted
description descriptions profile profiles deep tool tools steak venue
venues closed larger examine examined sample samples dataset datasets
location locations recommendation recommendations syntax map maps
review reviews
""".split())

# code / template vocabulary that shows up as quoted spans
CODE_TOKEN_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
CODE_WORDS = {"__main__", "main", "self", "none", "true", "false", "print",
              "return", "import", "def", "class", "try", "except", "exception",
              "valueerror", "typeerror", "importerror", "syntaxerror",
              "keyerror", "indexerror", "filenotfounderror", "attributeerror",
              "runtimeerror", "nameerror", "googlemaps", "pandas", "numpy",
              "dataframe", "matplotlib", "requests", "beautifulsoup"}
# metadata-key shape: ns:key with no slash (og:type, twitter:card)
META_KEY_RE = re.compile(r"^[\w.-]+:[\w.-]+$")
ANCHOR_BLACKLIST_SUBSTR = ("bing.com/sa", "facebook_sharing", "form=qblh",
                           "og:image", "cvid=", "characters out of 2000",
                           "automatic ocr", "ledger_response")

# Scaffold stoplist: multi-agent framework vocabulary, orchestrator boiler-
# plate, role names, generic web chrome. Kept deliberately aggressive for
# precision; the audit measures what survives.
STOPLIST = {
    "the", "this", "that", "there", "then", "they", "their", "them", "when",
    "what", "where", "which", "while", "with", "would", "could", "should",
    "here", "have", "has", "had", "will", "shall", "please", "note", "also",
    "however", "therefore", "thus", "hence", "given", "based", "using",
    "user", "assistant", "human", "orchestrator", "websurfer", "filesurfer",
    "computer_terminal", "computerterminal", "terminal", "agent", "agents",
    "expert", "experts", "manager", "step", "steps", "task", "tasks",
    "question", "answer", "final", "plan", "fact", "facts", "sheet",
    "request", "response", "updated", "ledger", "progress", "next", "team",
    "chat", "code", "output", "input", "python", "script", "error", "errors",
    "true", "false", "none", "null", "yes", "no", "ok", "okay",
    "bing", "google", "search", "results", "page", "click", "clicked",
    "screenshot", "browser", "web", "website", "link", "links", "url",
    "address", "bar", "tab", "window", "scroll", "scrolled", "viewport",
    "i", "we", "you", "it", "he", "she", "a", "an", "in", "on", "at", "to",
    "as", "is", "are", "was", "were", "be", "been", "if", "of", "or", "and",
    "but", "not", "let", "lets", "let's", "some", "any", "all", "each",
    "reason", "mistake", "verify", "verified", "instruction", "instructions",
    "important", "remember", "context", "summary", "detail", "details",
    "information", "current", "state", "status", "date", "today", "time",
    "first", "second", "third", "last", "new", "old", "provided", "following",
    "above", "below", "check", "checked", "confirm", "confirmed", "ensure",
    "exitcode", "execution", "succeeded", "failed", "unknown", "sure",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "gaia", "excel_expert", "validation_expert", "requirements",
    "specifically", "additionally", "alternatively", "unfortunately",
    "meanwhile", "afterwards", "finally", "overall", "several", "various",
    # framework / chrome terms surfaced by audit iteration 1
    "terminate", "terminated", "copilot", "microsoft", "microsoft bing",
    "dummy", "assist", "groupchat", "autogen", "orchestrator", "websurfer",
    "filesurfer", "magentic", "deep search", "sign in", "dataset preview",
    "is_request_satisfied", "next_speaker", "is_progress_being_made",
    "is_in_loop", "instruction_or_question", "ocr", "automatic ocr",
    "screenshot", "webpage", "metadata", "meta_tags", "referrer",
    "linux", "tomtom", "captainagent", "dataframe", "wordreference",
    "i'll", "we'll", "you'll", "i'm", "i've", "we're", "won't", "don't",
    "can't", "it's", "that's", "he'll", "she'll", "they'll", "what's",
    "here's", "there's", "contact us", "mission and history",
    "educated guesses", "related searches", "see more", "sign in",
}

# numbers that are scaffold noise regardless of DF (step indices, tiny ints
# are already excluded by the >=2-digit rule; exclude ubiquitous round sizes)
NUM_STOP = {"10", "100", "1000", "2023", "2024", "12", "15", "20", "24", "30",
            "50", "60", "11", "13", "14", "16", "17", "18", "19", "21", "22",
            "23", "25"}
# Rationale: two-digit integers 10..25 and round hundreds collide with list
# indices, viewport coordinates, dates and pagination chrome in these traces;
# years and specific quantities >= 26 or with decimals/commas are kept.

DF_MAX_FRAC = 0.35   # drop anchors present in > 35% of all trajectories


def _is_common(word: str) -> bool:
    w = word.lower().strip("'’")
    if w in COMMON_WORDS:
        return True
    for suf in ("s", "es", "ed", "ing", "'s", "’s"):
        if w.endswith(suf) and w[: -len(suf)] in COMMON_WORDS:
            return True
    return False


def _content_ok(words: List[str]) -> bool:
    """Multi-word span rule: at least one non-common, non-stoplist word."""
    return any(not _is_common(w) and w.lower() not in STOPLIST for w in words)


def _non_scaffold_occurrence(text: str, span_start: int, span_end: int) -> bool:
    lo = max(0, span_start - SCAFFOLD_CTX_WINDOW)
    hi = min(len(text), span_end + SCAFFOLD_CTX_WINDOW)
    return not SCAFFOLD_CTX_RE.search(text, lo, hi)


def _cap_candidates(text: str) -> List[str]:
    """Capitalized n-gram candidates with unigram mid-sentence evidence."""
    out = []
    for m in CAP_SEQ_RE.finditer(text):
        if not _non_scaffold_occurrence(text, m.start(), m.end()):
            continue
        cand = m.group(1)
        words = cand.split()
        if len(words) > 1:
            # trim leading/trailing stoplist/common words first; if a single
            # word survives, it must pass the unigram rules below (audit
            # iteration 2 leak: 'Automatic OCR' -> 'automatic')
            while words and (words[0].lower() in STOPLIST or _is_common(words[0])):
                words = words[1:]
            while words and (words[-1].lower() in STOPLIST or _is_common(words[-1])):
                words = words[:-1]
            if not words:
                continue
            cand = " ".join(words)          # iteration-3 bug: trimmed words
                                            # were validated but not joined
        if len(words) == 1:
            w = words[0]
            if len(w) < 4 or w.lower() in STOPLIST or _is_common(w) \
                    or w.lower() in CODE_WORDS:
                continue
            # require at least one occurrence NOT at a sentence start
            starts = [mm.start() for mm in re.finditer(
                re.escape(w), text)]
            mid = False
            for st in starts:
                pre = text[max(0, st - 2):st]
                if pre and not re.search(r"(^|[.!?:\n]\s?|\n)$", pre):
                    mid = True
                    break
            if not mid:
                continue
        else:
            if not _content_ok(words):
                continue
            cand = " ".join(words)
        out.append(cand)
    return out


def _blacklisted(a: str) -> bool:
    return any(b in a for b in ANCHOR_BLACKLIST_SUBSTR)


@lru_cache(maxsize=100000)
def extract_anchors_cached(text: str):
    return extract_anchors(text)


def extract_anchors(text: str) -> Dict[str, str]:
    """Extract {normalized_anchor: type} from a text block."""
    anchors: Dict[str, str] = {}

    for m in URL_RE.finditer(text):
        u = norm_url(m.group(0))
        if "{" in u or "}" in u:
            continue                                   # f-string URL template
        if 4 <= len(u) <= 120 and not _blacklisted(u) and \
                _non_scaffold_occurrence(text, m.start(), m.end()):
            anchors.setdefault(u, "url")

    GENERIC_FILES = {"index.html", "main.py", "script.py", "requirements.txt",
                     "temp.py", "test.py", "output.txt", "functions.py",
                     "response.json", "scenario.py", "run.sh", "data.json",
                     "console_log.txt", "global_init.sh", "api.php"}
    for m in FILE_RE.finditer(text):
        f = norm_anchor(m.group(0))
        if len(f) >= 5 and f not in GENERIC_FILES and not _blacklisted(f) and \
                _non_scaffold_occurrence(text, m.start(), m.end()):
            anchors.setdefault(f, "file")

    for m in NUM_RE.finditer(text):
        n = norm_number(m.group(1))
        if n in NUM_STOP or len(n.replace(".", "")) < 2:
            continue
        if n[0] == "0" and "." not in n:      # leading-zero fragments (00, 0200)
            continue
        if not _non_scaffold_occurrence(text, m.start(), m.end()):
            continue
        anchors.setdefault(n, "number")

    ent_seen = []
    for cand in _cap_candidates(text):
        a = norm_anchor(cand)
        if 4 <= len(a) <= 60 and a not in STOPLIST and not _blacklisted(a):
            ent_seen.append(a)
    # sub-span dedup within entities: keep only maximal spans.  Ties in length
    # are broken lexicographically: sorting on length alone left equal-length
    # entities in ``set`` iteration order, which Python randomises per process,
    # so the *order* of entity anchors -- and through salience ties the
    # distilled block -- varied with PYTHONHASHSEED (fixed 2026-09-16, AUDIT G).
    ent_seen = sorted(set(ent_seen), key=lambda a: (-len(a), a))
    kept: List[str] = []
    for a in ent_seen:
        if not any(a != b and a in b for b in kept):
            kept.append(a)
    for a in kept:
        ws = a.split()
        if len(ws) == 2 and ws[0] == ws[1]:
            continue                                   # doubled-word artifact
        anchors.setdefault(a, "entity")

    for m in QUOTE_RE.finditer(text):
        q = norm_anchor(m.group(1))
        words = q.split()
        if not (3 <= len(q) <= 60 and len(words) <= 6):
            continue
        if q in STOPLIST or q in CODE_WORDS or _blacklisted(q):
            continue
        if q.startswith(("s ", "t ", "re ", "ll ", "ve ", "d ", "@")):
            continue                                   # contraction-split / @key
        if m.end() < len(text) and text[m.end():m.end() + 2].lower() in ("s ", "s.", "s,"):
            continue                                   # possessive-split ('Murray')
        if m.end() < len(text) and text[m.end():m.end() + 1] == ":":
            continue                                   # JSON/dict key position
        if "%" in q:
            continue                                   # strftime-style format
        if re.match(r"^[a-z_][\w.]*\.[a-z_][\w.]*$", q) and "/" not in q \
                and not re.match(r"^[\d.]+$", q):
            continue                                   # dotted code identifier
        if "\\n" in q or "\\t" in q:
            continue                                   # print-template escapes
        if "{" in q or "}" in q:                       # template fragments
            continue
        if META_KEY_RE.match(q) and "/" not in q:      # og:type-style keys
            continue
        if len(words) == 1:
            if (CODE_TOKEN_RE.match(q) and "_" in q):  # snake_case identifiers
                continue
            if _is_common(q):
                continue
        elif not _content_ok(words):
            continue
        if not _non_scaffold_occurrence(text, m.start(), m.end()):
            continue
        anchors.setdefault(q, "quoted")

    return anchors


@lru_cache(maxsize=200000)
def _boundary_pattern(anchor: str) -> re.Pattern:
    """Word-boundary containment pattern for a normalized anchor."""
    esc = re.escape(anchor)
    if re.match(r"^[\d.]+$", anchor):
        # numbers: hard boundaries, also forbid adjacent digits/decimal
        return re.compile(r"(?<![\d.])" + esc + r"(?![\d.])")
    return re.compile(r"(?<!\w)" + esc + r"(?!\w)")


def contains_anchor(norm_haystack: str, anchor: str) -> bool:
    if re.search(r"\w", anchor) is None:
        return False
    return bool(_boundary_pattern(anchor).search(norm_haystack))


# --------------------------------------------------------------------------
# Corpus-level document frequency (for scaffold filtering)
# --------------------------------------------------------------------------
_DF_CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                         "anchor_df.json")


def build_df(trajs, cache: bool = True) -> Dict[str, int]:
    cache_p = os.path.abspath(_DF_CACHE)
    if cache and os.path.exists(cache_p):
        with open(cache_p) as f:
            return json.load(f)
    df: Counter = Counter()
    for t in trajs:
        full = t.question + "\n" + "\n".join(s.content for s in t.steps)
        df.update(set(extract_anchors(full).keys()))
    out = dict(df)
    if cache:
        os.makedirs(os.path.dirname(cache_p), exist_ok=True)
        with open(cache_p, "w") as f:
            json.dump(out, f)
    return out


class AnchorIndex:
    """Precomputes per-trajectory anchor machinery with the DF filter."""

    def __init__(self, trajs):
        self.df = build_df(trajs)
        self.n_trajs = len(trajs)
        self.df_cut = DF_MAX_FRAC * self.n_trajs

    def keep(self, anchor: str) -> bool:
        return self.df.get(anchor, 0) <= self.df_cut

    def future_relevant(self, traj, k: int) -> Dict[str, str]:
        """A(T,k): anchors originating in prefix (steps <= k, question
        included) that are referenced again in the suffix. Returns
        {anchor: type}."""
        prefix = traj.prefix_text(k)
        suffix = norm_text(traj.suffix_text(k))
        cand = {a: ty for a, ty in extract_anchors(prefix).items()
                if self.keep(a)}
        out = {}
        for a, ty in cand.items():
            if contains_anchor(suffix, a):
                out[a] = ty
        # ground-truth answer string, when it originates in the prefix
        gt = norm_anchor(traj.ground_truth)
        if (gt and 2 <= len(gt) <= 80 and gt not in STOPLIST
                and re.search(r"\w", gt)
                and contains_anchor(norm_text(prefix), gt)):
            out[gt] = "answer"
        return out

    @staticmethod
    def recall(block_text: str, anchors: Dict[str, str]) -> Tuple[float, int, int]:
        """Fraction of future-relevant anchors recoverable from the shipped
        block text. Returns (recall, hit, total); recall=1.0 when the anchor
        set is empty (nothing the future needed -> nothing lost)."""
        if not anchors:
            return 1.0, 0, 0
        h = norm_text(block_text)
        hit = sum(1 for a in anchors if contains_anchor(h, a))
        return hit / len(anchors), hit, len(anchors)
