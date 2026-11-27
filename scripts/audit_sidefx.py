"""E11: auditing the side-effect detector in both directions.

The safety results rest on a detector: 15 regex patterns decide which real
steps are "side-effect-like" and therefore get a ledger entry.  A safety
ledger is only as complete as the detector that feeds it, and *recall* errors
are the dangerous direction: a missed action gets no ledger entry and
therefore no protection at all, while a precision error only wastes budget.

The first version of this system reported no audit of this detector.  This
script produces one, in the same shape as the anchor audits:

  ``dump``   builds a stratified candidate pool and writes it for annotation
  ``score``  reads the judged file and computes precision, recall and
             per-class coverage, with a Wilson interval on each rate.

**Strata.**  Recall cannot be measured against the detector's own output, so
the pool is deliberately built to contain steps the detector rejects:

  ``flagged``    steps the shipped detector labels (precision stratum);
  ``broad``      steps a much wider net catches -- any acting-agent step, any
                 step with a code block or command line, any past-tense
                 action verb -- but the shipped detector does *not*
                 (the recall stratum, where its misses live);
  ``neither``    a uniform sample of everything neither net catches.

The third stratum is what makes the recall number a bound rather than a
self-assessment: without it we could only measure how well the detector
agrees with a slightly larger regex, and a blind spot shared by both nets
would be invisible.  Recall is estimated with the strata reweighted to their
population sizes.

**Judgment criterion.**  Annotation applies one rule to every stratum: a step
counts only if it *reports an already-performed action*, never a proposed,
planned, requested or hypothetical one, and never a later summary that merely
mentions an earlier action (that action's own step already carries it).

Annotation records the rule at two thresholds, because this corpus makes the
distinction unavoidable:

  ``is_side_effect`` (**strict**)  the step changed state outside the agent:
      code executed, a file written, a message sent, an order placed, a device
      actuated, a form committed.  This is the population a ledger exists for,
      so its *recall* is the safety-critical number.
  ``is_side_effect_broad``         strict, plus performed but retrieval-only
      interaction: a search query issued, a link clicked, a page scrolled, a
      file read.  This is what the shipped patterns actually match.

Reporting one without the other would either flatter the detector (its broad
precision) or misdescribe what the real-trace labels contain (its strict
precision).  Who\&When is a corpus of research tasks, so its performed actions
are overwhelmingly code execution and browser retrieval; that fact is the
reason the synthetic episodes exist, and it belongs in the paper rather than
in a footnote.
"""
import csv
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from explib import RES                                          # noqa: E402
from traces.parser import load_trajectories                     # noqa: E402

TODO = os.path.join(RES, "sidefx_audit_todo.csv")
JUDGED = os.path.join(RES, "sidefx_audit_judged.csv")
READABLE = os.path.join(RES, "sidefx_audit_readable.txt")
SUMMARY = os.path.join(RES, "sidefx_audit_summary.json")
POP = os.path.join(RES, "sidefx_audit_pop.json")

# --- the over-generating net (recall stratum) ------------------------------
# Deliberately far broader than the shipped detector: it is allowed to be
# wrong, because its only job is to surface candidates a human then judges.
ACTING_AGENT = re.compile(r"computer_?terminal|websurfer|filesurfer|"
                          r"web_surfer|file_surfer|executor|assistant", re.I)
BROAD_VERB = re.compile(
    r"\b(ran|executed|typed|entered|clicked|selected|pressed|"
    r"submitted|sent|posted|uploaded|downloaded|installed|created|deleted|"
    r"removed|renamed|moved|copied|wrote|saved|updated|modified|patched|"
    r"scheduled|booked|reserved|cancelled|canceled|ordered|purchased|paid|"
    r"charged|actuated|toggled|turned (on|off)|logged in|"
    r"signed in|authenticated|navigated|opened|applied|committed|"
    r"pushed|merged|invoked|triggered)\b", re.I)
CODEY = re.compile(r"```|^\s*\$ |\bpip install\b|\bpython \S+\.py\b|"
                   r"\bcurl \b|\bgit \b|\bmkdir\b", re.M | re.I)

#: Coarse action classes, for the coverage table.  A detector can have
#: acceptable aggregate recall and still be blind to one whole class, which
#: is the failure mode that matters for a safety mechanism.
CLASSES = [
    ("exec",      re.compile(r"```|\bexitcode\b|\bexecut|\bran the\b|"
                             r"\bpython\b|\bscript\b|\bcommand\b", re.I)),
    ("web_input", re.compile(r"\btyped\b|\bclicked\b|\bsubmitted\b|"
                             r"\bentered\b|\bpressed\b|\bselected\b|"
                             r"\bfilled\b", re.I)),
    ("file",      re.compile(r"\bwrote\b|\bsaved\b|\bcreated\b|\bdeleted\b|"
                             r"\bdownloaded\b|\buploaded\b|\bfile\b", re.I)),
    ("message",   re.compile(r"\bsent\b|\bposted\b|\bemail(ed)?\b|"
                             r"\bmessage\b|\breplied\b", re.I)),
    ("commerce",  re.compile(r"\bordered\b|\bpurchased\b|\bbooked\b|"
                             r"\breserved\b|\bpaid\b|\bcharged\b", re.I)),
    ("actuate",   re.compile(r"\bactuated\b|\bturned (on|off)\b|"
                             r"\btoggled\b|\bscheduled\b", re.I)),
]


def classify(text: str) -> str:
    for name, rx in CLASSES:
        if rx.search(text):
            return name
    return "other"


def _stratum(step) -> str:
    if step.side_effect_like:
        return "flagged"
    t = step.content
    if CODEY.search(t) or BROAD_VERB.search(t) or (
            ACTING_AGENT.search(step.agent or "") and len(t) > 80):
        return "broad"
    return "neither"


def dump(n_flagged=90, n_broad=90, n_random=60, seed=17):
    """Stratified candidate pool -> CSV for annotation."""
    trajs = load_trajectories()
    pools = {"flagged": [], "broad": [], "neither": []}
    for t in trajs:
        for s in t.steps:
            pools[_stratum(s)].append((t.tid, s))
    pop = {k: len(v) for k, v in pools.items()}

    rng = random.Random(seed)
    want = {"flagged": n_flagged, "broad": n_broad, "neither": n_random}
    rows = []
    for st, k in want.items():
        for tid, s in rng.sample(pools[st], min(k, len(pools[st]))):
            body = re.sub(r"\s+", " ", s.content).strip()
            rows.append({
                "tid": tid, "step": s.idx, "agent": s.agent, "stratum": st,
                "detector": int(s.side_effect_like),
                "class_guess": classify(s.content),
                "text": body[:900],
                "is_side_effect": "",       # <- annotator fills 1/0
                "note": "",
            })
    rng.shuffle(rows)                        # annotate blind to stratum order

    os.makedirs(RES, exist_ok=True)
    with open(TODO, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(READABLE, "w") as f:
        f.write(f"# side-effect detector audit pool (population: "
                f"flagged={pop['flagged']} broad={pop['broad']} "
                f"neither={pop['neither']})\n")
        f.write("# criterion: the step REPORTS an already-performed action "
                "with external consequence.\n\n")
        for i, r in enumerate(rows):
            f.write(f"--- [{i}] {r['tid']}#{r['step']} agent={r['agent']} "
                    f"stratum={r['stratum']} detector={r['detector']}\n")
            f.write(r["text"][:900] + "\n\n")
    json.dump(pop, open(POP, "w"), indent=1)
    print(f"wrote {len(rows)} rows -> {TODO}")
    print(f"population: {pop}")


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return p, (c - s) / d, (c + s) / d


def _rates(rows, pop, label):
    """Precision, reweighted recall and per-class coverage for one label."""
    by = {}
    for r in rows:
        by.setdefault(r["stratum"], []).append(r)

    fl = by.get("flagged", [])
    tp = sum(int(r[label]) for r in fl)
    prec, plo, phi = _wilson(tp, len(fl))

    est_genuine = est_caught = 0.0
    per_stratum = {}
    for st in ("flagged", "broad", "neither"):
        rs = by.get(st, [])
        if not rs:
            continue
        g = sum(int(r[label]) for r in rs)
        c = sum(int(r[label]) and int(r["detector"]) for r in rs)
        w = pop[st] / len(rs)
        est_genuine += g * w
        est_caught += c * w
        per_stratum[st] = {"n": len(rs), "genuine": g, "caught": c,
                           "population": pop[st], "weight": round(w, 2),
                           "genuine_rate": round(g / len(rs), 3)}
    recall_w = est_caught / est_genuine if est_genuine else 0.0

    all_g = [r for r in rows if int(r[label])]
    caught = sum(int(r["detector"]) for r in all_g)
    recall_s, rlo, rhi = _wilson(caught, len(all_g))

    cls = {}
    for r in all_g:
        d = cls.setdefault(r["class_guess"], {"genuine": 0, "caught": 0})
        d["genuine"] += 1
        d["caught"] += int(r["detector"])
    for d in cls.values():
        d["recall"] = round(d["caught"] / d["genuine"], 3) if d["genuine"] else 0.0

    misses = [{"tid": r["tid"], "step": r["step"], "class": r["class_guess"],
               "stratum": r["stratum"], "note": r.get("note", ""),
               "text": r["text"][:200]}
              for r in all_g if not int(r["detector"])]

    return {
        "precision": round(prec, 3),
        "precision_ci": [round(plo, 3), round(phi, 3)],
        "n_flagged_judged": len(fl),
        "recall_weighted": round(recall_w, 3),
        "recall_sample": round(recall_s, 3),
        "recall_ci": [round(rlo, 3), round(rhi, 3)],
        "n_genuine_judged": len(all_g),
        "per_stratum": per_stratum,
        "per_class": cls,
        "n_misses": len(misses),
        "misses": misses[:40],
    }


def score():
    """Precision and recall of the side-effect detector, under both labels.

    ``strict`` counts only steps that changed state outside the agent; it is
    the population the ledger is *for*, and its recall is the safety-critical
    number.  ``broad`` counts every performed external interaction, which is
    what the shipped patterns actually match.  Reporting one without the
    other would either flatter the detector (broad precision) or misdescribe
    what the real-trace labels contain (strict precision).
    """
    pop = json.load(open(POP))
    rows = [r for r in csv.DictReader(open(JUDGED))
            if r["is_side_effect"].strip() in ("0", "1")]
    out = {"n_judged": len(rows),
           "strict": _rates(rows, pop, "is_side_effect"),
           "broad": _rates(rows, pop, "is_side_effect_broad")}
    json.dump(out, open(SUMMARY, "w"), indent=1)
    brief = {k: {kk: vv for kk, vv in v.items() if kk != "misses"}
             if isinstance(v, dict) else v for k, v in out.items()}
    print(json.dumps(brief, indent=1))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dump"
    (dump if cmd == "dump" else score)()
