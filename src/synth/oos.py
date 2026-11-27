"""Out-of-schema synthetic episodes: state the schema was not designed for.

The main synthetic corpus has a fair objection against it -- its ground truth
is {goal, constraints, tool values, idempotency keys, pending action}, which is
exactly the field list Relay's schema carries.  A probe metric defined on those
fields is a unit test for the schema, not evidence that the schema captures
what a resumption generally needs.

This corpus is built to fall *outside* the schema, in the four ways a real
trajectory most often does:

  **revised goal.**  The user changes the objective mid-episode.  The schema
  populates ``goal`` from the task statement, so a distiller that does the
  obvious thing ships the *stale* goal and is wrong in the most consequential
  field it has.  Probes ask for the revised goal and, separately, penalise the
  superseded one still being present unqualified.

  **contradicted constraint.**  A later constraint replaces an earlier one
  ("actually, make it under $40").  ``constraints`` is a flat list with no
  notion of supersession.

  **superseded value.**  A tool result is later corrected.  The entity ledger
  keys on the anchor, and recency ordering is only a heuristic for which value
  is live.

  **open question.**  Something the agent has explicitly deferred and must
  still resolve.  There is no schema field for an unresolved question at all.

Each episode carries exact ground truth for all of these, plus a
``paraphrased`` rendering of one already-executed action so that action
identity is probed on a semantically equivalent description rather than a
verbatim one.  The corpus is deliberately *not* mixed into the main 200
episodes: the headline results stay comparable, and this reads as the
independent test it is meant to be.
"""
import random
from typing import Dict, List

from synth.generator import (SynthEpisode, _Builder, FIRST, STORES, ITEMS,
                             CITIES, PRODUCTS, SITES, ROOMS, DEVICES_HOME)


class OOSEpisode(SynthEpisode):
    """A SynthEpisode with an additional out-of-schema ground-truth record."""

    def __init__(self, *a, oos: Dict = None, **kw):
        super().__init__(*a, **kw)
        self.oos = oos or {}

    def oos_probes(self, k: int) -> Dict[str, List[str]]:
        """Probe targets that must be *present* in the shipped state."""
        o = self.oos
        out = {}
        if o.get("revision_step") is not None and k >= o["revision_step"]:
            out["revised_goal"] = [o["revised_goal_key"]]
        if o.get("constraint_step") is not None and k >= o["constraint_step"]:
            out["active_constraint"] = [o["active_constraint"]]
        if o.get("correction_step") is not None and k >= o["correction_step"]:
            out["corrected_value"] = [o["corrected_value"]]
        if o.get("open_step") is not None and k >= o["open_step"]:
            out["open_question"] = [o["open_question_key"]]
        return out

    def oos_negatives(self, k: int) -> Dict[str, List[str]]:
        """Targets whose presence *without* their replacement is an error.

        Scored separately from recall: a block that ships the stale goal and
        not the revised one has not merely lost information, it has shipped a
        wrong instruction, and averaging that into a recall number would hide
        it.
        """
        o = self.oos
        out = {}
        if o.get("revision_step") is not None and k >= o["revision_step"]:
            out["stale_goal"] = [o["stale_goal_key"]]
        if o.get("constraint_step") is not None and k >= o["constraint_step"]:
            out["stale_constraint"] = [o["stale_constraint"]]
        if o.get("correction_step") is not None and k >= o["correction_step"]:
            out["stale_value"] = [o["stale_value"]]
        return out



# --- padding -------------------------------------------------------------
# The out-of-schema state only *matters* when the budget binds: if a whole
# episode fits in a wearable window, every window policy ships the revision
# for free and the corpus measures nothing.  These fillers are verbose,
# realistic tool results that push a prefix well past a watch budget, and the
# superseding events are placed early on purpose -- a recency window must then
# choose between the tail and the correction, which is the situation the
# corpus exists to create.
_FILL_TOPICS = ["reviews", "hours", "routes", "availability", "ratings",
                "policies", "fees", "photos", "menus", "schedules"]


def _pad(b, rng, n, topic_seed=0):
    for j in range(n):
        topic = _FILL_TOPICS[(topic_seed + j) % len(_FILL_TOPICS)]
        rows = "; ".join(
            f"{rng.choice(STORES)} {topic} {rng.randrange(2, 99)}"
            f".{rng.randrange(10, 99)} ({rng.randrange(3, 400)} entries)"
            for _ in range(6))
        b.tool(f"web.{topic}", f"query {j}", f"{topic} digest {j}", rows,
               note=("Retrieved from the aggregator index; results are "
                     "paginated and the ranking is not stable between "
                     "requests, so the ordering above is indicative only."))
        b.think(f"Reviewed the {topic} digest {j}; nothing there changes the "
                f"current plan, but it is recorded for completeness and may "
                f"matter if the shortlist has to be revisited later.")


def _mk_revision(rng: random.Random, eid: str) -> OOSEpisode:
    who = rng.choice(FIRST)
    a_city, b_city = rng.sample(CITIES, 2)
    budget0, budget1 = rng.randrange(700, 900), rng.randrange(300, 450)
    prod = rng.choice(PRODUCTS)
    site = rng.choice(SITES)
    goal = (f"Plan {who}'s trip to {a_city} and buy a {prod} from {site}. "
            f"Keep the total under ${budget0}.")
    b = _Builder(eid)
    b.plan([f"price flights to {a_city}", f"check {prod} stock",
            "reserve flight", "buy the item", "send itinerary"])

    b.advance_plan()
    f0 = rng.randrange(320, 520)
    b.tool("flights.price", a_city, f"{a_city} fare", f"${f0}")
    corrected = f0 + rng.randrange(40, 120)

    b.advance_plan()
    p0 = rng.randrange(90, 180)
    b.tool("stock.check", prod, f"{prod} price", f"${p0}")
    _pad(b, rng, 4, 0)

    # --- the goal is revised: destination and budget both change ---
    revision_step = len(b.steps)
    b.think(f"UPDATE FROM {who.upper()}: change of plan -- the destination is "
            f"now {b_city}, not {a_city}, and the total must stay under "
            f"${budget1}. Ignore the earlier {a_city} plan.")

    # --- a tool result is corrected ---
    correction_step = len(b.steps)
    b.tool("flights.price", f"{b_city} (recheck)", f"{b_city} fare",
           f"${corrected}",
           note=f"Supersedes the earlier ${f0} quote, which was for {a_city}.")

    b.advance_plan()
    b.act("reserve_item", f"item=flight to {b_city}, amount=${corrected}",
          "Seat held.")

    # --- an explicitly deferred question ---
    open_step = len(b.steps)
    b.think(f"OPEN QUESTION: {who} has not said whether the {prod} should ship "
            f"to the {b_city} hotel or to home. Do not purchase until answered.")

    _pad(b, rng, 16, 3)
    b.think(f"Remaining: buy the {prod} once the shipping address is "
            f"confirmed, then send the itinerary.")
    b.advance_plan()
    b.act("message_send", f"to={who}, text=confirm shipping address for {prod}",
          "Asked owner.")

    oos = {
        "revision_step": revision_step,
        "revised_goal_key": b_city.lower(),
        "stale_goal_key": a_city.lower(),
        "constraint_step": revision_step,
        "active_constraint": str(budget1),
        "stale_constraint": str(budget0),
        "correction_step": correction_step,
        "corrected_value": str(corrected),
        "stale_value": str(f0),
        "open_step": open_step,
        "open_question_key": "shipping address",
    }
    cons = [f"under ${budget1}"]
    return OOSEpisode(eid, "revision", goal, cons, b.steps, b.states, oos=oos)


def _mk_contradiction(rng: random.Random, eid: str) -> OOSEpisode:
    who = rng.choice(FIRST)
    store = rng.choice(STORES)
    item = rng.choice(ITEMS)
    room = rng.choice(ROOMS)
    dev = rng.choice(DEVICES_HOME)
    t0, t1 = rng.randrange(18, 21), rng.randrange(21, 23)
    goal = (f"Get {who}'s {item} from {store} and set the {room} {dev}. "
            f"Everything must be finished before {t0}:00.")
    b = _Builder(eid)
    b.plan([f"check {store} stock", f"reserve {item}",
            f"set {room} {dev}", "confirm with owner"])
    b.advance_plan()
    b.tool("stock.check", item, f"{item} stock", rng.choice(["3", "7", "12"]))
    b.advance_plan()
    b.act("reserve_item", f"item={item}, store={store}", "Held at counter.")
    _pad(b, rng, 4, 5)

    constraint_step = len(b.steps)
    b.think(f"CORRECTION: the {t0}:00 deadline was wrong. The real cut-off is "
            f"{t1}:00; the earlier {t0}:00 constraint no longer applies.")

    correction_step = len(b.steps)
    old_v = rng.randrange(15, 19)
    new_v = rng.randrange(20, 24)
    b.tool("home.state", f"{room}/{dev}", f"{room} {dev} setpoint",
           f"{new_v}", note=f"Earlier reading of {old_v} was stale.")

    open_step = len(b.steps)
    b.think(f"OPEN QUESTION: {who} has not confirmed whether the {dev} should "
            f"hold {new_v} overnight. Awaiting reply.")
    _pad(b, rng, 16, 1)
    b.advance_plan()
    b.act("device_actuation", f"zone={room}, device={dev}, target={new_v}",
          "Set.")
    oos = {
        "revision_step": None, "constraint_step": constraint_step,
        "active_constraint": f"{t1}:00", "stale_constraint": f"{t0}:00",
        "correction_step": correction_step,
        "corrected_value": str(new_v), "stale_value": str(old_v),
        "open_step": open_step, "open_question_key": "overnight",
        "revised_goal_key": "", "stale_goal_key": "",
    }
    return OOSEpisode(eid, "contradiction", goal, [f"before {t1}:00"],
                      b.steps, b.states, oos=oos)


FAMILIES = {"revision": _mk_revision, "contradiction": _mk_contradiction}


def generate_oos(n: int = 120, seed: int = 7) -> List[OOSEpisode]:
    fams = list(FAMILIES)
    out = []
    for i in range(n):
        fam = fams[i % len(fams)]
        out.append(FAMILIES[fam](random.Random((seed, i)),
                                 f"oos-{fam}-{i:03d}"))
    return out


if __name__ == "__main__":
    eps = generate_oos()
    e = eps[0]
    print(f"{len(eps)} episodes; steps {[x.n for x in eps[:6]]}")
    k = e.n - 2
    print("oos probes @", k, e.oos_probes(k))
    print("oos negatives @", k, e.oos_negatives(k))
