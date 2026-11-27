"""Synthetic pervasive-assistant episode generator.

200 episodes across four task families (errand-planning, home-control,
shopping, travel-desk), each with EXACT ground-truth state at every step:
known entities, tool results, executed side-effects (with idempotency keys),
and the pending plan. Steps are rendered as natural agent-log text so the
same handoff policies that operate on real traces operate here.

Probe set per (episode, k): {goal, each constraint, each tool-result value
observed by step k, each executed side-effect key, the pending next action}.
Probe recall of a shipped block = fraction of probe target strings contained
(normalized, word-boundary) in the rendered resumption context. Ground truth
is by construction; no LLM anywhere.
"""
import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from traces.parser import Step, Trajectory, est_tokens

FIRST = ["alex", "sam", "jordan", "maya", "ravi", "lena", "omar", "nina"]
STORES = ["GreenLeaf Pharmacy", "Cortado Roasters", "Maple Hardware",
          "Luna Groceries", "Pergola Books", "Vela Electronics"]
ITEMS = ["prescription refill", "espresso beans", "wall anchors",
         "oat milk", "field notebook", "usb-c dock"]
DOCS = ["signed lease", "insurance form", "visa application",
        "warranty claim", "tax packet"]
OFFICES = ["Northgate Notary", "CityHall Annex", "Harbor Clinic",
           "Westside FedEx", "Consulate Desk"]
ROOMS = ["living room", "kitchen", "bedroom", "study", "hallway"]
DEVICES_HOME = ["thermostat", "ceiling light", "robot vacuum",
                "door lock", "speaker volume", "blinds"]
PRODUCTS = ["trail running shoes", "noise-cancel headphones",
            "standing desk mat", "ceramic pour-over set", "bike pannier"]
SITES = ["nordvia.com", "peakgear.io", "hausmarkt.de", "brightcart.com"]
CITIES = ["Lisbon", "Osaka", "Tromso", "Valparaiso", "Ljubljana", "Da Nang"]
AIRLINES = ["Aerlin", "TransBora", "Cielo Azul", "NordJet"]
HOTELS = ["Hotel Miradouro", "Ryokan Hoshi", "Fjord Lodge",
          "Casa Andina", "Vila Bled Inn"]


@dataclass
class SynthState:
    """Exact ground-truth state after each step."""
    entities: Dict[str, str] = field(default_factory=dict)   # name -> value
    tool_results: List[Tuple[str, str]] = field(default_factory=list)
    side_effects: List[Dict] = field(default_factory=list)   # {action,args,key}
    plan_remaining: List[str] = field(default_factory=list)


@dataclass
class SynthEpisode:
    eid: str
    family: str
    goal: str
    constraints: List[str]
    steps: List[Step]
    states: List[SynthState]          # states[i] = state AFTER step i

    @property
    def n(self):
        return len(self.steps)

    def to_trajectory(self) -> Trajectory:
        return Trajectory(tid=self.eid, source="synthetic",
                          question=self.goal, ground_truth="", steps=self.steps)

    def probes(self, k: int) -> Dict[str, List[str]]:
        """Probe targets at handoff position k (state after step k)."""
        st = self.states[k]
        return {
            "goal": [self.goal.split(".")[0]],
            "constraints": list(self.constraints),
            "tool_results": [v for _, v in st.tool_results],
            "side_effects": [se["key"] for se in st.side_effects],
            "pending": st.plan_remaining[:1],
        }


def _key(eid, i, action):
    return f"idem-{hashlib.sha1(f'{eid}|{i}|{action}'.encode()).hexdigest()[:10]}"


class _Builder:
    def __init__(self, eid):
        self.eid = eid
        self.steps: List[Step] = []
        self.states: List[SynthState] = []
        self.cur = SynthState()

    def _push(self, agent, text, side_effect=False, tool=False):
        i = len(self.steps)
        self.steps.append(Step(
            idx=i, role="assistant", agent=agent, content=text,
            side_effect_like=side_effect, is_tool_result=tool,
            tokens=est_tokens(text)))
        # snapshot state AFTER this step
        self.states.append(SynthState(
            entities=dict(self.cur.entities),
            tool_results=list(self.cur.tool_results),
            side_effects=[dict(s) for s in self.cur.side_effects],
            plan_remaining=list(self.cur.plan_remaining)))

    def plan(self, items: List[str]):
        self.cur.plan_remaining = list(items)
        self._push("planner", "Plan:\n" + "\n".join(
            f"{j+1}. {p}" for j, p in enumerate(items)))

    def advance_plan(self):
        if self.cur.plan_remaining:
            self.cur.plan_remaining.pop(0)

    def tool(self, tool_name, query, result_name, result_value, note=""):
        self.cur.entities[result_name] = result_value
        self.cur.tool_results.append((result_name, result_value))
        self._push("tool-runner",
                   f"TOOL {tool_name}({query}) -> {result_name} = "
                   f"{result_value}. {note}", tool=True)

    def act(self, action, args, note="", preamble="", repeat_of=None):
        """Record one executed side-effect.

        ``action_id`` is the episode's exact ground truth for action
        *identity*: two ledger entries denote the same world action iff their
        ids match.  E8 uses it to score both error directions of the
        fingerprint function.  ``preamble`` pads the action text before the
        distinguishing argument, reproducing the real-trace situation where
        the head of a step is boilerplate and the discriminating argument
        is far into the body.
        """
        aid = repeat_of if repeat_of is not None else f"{action}|{args}"
        key = _key(self.eid, len(self.steps), action) if repeat_of is None \
            else self._key_for(aid)
        self.cur.side_effects.append(
            {"action": action, "args": args, "key": key, "action_id": aid})
        self._push("executor",
                   f"EXECUTED {action}({preamble}{args}) "
                   f"[idempotency-key {key}] status=done. {note}",
                   side_effect=True)

    def _key_for(self, aid):
        for se in self.cur.side_effects:
            if se.get("action_id") == aid:
                return se["key"]
        return _key(self.eid, len(self.steps), aid)

    def think(self, text):
        self._push("planner", text)


def _mk_errand(rng: random.Random, eid: str) -> SynthEpisode:
    who = rng.choice(FIRST)
    item, store = rng.choice(ITEMS), rng.choice(STORES)
    doc, office = rng.choice(DOCS), rng.choice(OFFICES)
    deadline = f"{rng.randrange(3, 7)}:{rng.choice(['00', '30'])} pm"
    budget = rng.randrange(20, 90)
    goal = (f"Run {who}'s errands: pick up the {item} at {store} and drop "
            f"the {doc} at {office}. Must finish before {deadline} and keep "
            f"spending under ${budget}.")
    cons = [f"before {deadline}", f"under ${budget}"]
    b = _Builder(eid)
    plan = [f"check {store} hours", f"reserve {item}",
            f"check {office} queue", f"send eta message",
            f"pay for {item}", f"confirm dropoff slot"]
    b.plan(plan)
    hours = f"{rng.randrange(8, 11)}am-{rng.randrange(5, 9)}pm"
    b.advance_plan()
    b.tool("maps.hours", store, f"{store} hours", hours)
    price = round(rng.uniform(6, budget * 0.8), 2)
    b.advance_plan()
    b.act("reserve_item", f"item={item}, store={store}",
          f"Reserved; quoted price ${price}.")
    b.tool("price.check", item, f"{item} price", f"${price}")
    wait = rng.randrange(5, 40)
    b.advance_plan()
    b.tool("queue.status", office, f"{office} wait", f"{wait} min")
    b.advance_plan()
    b.act("message_send", f"to={who}, text=eta with {wait} min queue",
          "Notified owner.")
    b.think(f"Queue is {wait} min; paying now keeps us before {deadline}.")
    b.advance_plan()
    b.act("purchase", f"item={item}, amount=${price}", "Card charged once.")
    b.advance_plan()
    b.act("booking", f"slot at {office} before {deadline}",
          "Dropoff slot held.")
    b.think(f"All done except physical dropoff of the {doc}; "
            f"remaining budget ${round(budget - price, 2)}.")
    return SynthEpisode(eid, "errand", goal, cons, b.steps, b.states)


def _mk_home(rng: random.Random, eid: str) -> SynthEpisode:
    room1, room2 = rng.sample(ROOMS, 2)
    dev1, dev2, dev3 = rng.sample(DEVICES_HOME, 3)
    temp = rng.randrange(18, 24)
    hour = rng.choice(["9 pm", "10 pm", "11 pm"])
    goal = (f"Prepare the flat for the evening: set the {dev1} in the "
            f"{room1}, then the {dev2}, and schedule the {dev3}. Must not "
            f"run the {dev3} after {hour} and keep the {room1} at {temp}C.")
    cons = [f"not .* after {hour}", f"{temp}C"]
    cons = [f"after {hour}", f"{temp}C"]
    b = _Builder(eid)
    b.plan([f"read {room1} sensors", f"set {dev1}", f"set {dev2}",
            f"schedule {dev3}", "verify states"])
    cur_t = temp + rng.choice([-3, -2, 2, 3])
    b.advance_plan()
    b.tool("sensor.read", room1, f"{room1} temperature", f"{cur_t}C",
           f"target is {temp}C")
    b.advance_plan()
    b.act("device_actuation", f"{dev1} in {room1} -> target {temp}C",
          "Actuated.")
    lvl = rng.randrange(20, 80)
    b.advance_plan()
    b.act("device_actuation", f"{dev2} -> {lvl}%", "Actuated.")
    start = f"{rng.randrange(6, 9)}:15 pm"
    b.think(f"Scheduling {dev3} at {start} finishes before {hour}.")
    b.advance_plan()
    b.act("device_actuation", f"schedule {dev3} at {start}", "Scheduled.")
    b.advance_plan()
    b.tool("sensor.read", room1, f"{room1} verified temperature",
           f"{temp}C", "converged")
    b.think("Evening routine armed; nothing left except verification at "
            f"{hour}.")
    return SynthEpisode(eid, "home", goal, cons, b.steps, b.states)


def _mk_shopping(rng: random.Random, eid: str) -> SynthEpisode:
    prod = rng.choice(PRODUCTS)
    s1, s2 = rng.sample(SITES, 2)
    cap = rng.randrange(60, 240)
    goal = (f"Buy {prod} at the best price across {s1} and {s2}. Must stay "
            f"under ${cap} and use the saved shipping address only.")
    cons = [f"under ${cap}", "saved shipping address"]
    b = _Builder(eid)
    b.plan([f"search {s1}", f"search {s2}", "compare offers",
            "add to cart", "purchase", "email receipt"])
    p1 = round(rng.uniform(cap * 0.5, cap * 1.2), 2)
    p2 = round(rng.uniform(cap * 0.5, cap * 1.2), 2)
    b.advance_plan()
    b.tool("web.search", f"{prod} site:{s1}", f"{s1} offer", f"${p1}")
    b.advance_plan()
    b.tool("web.search", f"{prod} site:{s2}", f"{s2} offer", f"${p2}")
    best_site, best = (s1, p1) if p1 <= p2 else (s2, p2)
    b.advance_plan()
    b.think(f"Best offer is ${best} at {best_site}; cap is ${cap}.")
    if best <= cap:
        b.advance_plan()
        b.act("cart_add", f"{prod} at {best_site} for ${best}", "In cart.")
        b.advance_plan()
        b.act("purchase", f"{prod}, ${best}, saved address", "Order placed.")
        oid = f"ORD-{rng.randrange(10**5, 10**6)}"
        b.tool("order.status", best_site, "order id", oid, "confirmation")
        b.advance_plan()
        b.act("message_send", f"receipt {oid} to owner email", "Receipt sent.")
    else:
        b.think(f"Both offers exceed ${cap}; setting a price watch instead.")
        b.advance_plan()
        b.act("file_write", f"pricewatch.json for {prod} threshold ${cap}",
              "Watch file written.")
    return SynthEpisode(eid, "shopping", goal, cons, b.steps, b.states)


def _mk_travel(rng: random.Random, eid: str) -> SynthEpisode:
    city = rng.choice(CITIES)
    airline = rng.choice(AIRLINES)
    hotel = rng.choice(HOTELS)
    nights = rng.randrange(2, 6)
    cap = rng.randrange(500, 1400)
    dep = f"202{rng.randrange(5, 7)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 28):02d}"
    goal = (f"Arrange a {nights}-night trip to {city}: flight, hotel, and an "
            f"itinerary file. Must depart {dep}, keep total under ${cap}, "
            f"and book refundable fares only.")
    cons = [f"depart {dep}", f"under ${cap}", "refundable"]
    b = _Builder(eid)
    b.plan([f"search flights to {city}", "hold flight",
            f"search hotels in {city}", "book hotel", "write itinerary",
            "send summary"])
    fare = round(rng.uniform(cap * 0.25, cap * 0.6), 2)
    fl = f"{airline} {rng.randrange(100, 900)}"
    b.advance_plan()
    b.tool("flights.search", f"{city} {dep}", "best flight",
           f"{fl} at ${fare}", "refundable fare class")
    b.advance_plan()
    b.act("booking", f"hold {fl} on {dep} for ${fare}", "Fare held 24h.")
    rate = round(rng.uniform(60, (cap - fare) / max(nights, 1)), 2)
    b.advance_plan()
    b.tool("hotels.search", city, "best hotel", f"{hotel} at ${rate}/night")
    total = round(fare + rate * nights, 2)
    b.think(f"Projected total ${total} vs cap ${cap}.")
    b.advance_plan()
    b.act("booking", f"{hotel}, {nights} nights at ${rate}", "Refundable.")
    b.advance_plan()
    b.act("file_write", f"itinerary_{city.lower()}.md",
          f"Wrote itinerary with {fl} and {hotel}.")
    b.advance_plan()
    b.act("message_send", f"trip summary for {city} to owner",
          "Summary sent.")
    return SynthEpisode(eid, "travel", goal, cons, b.steps, b.states)


def _mk_fleet(rng: random.Random, eid: str) -> SynthEpisode:
    """Device-fleet operations: long trajectories whose executed actions
    share a long boilerplate head and differ only in a late argument, plus
    genuinely repeated actions.

    This family exists to stress the *ledger*, not the distiller: it is the
    regime in which a head-truncated action fingerprint cannot tell two
    different actuations apart (collision -> false suppression) and in which
    the mandatory ledger competes with the rest of the block for a wearable
    budget (E8).
    """
    site = rng.choice(["Northfield", "Bayview", "Harbour", "Ridgeline"])
    zones = rng.sample(ROOMS, 4)
    n_dev = rng.randrange(5, 9)
    goal = (f"Run the evening close-down for the {site} site: verify each "
            f"zone, actuate every managed device, and file the report. Must "
            f"not actuate any device twice and must finish before 11 pm.")
    cons = ["not actuate any device twice", "before 11 pm"]
    b = _Builder(eid)
    b.plan([f"survey {z}" for z in zones] + ["actuate fleet", "file report"])
    PRE = (f"site={site}, operator=scheduler, ticket=close-down, "
           f"policy=evening-v3, retries=0, window=18:00-23:00, "
           f"zones={'/'.join(zones)}, mode=sequential, confirm=true, ")
    acted = []
    for j in range(n_dev):
        z = zones[j % len(zones)]
        dev = DEVICES_HOME[j % len(DEVICES_HOME)]
        b.advance_plan()
        b.tool("fleet.status", f"{z}/{dev}", f"{z} {dev} state",
               rng.choice(["on", "off", "idle"]))
        b.act("device_actuation", f"zone={z}, device={dev}, target=off",
              "Actuated.", preamble=PRE)
        acted.append((z, dev))
    # one genuinely repeated action: the scheduler retries the first device
    z0, d0 = acted[0]
    b.think("Scheduler retried the first zone after a transient timeout.")
    b.act("device_actuation", f"zone={z0}, device={d0}, target=off",
          "Actuated (retry).", preamble=PRE,
          repeat_of=f"device_actuation|zone={z0}, device={d0}, target=off")
    b.advance_plan()
    b.act("file_write", f"closedown_{site.lower()}.md",
          "Report filed.", preamble=PRE)
    return SynthEpisode(eid, "fleet", goal, cons, b.steps, b.states)


FAMILIES = {"errand": _mk_errand, "home": _mk_home,
            "shopping": _mk_shopping, "travel": _mk_travel,
            "fleet": _mk_fleet}


def generate(n: int = 200, seed: int = 0) -> List[SynthEpisode]:
    rng = random.Random(seed)
    fams = list(FAMILIES)
    out = []
    for i in range(n):
        fam = fams[i % len(fams)]
        eid = f"synth-{fam}-{i:03d}"
        out.append(FAMILIES[fam](random.Random((seed, i)), eid))
    return out


if __name__ == "__main__":
    eps = generate()
    import statistics
    lens = [e.n for e in eps]
    sfx = [len(e.states[-1].side_effects) for e in eps]
    print(f"{len(eps)} episodes; steps med {statistics.median(lens)} "
          f"range {min(lens)}-{max(lens)}; side-effects med "
          f"{statistics.median(sfx)}")
    e = eps[0]
    k = e.n // 2
    print("probes @k:", {k2: len(v) for k2, v in e.probes(k).items()})
