"""Target capability manifests and plan repair.

Sizing state to a context window is *one* capability dimension, and by
itself it does not deserve the word "capability-aware".  A speaker has no
screen; a watch has no browser and no file system worth the name; a target
device may hold no credential for the payment endpoint the pending plan is
about to call; and a tool-result pointer is only faultable while the source
is reachable.  A block that arrives perfectly sized and then asks the target
to do something it cannot do has migrated the task onto a dead end.

This module adds the missing dimension in the smallest form that is still
honest:

  * a **manifest** per device class -- tools, modalities, credential scopes,
    source reachability -- travelling in the block so the target's answer to
    "can I continue?" is decidable rather than discovered by failure;
  * a **requirement model** that maps a pending plan step to the capabilities
    it needs;
  * **plan repair**, which classifies each pending step against the target's
    manifest and rewrites what it can.

The verdicts are deliberately few, and each names a real deployment
response:

  ``FEASIBLE``    the target can do it as written.
  ``SUBSTITUTE``  same action, different modality (read a result aloud
                  instead of rendering it) -- the plan step is rewritten.
  ``DELEGATE``    the target cannot act but the source (or a cloud
                  continuation) is reachable and can, so the step is routed
                  back over the link that carried the handoff.
  ``DEFER``       nothing reachable can do it now; it stays pending and the
                  user is told, rather than being failed silently.
  ``BLOCKED``     a credential or scope is missing.  Never silently retried:
                  a missing scope is a security boundary, not an outage.

We claim no learned planner here.  Repair is a typed rewrite over a
requirement model, which is what makes it auditable; the point of the
experiment (E15) is to show how often the requirement actually binds on real
pending plans, which turns out to be often enough that ignoring it was the
weakest part of the original design.
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple


class Verdict(str, Enum):
    FEASIBLE = "feasible"
    SUBSTITUTE = "substitute"
    DELEGATE = "delegate"
    DEFER = "defer"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
#: Tool capabilities a pending step can require.
TOOLS = {"browser", "filesystem", "shell", "email", "payment", "calendar",
         "home_control", "camera", "location"}
#: Interaction modalities a step can require of the device.
MODALITIES = {"screen", "audio_out", "audio_in", "keyboard", "large_text"}
#: Credential scopes.  Held separately from tools: a watch may have a mail
#: *client* and no send scope.
SCOPES = {"mail.send", "pay.charge", "home.write", "files.write",
          "calendar.write"}


@dataclass
class CapabilityManifest:
    """What a device class can actually do, shipped with the block."""
    device: str
    tools: Set[str] = field(default_factory=set)
    modalities: Set[str] = field(default_factory=set)
    scopes: Set[str] = field(default_factory=set)
    source_reachable: bool = True     # can we fault in a digest pointer?
    cloud_reachable: bool = False     # is a cloud continuation available?

    def to_dict(self) -> Dict:
        return {"device": self.device, "tools": sorted(self.tools),
                "modalities": sorted(self.modalities),
                "scopes": sorted(self.scopes),
                "source_reachable": self.source_reachable,
                "cloud_reachable": self.cloud_reachable}

    def render(self) -> str:
        """One compact line for the block: the target reads its own limits."""
        return (f"device={self.device} tools={','.join(sorted(self.tools)) or '-'} "
                f"modes={','.join(sorted(self.modalities)) or '-'} "
                f"scopes={','.join(sorted(self.scopes)) or '-'} "
                f"src={'up' if self.source_reachable else 'down'}")


#: Default manifests for the four modelled classes.  These are *modelled*
#: like the throughput constants, and chosen conservatively from what the
#: platforms actually expose to a background agent: a watch has no general
#: browser and no shell, a speaker has no screen at all, a phone has
#: everything but a shell, a laptop has everything.
DEFAULT_MANIFESTS: Dict[str, CapabilityManifest] = {
    "watch": CapabilityManifest(
        "watch",
        tools={"calendar", "home_control", "location"},
        modalities={"screen", "audio_out", "audio_in"},
        scopes={"home.write", "calendar.write"}),
    "speaker": CapabilityManifest(
        "speaker",
        tools={"home_control", "calendar", "email"},
        modalities={"audio_out", "audio_in"},
        scopes={"home.write", "calendar.write", "mail.send"}),
    "phone": CapabilityManifest(
        "phone",
        tools={"browser", "filesystem", "email", "payment", "calendar",
               "home_control", "camera", "location"},
        modalities={"screen", "audio_out", "audio_in", "keyboard"},
        scopes={"mail.send", "pay.charge", "home.write", "calendar.write",
                "files.write"}),
    "laptop": CapabilityManifest(
        "laptop",
        tools=set(TOOLS),
        modalities={"screen", "keyboard", "large_text", "audio_out"},
        scopes=set(SCOPES)),
}


# ---------------------------------------------------------------------------
# Requirement model: what does this pending step need?
# ---------------------------------------------------------------------------
@dataclass
class Requirement:
    tools: Set[str] = field(default_factory=set)
    modalities: Set[str] = field(default_factory=set)
    scopes: Set[str] = field(default_factory=set)


#: (capability, regex) -- deterministic and auditable, in the same spirit as
#: the side-effect patterns, and audited the same way (E11b).
_TOOL_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("browser", re.compile(r"\b(browse|navigate|open the (page|site|url)|"
                           r"search the web|web ?search|click|scroll|"
                           r"web ?surfer|visit|url|https?://|webpage|website)\b", re.I)),
    ("filesystem", re.compile(r"\b(open|read|write|save|download|upload|"
                              r"attach) (the )?(file|document|spreadsheet|pdf|"
                              r"csv|folder)\b|\.(pdf|csv|xlsx?|docx?|txt|zip)\b", re.I)),
    ("shell", re.compile(r"\b(run|execute) (the )?(script|code|command)\b|"
                         r"\bpython\b|\bbash\b|\bterminal\b|```", re.I)),
    ("email", re.compile(r"\b(email|e-mail|mail|inbox|reply to|compose)\b", re.I)),
    ("payment", re.compile(r"\b(pay|purchase|checkout|order|charge|card|"
                           r"invoice|refund)\b", re.I)),
    ("calendar", re.compile(r"\b(calendar|schedule|book|reschedul|appointment|"
                            r"slot|meeting|reserve)\b", re.I)),
    ("home_control", re.compile(r"\b(thermostat|light|vacuum|door lock|blinds|"
                                r"speaker volume|actuate|turn (on|off)|set the "
                                r"\w+ to)\b", re.I)),
    ("camera", re.compile(r"\b(photo|camera|scan the|take a picture)\b", re.I)),
    ("location", re.compile(r"\b(directions|navigate to|nearby|my location|"
                            r"how far|route)\b", re.I)),
]

_MODALITY_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("screen", re.compile(r"\b(show|display|render|view|look at|read the "
                          r"(chart|table|graph|screenshot)|screenshot|"
                          r"click|scroll|the page)\b", re.I)),
    ("keyboard", re.compile(r"\b(type|enter|fill (in|out)|paste|compose a "
                            r"(long|detailed))\b", re.I)),
    ("large_text", re.compile(r"\b(read (the )?(full|entire|whole) "
                              r"(document|paper|report|transcript)|"
                              r"compare .{0,20}(tables|documents))\b", re.I)),
]

_SCOPE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("mail.send", re.compile(r"\b(send|reply|forward)\b.{0,20}\b(email|mail|"
                             r"message)\b|\bemail\b.{0,10}\b(to|him|her|them)\b", re.I)),
    ("pay.charge", re.compile(r"\b(pay|purchase|checkout|charge|place (the )?"
                              r"order|buy)\b", re.I)),
    ("home.write", re.compile(r"\b(set|turn (on|off)|actuate|schedule) (the )?"
                              r"(thermostat|light|vacuum|lock|blinds|\w+)\b", re.I)),
    ("files.write", re.compile(r"\b(write|save|export|download)\b.{0,20}\b"
                               r"(file|document|csv|report|output)\b", re.I)),
    ("calendar.write", re.compile(r"\b(book|schedule|reserve|create .{0,10}"
                                  r"(event|appointment))\b", re.I)),
]


def requirements_of(step_text: str) -> Requirement:
    """Capabilities a pending plan step needs, from its text."""
    r = Requirement()
    for cap, rx in _TOOL_PATTERNS:
        if rx.search(step_text):
            r.tools.add(cap)
    for cap, rx in _MODALITY_PATTERNS:
        if rx.search(step_text):
            r.modalities.add(cap)
    for cap, rx in _SCOPE_PATTERNS:
        if rx.search(step_text):
            r.scopes.add(cap)
    return r


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
#: Modality substitutions that preserve the action and change the channel.
_SUBSTITUTIONS = {
    ("screen", "audio_out"): "read the result aloud instead of displaying it",
    ("keyboard", "audio_in"): "dictate instead of typing",
}


@dataclass
class RepairedStep:
    text: str
    verdict: Verdict
    missing: List[str] = field(default_factory=list)
    note: str = ""


def repair_step(step_text: str, m: CapabilityManifest) -> RepairedStep:
    """Classify one pending step against a manifest, rewriting where possible."""
    req = requirements_of(step_text)

    missing_scopes = sorted(req.scopes - m.scopes)
    if missing_scopes:
        # A missing scope is checked first and never downgraded to DEFER: the
        # target is not merely unable, it is unauthorised, and the two need
        # different responses from the user.
        return RepairedStep(step_text, Verdict.BLOCKED, missing_scopes,
                            "missing credential scope")

    missing_tools = sorted(req.tools - m.tools)
    if missing_tools:
        if m.source_reachable or m.cloud_reachable:
            where = "source" if m.source_reachable else "cloud"
            return RepairedStep(step_text, Verdict.DELEGATE, missing_tools,
                                f"execute on {where}, return result")
        return RepairedStep(step_text, Verdict.DEFER, missing_tools,
                            "no reachable executor")

    missing_modes = sorted(req.modalities - m.modalities)
    if missing_modes:
        for miss in missing_modes:
            for have in m.modalities:
                sub = _SUBSTITUTIONS.get((miss, have))
                if sub:
                    return RepairedStep(f"{step_text} [{sub}]",
                                        Verdict.SUBSTITUTE, [miss], sub)
        if m.source_reachable or m.cloud_reachable:
            return RepairedStep(step_text, Verdict.DELEGATE, missing_modes,
                                "no equivalent modality; delegate")
        return RepairedStep(step_text, Verdict.DEFER, missing_modes,
                            "no equivalent modality")

    return RepairedStep(step_text, Verdict.FEASIBLE)


_PLAN_SPLIT = re.compile(r"(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+")


def split_plan(plan_text: str, cap: int = 12) -> List[str]:
    """Pending plan text -> individual steps, best-effort."""
    if not plan_text:
        return []
    parts = [p.strip() for p in _PLAN_SPLIT.split(plan_text) if p.strip()]
    if len(parts) <= 1:
        parts = [s.strip() for s in re.split(r"(?<=[.;])\s+", plan_text)
                 if len(s.strip()) > 12]
    return parts[:cap]


def repair_plan(plan_text: str, m: CapabilityManifest) -> List[RepairedStep]:
    return [repair_step(s, m) for s in split_plan(plan_text)]


def digest_pointer_status(m: CapabilityManifest) -> str:
    """Whether ``full@ptr`` pointers in the digest can be faulted in.

    When the source is unreachable the pointers are dead weight, and saying
    so in the block is what stops the target from planning around a fetch
    that will never complete.
    """
    return "faultable" if m.source_reachable else "unfaultable (source down)"
