"""Parser for the Who&When multi-agent failure-attribution corpus.

Loads both subsets (Hand-Crafted: Magentic-One/GAIA runs; Algorithm-Generated:
AutoGen runs) into a unified Trajectory representation. Standalone: no imports
from sibling projects.

Token accounting lives in ``tok.tokenizer``: a real Qwen2.5 BPE count is
used for every cost/window decision, and a conservative chars/3.5 estimate
is used inside the distiller's fill loop (see that module's docstring).
"""
import json
import glob
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from state.fingerprint import fp_action, fp_head
from tok.tokenizer import (est_tokens, exact_tokens,           # noqa: F401
                           exact_tokens_cached)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..",
                         "data", "who_and_when", "Who&When")

# ---------------------------------------------------------------------------
# Side-effect-like step detection (real traces).
# Honest framing: these are pattern-matched labels ("side-effect-like"), not
# verified external effects. Patterns target actions with plausible external
# consequence: code execution, form submission, file writes, sends/posts.
# ---------------------------------------------------------------------------
SIDE_EFFECT_PATTERNS = [
    r"exitcode:\s*\d+",                      # Computer_terminal executed code
    r"\bexecution succeeded\b",
    r"\bexecuting (the )?(code|script)\b",
    r"\bI (typed|clicked|pressed|submitted)\b",
    r"\bclicked (the )?['\"]?submit\b",
    r"\bsubmitted (the )?(form|query|request)\b",
    r"\b(sent|sending) (an? )?(email|message|request)\b",
    r"\bposted\b",
    r"\bwrote (the )?(file|output|results?) to\b",
    r"\bsaved (the )?(file|output|results?)\b",
    r"\bfile (was |has been )?(written|saved|created)\b",
    r"\bdownloaded\b",
    r"\bbooked\b",
    r"\bpurchased\b",
    r"\bplaced (the |an? )?order\b",
]
_SIDE_EFFECT_RE = re.compile("|".join(SIDE_EFFECT_PATTERNS), re.IGNORECASE)

# Tool-result-like steps: outputs of an acting agent (terminal, web surfer).
# Anchored at the start of the agent name on purpose: an orchestrator step
# labelled "Orchestrator (-> WebSurfer)" is a routing *instruction*, not a
# tool result, and treating it as one fills the digest with prose.
TOOL_ROLE_HINTS = re.compile(
    r"\s*(computer_?terminal|websurfer|filesurfer|web_surfer|file_surfer)\b",
    re.I)

@dataclass
class Step:
    idx: int
    role: str          # raw role string
    agent: str         # best-effort agent name
    content: str
    side_effect_like: bool
    is_tool_result: bool
    tokens: int

    def fingerprint(self, version: str = "v2") -> str:
        """Action fingerprint for the side-effect ledger.

        ``v2`` (default, the shipped design) hashes the acting agent, the
        side-effect verb class, and canonicalised argument evidence drawn
        from the whole step with volatile spans removed.  ``v1`` is the
        head-hash the first version of this system used; E8 compares them
        on both error directions (divergence and collision).
        """
        if version == "v1":
            return fp_head(self.agent, self.content)
        return fp_action(self.agent, self.content)

    def exact_tokens(self) -> int:
        return exact_tokens_cached(self.content)


@dataclass
class Trajectory:
    tid: str
    source: str        # "handcrafted" | "alggen"
    question: str
    ground_truth: str
    steps: List[Step] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.steps)

    def prefix_text(self, k: int) -> str:
        """Concatenated raw text of steps 0..k inclusive, plus question."""
        parts = [self.question]
        parts += [s.content for s in self.steps[:k + 1]]
        return "\n".join(parts)

    def suffix_text(self, k: int) -> str:
        return "\n".join(s.content for s in self.steps[k + 1:])

    def total_tokens(self) -> int:
        return est_tokens(self.question) + sum(s.tokens for s in self.steps)

    def prefix_tokens(self, k: int) -> int:
        """Exact BPE token count of the raw prefix (cost accounting)."""
        return exact_tokens_cached(self.prefix_text(k))


def _mk_step(idx: int, role: str, name: Optional[str], content: str) -> Step:
    agent = name or role or "unknown"
    content = content if isinstance(content, str) else json.dumps(content)
    se = bool(_SIDE_EFFECT_RE.search(content))
    tool = bool(TOOL_ROLE_HINTS.match(str(agent))) or bool(
        re.search(r"exitcode:|Code output:", content))
    return Step(idx=idx, role=role or "", agent=str(agent), content=content,
                side_effect_like=se, is_tool_result=tool,
                tokens=est_tokens(content))


def load_trajectories(root: str = DATA_ROOT, min_steps: int = 5) -> List[Trajectory]:
    trajs = []
    for sub, tag in (("Hand-Crafted", "handcrafted"),
                     ("Algorithm-Generated", "alggen")):
        for path in sorted(glob.glob(os.path.join(root, sub, "*.json")),
                           key=lambda p: int(os.path.basename(p)[:-5])):
            with open(path) as f:
                d = json.load(f)
            hist = d.get("history", [])
            if not isinstance(hist, list) or len(hist) < min_steps:
                continue
            gt = d.get("groundtruth", d.get("ground_truth", ""))
            steps = [_mk_step(i, s.get("role", ""), s.get("name"),
                              s.get("content", "")) for i, s in enumerate(hist)]
            trajs.append(Trajectory(
                tid=f"{tag}-{os.path.basename(path)[:-5]}",
                source=tag,
                question=str(d.get("question", "")),
                ground_truth=str(gt) if gt is not None else "",
                steps=steps))
    return trajs


if __name__ == "__main__":
    ts = load_trajectories()
    import statistics
    lens = [t.n for t in ts]
    toks = [t.total_tokens() for t in ts]
    se = sum(1 for t in ts for s in t.steps if s.side_effect_like)
    print(f"{len(ts)} trajectories; steps med {statistics.median(lens)} "
          f"mean {statistics.mean(lens):.1f} max {max(lens)}")
    print(f"tokens med {statistics.median(toks)} mean {statistics.mean(toks):.0f} "
          f"max {max(toks)}")
    print(f"side-effect-like steps: {se} "
          f"({se / sum(lens) * 100:.1f}% of steps); "
          f"traces with >=1: {sum(1 for t in ts if any(s.side_effect_like for s in t.steps))}")
