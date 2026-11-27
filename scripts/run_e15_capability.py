"""E15: how often does capability -- not context -- actually bind?

The first version of this system called itself capability-aware while modelling
capability as exactly two numbers: usable context window and prefill
throughput.  Three reviewers of that version made the same objection, and it
is correct: a block that arrives perfectly sized and then asks a speaker to
read a chart, or a watch to run a shell command, or a phone to charge a card
it holds no scope for, has migrated the task onto a dead end.

This experiment asks whether that objection has empirical weight or is merely
tidy.  For every sampled handoff we take the pending plan the distiller
already ships, classify each of its steps against the target's capability
manifest (``state/capability.py``), and report:

  **E15a** the fraction of handoffs whose pending plan contains at least one
     step the target cannot execute as written, per device class -- the rate
     at which context-only sizing is not enough;
  **E15b** the distribution of repair verdicts over plan steps, which says
     *what kind* of gap it is: a modality that can be substituted, a tool that
     must be delegated back over the link, a scope that must be refused;
  **E15c** what repair costs -- the tokens the manifest and the annotations add
     to a wearable-budget block, and the fidelity that displaces;
  **E15d** how the picture changes when the source goes offline, which turns
     both delegation and digest-pointer faulting from available into absent.

The requirement model is regex-based and therefore has the same standing as
the side-effect detector: a deterministic, auditable heuristic whose errors
we report rather than a claim about ground truth.  We label the outputs
"capability-gap-like" for the same reason we label side effects
"side-effect-like".
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from explib import (load_all, sample_positions, real_fidelity, boot_ci,  # noqa
                    RES)
from state.capability import (DEFAULT_MANIFESTS, CapabilityManifest,     # noqa
                              Verdict, repair_plan, digest_pointer_status)
from state.schema import Distiller                                       # noqa
from policies.policies import relay_budget                               # noqa
from traces.parser import exact_tokens_cached                            # noqa

OUT = os.path.join(RES, "e15_capability.json")
SEEDS = [0, 1, 2, 3, 4]
CLASSES = ["watch", "speaker", "phone", "laptop"]


def _plans(trajs, idx, dist, limit_traj=None):
    """(traj, k, plan_text) for every sampled handoff whose block has a plan."""
    out = []
    for t in (trajs[:limit_traj] if limit_traj else trajs):
        for seed in SEEDS:
            for k in sample_positions(len(t.steps), t.tid, seed):
                if k >= len(t.steps):
                    continue
                block = dist.distill(t, k, relay_budget("watch"))
                if block.plan:
                    out.append((t, k, block.plan, block))
    return out


def _analyse(plans, manifests):
    """Verdict distribution and gap rate per device class."""
    per_class = {}
    for dev, m in manifests.items():
        verd = defaultdict(int)
        gap_flags, n_steps_l = [], []
        for _, _, plan, _ in plans:
            steps = repair_plan(plan, m)
            if not steps:
                continue
            gap = any(s.verdict is not Verdict.FEASIBLE for s in steps)
            gap_flags.append(1.0 if gap else 0.0)
            n_steps_l.append(len(steps))
            for s in steps:
                verd[s.verdict.value] += 1
        tot = sum(verd.values()) or 1
        mean, lo, hi = boot_ci(gap_flags)
        per_class[dev] = {
            "n_handoffs": len(gap_flags),
            "mean_plan_steps": round(sum(n_steps_l) / max(len(n_steps_l), 1), 2),
            "gap_rate": round(mean, 4), "gap_ci": [round(lo, 4), round(hi, 4)],
            "verdicts": {k: round(v / tot, 4) for k, v in sorted(verd.items())},
            "verdict_counts": dict(sorted(verd.items())),
        }
    return per_class


def e15c_cost(plans, idx, dist, dst="watch"):
    """Tokens the manifest and repair annotations add, and what they displace."""
    m = DEFAULT_MANIFESTS[dst]
    budget = relay_budget(dst)
    add_tok, base_tok, base_recall, cap_recall = [], [], [], []
    for t, k, plan, block in plans:
        base = block.render()
        steps = repair_plan(plan, m)
        ann = [f"- [{s.verdict.value}] {s.text}"
               + (f"  ({s.note}; missing {','.join(s.missing)})"
                  if s.missing else "")
               for s in steps]
        extra = ("\n== TARGET CAPABILITY ==\n" + m.render()
                 + f"\ntool_digest pointers: {digest_pointer_status(m)}"
                 + ("\n== PLAN (repaired) ==\n" + "\n".join(ann) if ann else ""))
        add_tok.append(exact_tokens_cached(extra))
        base_tok.append(exact_tokens_cached(base))
        # what the annotation displaces: re-distil at a budget reduced by the
        # annotation's cost, which is what a deployment would actually do
        A = idx.future_relevant(t, k)
        r0, _, n = real_fidelity(idx, t, k, base, anchors=A)
        shrunk = dist.distill(t, k, max(64, budget - add_tok[-1]))
        r1, _, _ = real_fidelity(idx, t, k, shrunk.render() + extra, anchors=A)
        base_recall.append(r0)
        cap_recall.append(r1)
    b, blo, bhi = boot_ci(base_recall)
    c, clo, chi = boot_ci(cap_recall)
    return {
        "n": len(add_tok),
        "median_added_tokens": sorted(add_tok)[len(add_tok) // 2] if add_tok else 0,
        "mean_added_tokens": round(sum(add_tok) / max(len(add_tok), 1), 1),
        "mean_block_tokens": round(sum(base_tok) / max(len(base_tok), 1), 1),
        "recall_without_capability": [round(b, 4), round(blo, 4), round(bhi, 4)],
        "recall_with_capability": [round(c, 4), round(clo, 4), round(chi, 4)],
        "recall_delta": round(c - b, 4),
    }


def run():
    import time
    t0 = time.time()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    dist = Distiller(idx)
    dist_s = Distiller(idx_s)

    real_plans = _plans(trajs, idx, dist)
    synth_plans = _plans(synth_trajs, idx_s, dist_s)

    out = {
        "e15a_b_real": _analyse(real_plans, DEFAULT_MANIFESTS),
        "e15a_b_synth": _analyse(synth_plans, DEFAULT_MANIFESTS),
        "e15c_cost_watch": e15c_cost(real_plans, idx, dist, "watch"),
        "n_real_plans": len(real_plans), "n_synth_plans": len(synth_plans),
    }

    # E15d: source offline -> delegation and pointer faulting both disappear
    offline = {d: CapabilityManifest(
        m.device, set(m.tools), set(m.modalities), set(m.scopes),
        source_reachable=False, cloud_reachable=False)
        for d, m in DEFAULT_MANIFESTS.items()}
    out["e15d_source_offline_real"] = _analyse(real_plans, offline)
    out["runtime_s"] = round(time.time() - t0, 1)

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"real plans: {out['n_real_plans']}  synth plans: {out['n_synth_plans']}")
    for tag in ("e15a_b_real", "e15a_b_synth", "e15d_source_offline_real"):
        print(f"\n{tag}")
        for dev in CLASSES:
            d = out[tag][dev]
            print(f"  {dev:8s} gap={d['gap_rate']:.3f} "
                  f"steps={d['mean_plan_steps']:.1f} {d['verdicts']}")
    print("\ncost:", json.dumps(out["e15c_cost_watch"]))
    print("->", OUT)


if __name__ == "__main__":
    run()
