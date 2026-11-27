"""What the resumed model is shown at an E23b handoff (no model involved).

For every one of the 200 padded E23b handoffs, render the blocks the policies
ship and record, per policy:

  ``plan_shipped``         Relay's block contains a plan section;
  ``plan_from_step0``      that plan is the one written at the episode's
                           first step (which lists every step of the task);
  ``plan_from_filler``     that plan comes from an inserted filler step;
  ``plan_step_before_last_done``  the plan was written before the last
                           pre-handoff world-changing action, so it lists at
                           least one action the ledger records as done;
  ``done_keys_shipped``    fraction of pre-handoff action keys present in the
                           shipped text (what the enforced dispatcher can see);
  ``done_first_offset``    relative position (0 = start, 1 = end) of the first
                           mention of a performed action's key in the block;
  ``plan0_shipped``        the step-0 plan's step list is in the shipped text
                           (any policy);
  ``plan0_offset``         relative position of that plan in the text.

Output: ``results/e23b_block_analysis.json``.  It explains, rather than
measures, the advisory behaviour in Table IV.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, sample_positions, policy_budget, RES   # noqa: E402
from policies.policies import POLICIES                              # noqa: E402
from state.schema import Distiller, PLAN_HINT_RE                    # noqa: E402
from run_e23b_binding import padded, choose_P, DST                  # noqa: E402
from run_e23_continuation import build_world                        # noqa: E402

POLS = ["full", "trunc25", "recency", "typed-recency", "randdrop", "relay",
        "restart"]


def main():
    trajs, idx, eps, synth, idx_s = load_all()
    P = choose_P(eps)
    dist = Distiller(idx_s)
    acc = {p: defaultdict(list) for p in POLS}
    for e in eps:
        k = sample_positions(e.n, e.eid, 0, 1)[0]
        traj, imap = padded(e, P)
        kp = imap[k]
        items, done_before, required = build_world(e, k)
        done = [it for it in items if it["kind"] == "effect"
                and it["id"] in done_before]
        # step (padded index) of the last pre-handoff world-changing action
        last_done_step = max((imap[i] for i, s in enumerate(e.steps[:k + 1])
                              if s.side_effect_like), default=-1)
        plan_step = None
        for st in reversed(traj.steps[:kp + 1]):
            if PLAN_HINT_RE.search(st.content):
                plan_step = st.idx
                break
        for p in POLS:
            kw = {"distiller": dist} if p == "relay" else {}
            o = POLICIES[p](traj, kp, DST, seed=0,
                            budget=policy_budget(p, DST), **kw)
            text = o.resume_text
            low = text.lower()
            a = acc[p]
            if p == "relay":
                block = dist.distill(traj, kp, policy_budget("relay", DST))
                plan = block.plan
                a["plan_shipped"].append(float(bool(plan)))
                if plan:
                    orig = set(imap.values())
                    a["plan_from_step0"].append(float(plan_step == imap[0]))
                    a["plan_from_filler"].append(float(
                        plan_step is not None and plan_step not in orig))
                    a["plan_step_before_last_done"].append(float(
                        plan_step is not None and plan_step < last_done_step))
            plan0 = e.steps[0].content
            m0 = PLAN_HINT_RE.search(plan0)
            if m0:
                # the numbered step list of the step-0 plan, as the distiller
                # would quote it (first two list lines)
                lines = [ln.strip().lower() for ln in plan0.splitlines()
                         if ln.strip()[:2] in ("1.", "2.")]
                probe = lines[0] if lines else plan0[m0.start():][:40].lower()
                a["plan0_shipped"].append(float(probe in low))
                if probe in low:
                    a["plan0_offset"].append(low.find(probe) / max(len(low), 1))
            if done:
                keys = [d["key"].lower() for d in done]
                a["done_keys_shipped"].append(
                    float(np.mean([kk in low for kk in keys])))
                pos = [low.find(kk) for kk in keys if kk in low]
                if pos:
                    a["done_first_offset"].append(min(pos) / max(len(low), 1))
            a["tokens"].append(o.tokens_resume)
    out = {"pad_pairs_per_step": P, "n_episodes": len(eps),
           "per_policy": {p: {m: (round(float(np.mean(v)), 4) if v else None)
                              for m, v in d.items()}
                          for p, d in acc.items()},
           "n_with_plan_relay": len(acc["relay"]["plan_from_step0"])}
    json.dump(out, open(os.path.join(RES, "e23b_block_analysis.json"), "w"),
              indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
