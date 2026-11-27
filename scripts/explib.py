"""Shared experiment machinery: position sampling, fidelity metrics,
hazard counting, bootstrap CIs, caching."""
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from traces.parser import load_trajectories, est_tokens          # noqa: E402
from traces.anchors import (AnchorIndex, contains_anchor, norm_text)  # noqa: E402
from state.schema import Distiller                               # noqa: E402
from synth.generator import generate                             # noqa: E402

STRICT_TYPES = {"url", "file", "number", "answer"}
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


def sample_positions(traj_n: int, tid: str, seed: int, per: int = 2):
    """per handoff positions u.a.r. over steps 2..N-2 (deterministic)."""
    # Seeded with a *string*, not a tuple: ``random.Random`` hashes a tuple
    # with ``hash()``, and string hashing is randomised per process unless
    # PYTHONHASHSEED is set -- so ``Random((seed, tid))`` silently drew
    # different handoff positions in every run. A string seed is converted
    # deterministically (SHA-512) and reproduces across processes.
    rng = random.Random(f"{seed}|{tid}")
    lo, hi = 2, max(3, traj_n - 2)
    return sorted({rng.randrange(lo, hi) for _ in range(per)})


def policy_budget(pol: str, dst: str):
    """Budget kwarg for ``pol`` on ``dst``.

    ``relay-lite`` is the *half-budget* operating point of the same knob, so
    passing every policy one budget silently collapses it onto ``relay``.
    Experiments call this instead of hard-coding a single budget.
    """
    from policies.policies import relay_budget, relay_lite_budget
    return relay_lite_budget(dst) if pol == "relay-lite" else relay_budget(dst)


def real_fidelity(idx: AnchorIndex, traj, k: int, resume_text: str,
                  anchors=None):
    """(recall_all, recall_strict, n_anchors). anchors: precomputed A(T,k)."""
    A = anchors if anchors is not None else idx.future_relevant(traj, k)
    h = norm_text(resume_text)
    if not A:
        return 1.0, 1.0, 0
    hit_all = tot_all = hit_s = tot_s = 0
    for a, ty in A.items():
        tot_all += 1
        got = contains_anchor(h, a)
        hit_all += got
        if ty in STRICT_TYPES:
            tot_s += 1
            hit_s += got
    r_all = hit_all / tot_all
    r_s = hit_s / tot_s if tot_s else 1.0
    return r_all, r_s, tot_all


def probe_recall(episode, k: int, resume_text: str):
    """Synthetic probe recall (exact ground truth by construction)."""
    h = norm_text(resume_text)
    probes = episode.probes(k)
    hit = tot = 0
    per_field = {}
    for fname, targets in probes.items():
        f_hit = f_tot = 0
        for t in targets:
            tn = norm_text(str(t))
            if not tn:
                continue
            f_tot += 1
            ok = tn in h
            f_hit += ok
        hit += f_hit
        tot += f_tot
        if f_tot:
            per_field[fname] = f_hit / f_tot
    return (hit / tot if tot else 1.0), per_field


def hazards_real(traj, k: int, resume_text: str):
    """Side-effect-like prefix steps whose record (normalized head, 80 chars)
    is absent from the resumption context -> double-execution hazard."""
    h = norm_text(resume_text)
    tot = haz = 0
    for s in traj.steps[:k + 1]:
        if not s.side_effect_like:
            continue
        tot += 1
        head = norm_text(s.content)[:80]
        probe = head[:40]        # containment probe on the head
        if probe and probe not in h:
            haz += 1
    return haz, tot


def hazards_synth(episode, k: int, resume_text: str):
    """Exact: executed side-effects whose idempotency key is absent."""
    h = norm_text(resume_text)
    st = episode.states[k]
    tot = len(st.side_effects)
    haz = sum(1 for se in st.side_effects if se["key"].lower() not in h)
    return haz, tot


def boot_ci(values, n_boot=1000, seed=0):
    """Mean and 95% percentile bootstrap CI over units (trajectories)."""
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def load_all():
    trajs = load_trajectories()
    idx = AnchorIndex(trajs)
    eps = generate(200, seed=0)
    synth_trajs = [e.to_trajectory() for e in eps]
    idx_s = AnchorIndex.__new__(AnchorIndex)      # synthetic index: no DF cut
    idx_s.df = {}
    idx_s.n_trajs = len(synth_trajs)
    idx_s.df_cut = len(synth_trajs)               # keep everything
    return trajs, idx, eps, synth_trajs, idx_s
