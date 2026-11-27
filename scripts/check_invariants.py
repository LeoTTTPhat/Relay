"""Invariants the distiller must hold, checked over the whole corpus.

These are not statistics, they are the properties the paper's safety claims
assume, and they were added after one of them turned out to be false: the
exact-token trim loop was capped at 64 iterations, so a block with many
entity groups could exit *over* its budget, be truncated by the target's
drop-oldest cap, and lose the ledger -- the precise failure the mandatory
ledger exists to prevent. Every residual real-trace hazard we had measured
came from that bug. A claim of the form "never exceeds the budget" should be
enforced by a check, not by a comment, so this script runs in the reproduce
path and exits non-zero if any invariant breaks.

  I1  the block fits its budget, *or* the mandatory ledger alone exceeds it
      -- saturation, which E8c measures and which no allocator can fix
      without dropping action records
  I2  the block is never truncated at the target's usable window, so the
      ledger is never evicted by a drop-oldest cap
  I3  every side-effect-like prefix step has a ledger entry
  I4  every ledger entry keeps its fingerprint and idempotency key intact,
      whatever the budget did to its head
  I5  position sampling and the random-drop control reproduce across
      *processes*, not merely within one

I1 is stated as a disjunction on purpose. "Never exceeds the budget" is what
the first version of this paper claimed, and it is not achievable: a
wearable-budget block cannot hold forty action records with cryptographic
identifiers, and the honest options are to exceed the budget or to drop
records. We exceed the budget, report the saturation count, and rely on I2 --
the ledger is rendered last, so a drop-oldest cap trims prose and not
identity.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, sample_positions                     # noqa: E402
from state.schema import Distiller                                # noqa: E402
from policies.policies import relay_budget, usable                # noqa: E402
from traces.parser import exact_tokens_cached                     # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
TARGETS = ["watch", "speaker", "phone"]


def check(trajs, idx, label, dist):
    bad = {f"I{i}": 0 for i in range(1, 5)}
    worst = 0
    n = 0
    saturated = 0
    for t in trajs:
        for seed in SEEDS:
            for k in sample_positions(len(t.steps), t.tid, seed):
                if k >= len(t.steps):
                    continue
                want = sum(1 for s in t.steps[:k + 1] if s.side_effect_like)
                for dst in TARGETS:
                    b = relay_budget(dst)
                    for bud in (b, max(128, b // 2)):
                        # one distillation per case: the policy wrapper would
                        # redo exactly this work, and I2 follows from I1 by
                        # construction (the target caps at usable(dst) >= bud)
                        block = dist.distill(t, k, bud)
                        n += 1
                        tok = exact_tokens_cached(block.render())
                        worst = max(worst, tok - bud)
                        if tok > bud:
                            # saturation is allowed; anything else is a bug
                            minimal = all(not e.head for e in block.side_fx)
                            if minimal and block.side_fx:
                                saturated += 1
                            else:
                                bad["I1"] += 1
                        if tok > usable(dst):
                            bad["I2"] += 1
                        if len(block.side_fx) != want:
                            bad["I3"] += 1
                        for e in block.side_fx:
                            if not e.fingerprint or not e.idem_key:
                                bad["I4"] += 1
                                break
    print(f"[{label}] {n} blocks; {saturated} ledger-saturated "
          f"(worst overshoot {worst} tokens); "
          + " ".join(f"{k}={v}" for k, v in bad.items()))
    return sum(bad.values())


def check_cross_process_determinism():
    """I5: the same seeds must give the same sample in a different process.

    ``random.Random((seed, tid))`` seeds from ``hash()`` of a tuple, and
    string hashing is randomised per process unless PYTHONHASHSEED is set, so
    that construction drew *different* handoff positions on every run while
    looking seeded. The paper claims deterministic reruns, so the claim is
    checked here rather than assumed: we re-sample in a subprocess launched
    with a different PYTHONHASHSEED and require identical output.
    """
    import subprocess
    probe = (
        "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'scripts')\n"
        "from explib import sample_positions\n"
        "from policies.policies import p6_randdrop\n"
        "print([sample_positions(30,t,0,2) for t in "
        "('alggen-100','handcrafted-1','alggen-1')])\n")
    outs = []
    for hs in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                           text=True, env=env, cwd=os.path.join(
                               os.path.dirname(__file__), ".."))
        outs.append(r.stdout.strip())
    ok = len(set(outs)) == 1
    print(f"[determinism] sample_positions stable across PYTHONHASHSEED: {ok}")
    if not ok:
        for hs, o in zip(("0", "1", "12345"), outs):
            print(f"   PYTHONHASHSEED={hs}: {o}")
    return 0 if ok else 1


def check_pipeline_determinism():
    """I5b: the *distilled blocks* must not depend on PYTHONHASHSEED either.

    I5 only covered position sampling.  The 2026-09-16 revision found that
    entity anchors of equal length were ordered by ``set`` iteration, so block
    contents -- and occasionally recall -- changed with the hash seed.  This
    re-distils a handful of real handoffs in three subprocesses and requires
    byte-identical rendered blocks.
    """
    import subprocess
    probe = (
        "import sys,hashlib; sys.path.insert(0,'src'); sys.path.insert(0,'scripts')\n"
        "from explib import sample_positions, policy_budget\n"
        "from traces.parser import load_trajectories\n"
        "from traces.anchors import AnchorIndex\n"
        "from state.schema import Distiller\n"
        "ts = load_trajectories()[::23]\n"
        "d = Distiller(AnchorIndex(load_trajectories()))\n"
        "h = hashlib.sha256()\n"
        "for t in ts:\n"
        "    for k in sample_positions(len(t.steps), t.tid, 0):\n"
        "        h.update(d.distill(t, k, policy_budget('relay','watch')).render().encode())\n"
        "print(h.hexdigest())\n")
    outs = []
    for hs in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                           text=True, env=env, cwd=os.path.join(
                               os.path.dirname(__file__), ".."))
        outs.append(r.stdout.strip().splitlines()[-1] if r.stdout.strip()
                    else r.stderr[-200:])
    ok = len(set(outs)) == 1
    print(f"[determinism] distilled blocks stable across PYTHONHASHSEED: {ok}")
    if not ok:
        for hs, o in zip(("0", "1", "12345"), outs):
            print(f"   PYTHONHASHSEED={hs}: {o}")
    return 0 if ok else 1


def main():
    fails = check_cross_process_determinism()
    fails += check_pipeline_determinism()
    trajs, idx, eps, synth_trajs, idx_s = load_all()
    fails += check(trajs, idx, "real", Distiller(idx))
    fails += check(synth_trajs, idx_s, "synthetic", Distiller(idx_s))
    # conservative mode ships extra verbatim turns; it must keep I1-I4 too
    fails += check(trajs, idx, "real, conservative mode",
                   Distiller(idx, conservative=True))
    if fails:
        print(f"FAIL: {fails} invariant violations")
        sys.exit(1)
    print("all invariants hold")


if __name__ == "__main__":
    main()
