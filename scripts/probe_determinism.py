import sys, os, json
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from explib import sample_positions, real_fidelity, policy_budget
from traces.parser import load_trajectories
from traces.anchors import AnchorIndex
from state.schema import Distiller
from policies.policies import POLICIES
ts = load_trajectories()
sub = ts[::4][:20]
idx = AnchorIndex(sub)
dist = Distiller(idx)
out = []
for t in sub:
    for seed in (0, 1):
        for k in sample_positions(len(t.steps), t.tid, seed):
            A = idx.future_relevant(t, k)
            o = POLICIES['relay'](t, k, 'watch', distiller=dist, budget=policy_budget('relay','watch'))
            r, _, n = real_fidelity(idx, t, k, o.resume_text, anchors=A)
            out.append((t.tid, k, round(r, 6), n, o.tokens_resume))
print(json.dumps(out))
