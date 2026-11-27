"""Padding statistics for E23b (the budget rule), written to results/.

Recomputes, for the chosen P, the raw-prefix token count at every one of the
200 handoffs; stores min/median/max and how many exceed the watch window.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from explib import load_all, sample_positions, RES                  # noqa: E402
from policies.policies import usable                                # noqa: E402
from traces.parser import exact_tokens_cached                       # noqa: E402
from run_e23b_binding import padded, choose_P                       # noqa: E402


def main():
    trajs, idx, eps, synth, idx_s = load_all()
    P = choose_P(eps)
    toks = []
    for e in eps:
        k = sample_positions(e.n, e.eid, 0, 1)[0]
        t, imap = padded(e, P)
        toks.append(exact_tokens_cached(t.prefix_text(imap[k])))
    lim = usable("watch")
    out = {"pad_pairs_per_step": P, "watch_usable_window": lim,
           "n_handoffs": len(toks), "n_exceeding_window": int(sum(
               x > lim for x in toks)),
           "prefix_tokens_min": int(min(toks)),
           "prefix_tokens_median": float(np.median(toks)),
           "prefix_tokens_max": int(max(toks))}
    json.dump(out, open(os.path.join(RES, "e23b_padding.json"), "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
