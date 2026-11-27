"""Paired significance testing.

Every policy comparison in this paper is *paired*: the same trajectory and
the same handoff position are handed to every policy, so the informative
statistic is the per-trajectory difference, not two marginal means with
overlapping intervals.  This module provides:

  * ``paired_bootstrap`` -- the mean paired difference with a 95% percentile
    bootstrap CI over trajectories (the resampling unit, not evaluations);
  * ``wilcoxon`` -- the exact/normal-approximation signed-rank test on the
    same differences, which assumes nothing about their distribution;
  * ``cliffs_delta`` -- a non-parametric effect size, reported alongside the
    p-value so that a significant-but-tiny difference is visible as such;
  * ``holm`` -- Holm-Bonferroni adjustment across the family of comparisons
    made against Relay, so that reporting six baselines does not inflate the
    family-wise error rate.
"""
import numpy as np
from scipy import stats as sps


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    """Mean of (a - b) with a 95% percentile bootstrap CI over pairs."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    if len(d) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank p-value on the paired differences."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    if len(d) < 5 or np.allclose(d, 0):
        return 1.0
    try:
        return float(sps.wilcoxon(a, b, zero_method="wilcox",
                                  alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def cliffs_delta(a, b):
    """Cliff's delta for paired samples treated as two ordered samples."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    # rank-based O(n log n) form
    all_v = np.concatenate([a, b])
    r = sps.rankdata(all_v)
    ra = r[:len(a)].sum()
    u = ra - len(a) * (len(a) + 1) / 2.0
    return float(2.0 * u / (len(a) * len(b)) - 1.0)


def magnitude(delta):
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (n - i) * p[j])
        adj[j] = min(1.0, run)
    return [float(v) for v in adj]


def compare(target, others: dict, n_boot=10000, seed=0):
    """Compare ``target`` (list of per-trajectory values) against every entry
    of ``others``; returns rows with paired delta, CI, Holm-adjusted p and
    effect size."""
    rows, praw = [], []
    for name, vals in others.items():
        n = min(len(target), len(vals))
        a, b = target[:n], vals[:n]
        m, lo, hi = paired_bootstrap(a, b, n_boot, seed)
        p = wilcoxon(a, b)
        d = cliffs_delta(a, b)
        praw.append(p)
        rows.append({"vs": name, "delta": m, "ci": [lo, hi], "p_raw": p,
                     "cliffs_delta": d, "effect": magnitude(d), "n": n})
    for r, pa in zip(rows, holm(praw)):
        r["p_holm"] = pa
    return rows
