import math

import numpy as np

from tools.block_bootstrap import block_bootstrap_ci


def test_ci_contains_sample_mean_and_covers_true_mean():
    # The CI is centred on the sample mean, so check that directly ...
    x = np.random.default_rng(1).normal(0.3, 1.0, 300)
    lo, hi = block_bootstrap_ci(x, 5, 3000, 0.10, seed=2)
    assert lo < x.mean() < hi
    # ... and that a nominal 90% CI covers the true mean most of the time.
    hits = 0
    for seed in range(100):
        d = np.random.default_rng(100 + seed).normal(0.3, 1.0, 200)
        l, h = block_bootstrap_ci(d, 5, 500, 0.10, seed=seed)
        hits += l <= 0.3 <= h
    assert hits >= 80


def test_block_size_one_matches_iid_bootstrap():
    x = np.random.default_rng(3).normal(0, 1, 200)
    lo, hi = block_bootstrap_ci(x, 1, 20000, 0.10, seed=4)
    se = x.std(ddof=1) / math.sqrt(len(x))
    assert abs((hi - lo) - 2 * 1.645 * se) < 0.03


def test_larger_block_widens_ci_when_clustered():
    # AR(1) with phi=0.8: positively autocorrelated, so iid bootstrap is too narrow
    rng = np.random.default_rng(5)
    e = rng.normal(0, 1, 600)
    x = np.zeros(600)
    for i in range(1, 600):
        x[i] = 0.8 * x[i - 1] + e[i]
    w1 = np.subtract(*block_bootstrap_ci(x, 1, 4000, 0.10, seed=6)[::-1])
    w10 = np.subtract(*block_bootstrap_ci(x, 10, 4000, 0.10, seed=6)[::-1])
    assert w10 > 1.5 * w1


def test_empty_array():
    lo, hi = block_bootstrap_ci([], 5, 100)
    assert math.isnan(lo) and math.isnan(hi)
