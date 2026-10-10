import math

import numpy as np

from tools.deflated_sharpe import (deflated_sharpe, expected_max_z, norm_cdf,
                                   norm_ppf, sharpe_se, sharpe_stats)


def _sample(mean=0.1, sd=1.0, n=400, seed=0):
    r = np.random.default_rng(seed).normal(mean, sd, n)
    return r


def test_norm_roundtrip():
    for p in (0.001, 0.1, 0.5, 0.9, 0.999):
        assert abs(norm_cdf(norm_ppf(p)) - p) < 1e-6


def test_hand_computed_case():
    # normal returns, SR=0.2, T=101, N=10
    sr, t = 0.2, 101
    se = sharpe_se(sr, t, 0.0, 0.0)
    assert abs(se - math.sqrt((1 + 0.5 * sr ** 2) / (t - 1))) < 1e-12
    emax = expected_max_z(10)
    assert abs(emax - 1.5746) < 0.01  # known approx for N=10 with this estimator
    sr0 = math.sqrt(1 / (t - 1)) * emax
    expected = norm_cdf((sr - sr0) / se)
    x = _sample()
    res = deflated_sharpe(x, 10)
    s = sharpe_stats(x)
    se2 = sharpe_se(s["sharpe"], s["T"], s["skew"], s["excess_kurt"])
    sr0b = math.sqrt(1 / (s["T"] - 1)) * emax
    assert abs(res["dsr"] - norm_cdf((s["sharpe"] - sr0b) / se2)) < 1e-9
    assert 0 < expected < 1


def test_n1_equals_psr_vs_zero():
    x = _sample()
    res = deflated_sharpe(x, 1)
    assert res["sr0"] == 0.0
    assert abs(res["dsr"] - norm_cdf(res["sharpe"] / res["se"])) < 1e-12


def test_more_trials_lowers_dsr():
    x = _sample(0.15)
    assert deflated_sharpe(x, 100)["dsr"] < deflated_sharpe(x, 5)["dsr"] < deflated_sharpe(x, 1)["dsr"]


def test_larger_sample_raises_dsr():
    small = deflated_sharpe(_sample(0.15, n=100, seed=1)[:100], 20)
    # same per-trade SR, more observations: repeat the sample
    base = _sample(0.15, n=100, seed=1)
    big = deflated_sharpe(np.tile(base, 4), 20)
    assert big["dsr"] > small["dsr"]


def test_degenerate_inputs():
    assert math.isnan(deflated_sharpe([], 5)["dsr"])
    assert math.isnan(deflated_sharpe([1.0], 5)["dsr"])
    assert math.isnan(deflated_sharpe([0.5, 0.5, 0.5], 5)["dsr"])
    r = deflated_sharpe([1.0, np.nan, -1.0, 2.0, 0.5], 3)
    assert r["T"] == 4 and not math.isnan(r["dsr"])


def test_effective_trials_overrides_n_trials():
    x = _sample()
    r1 = deflated_sharpe(x, n_trials=80)
    r2 = deflated_sharpe(x, n_trials=80, effective_trials=10)
    assert r2["dsr"] > r1["dsr"]  # less penalization
    assert r2["dsr"] == deflated_sharpe(x, n_trials=10)["dsr"]
    assert r2["n_trials"] == 80 and r2["effective_trials"] == 10
