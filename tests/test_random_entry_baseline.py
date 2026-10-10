import numpy as np
import pandas as pd
import pytest

from tools.random_entry_baseline import baseline, check_guards, main, verdict


def _pool(n_days=120, per_day=6, edge=0.0, seed=0):
    """Pool of candidate trades; the 'signal' takes 2 trades per day.

    If edge > 0, the signal's trades really do earn `edge` more R.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)
        for j in range(per_day):
            rows.append(dict(
                entry_time=day + pd.Timedelta(minutes=5 * j),
                net_r=rng.normal(0, 1),
                direction="long" if j % 2 == 0 else "short",
                is_signal=1 if j < 2 else 0,
            ))
    df = pd.DataFrame(rows)
    df.loc[df["is_signal"] == 1, "net_r"] += edge
    return df


def test_no_edge_gives_unremarkable_p_values():
    # With no real edge, p-values should be spread out, and only about 5% of
    # pools should look "significant" by luck. Check many pools, not one.
    ps = [baseline(_pool(edge=0.0, seed=s), n_reps=300, seed=1)["p_value"]
          for s in range(20)]
    assert float(np.median(ps)) > 0.2
    assert sum(p < 0.05 for p in ps) <= 4  # expected about 1 of 20
    res = baseline(_pool(edge=0.0, seed=1), n_reps=300, seed=1)
    assert res["n_signal"] == 240 and res["n_pool"] == 720


def test_real_edge_is_detected():
    res = baseline(_pool(edge=0.5), n_reps=500, seed=1)
    assert res["p_value"] < 0.01
    assert res["edge_vs_random"] > 0.3
    assert "beats" in verdict(res)


def test_same_seed_is_reproducible():
    a = baseline(_pool(), n_reps=200, seed=7)
    b = baseline(_pool(), n_reps=200, seed=7)
    assert a == b


def test_stratification_removes_drift():
    # Days differ hugely in mean return (drift). The signal takes the best-day
    # trades only by being on good days, but within each day it is random.
    rng = np.random.default_rng(3)
    rows = []
    for d in range(100):
        day_mean = 5.0 if d % 2 == 0 else -5.0
        for j in range(6):
            rows.append(dict(
                entry_time=pd.Timestamp("2026-01-01") + pd.Timedelta(days=d, minutes=5 * j),
                net_r=day_mean + rng.normal(0, 1),
                is_signal=1 if (d % 2 == 0 and j < 2) else 0))
    res = baseline(pd.DataFrame(rows), n_reps=500, seed=1)
    # Signal only on good days, so a day-matched baseline shows no timing skill.
    # (Strata with no signal trades draw nothing, so the baseline also uses only
    # good days.)
    assert abs(res["edge_vs_random"]) < 0.3


def test_direction_strata_columns_are_used():
    res = baseline(_pool(), strata_cols=["direction"], n_reps=200, seed=1)
    assert res["n_strata"] == 240  # 120 days x 2 directions


def test_missing_column_raises():
    df = _pool().drop(columns=["net_r"])
    with pytest.raises(ValueError):
        baseline(df, n_reps=10)


def test_all_zero_signal_raises():
    df = _pool()
    df["is_signal"] = 0
    with pytest.raises(ValueError):
        baseline(df, n_reps=10)


def test_all_signal_raises():
    df = _pool()
    df["is_signal"] = 1
    with pytest.raises(ValueError):
        baseline(df, n_reps=10)


def test_guard_dev_requires_exploratory():
    with pytest.raises(ValueError):
        check_guards("dev", False, None, 500)
    check_guards("dev", True, None, 500)  # no error


def test_guard_holdout_requires_n_star_and_enough_trades():
    with pytest.raises(ValueError):
        check_guards("holdout", False, None, 500)
    with pytest.raises(ValueError):
        check_guards("holdout", False, 300, 299)
    check_guards("holdout", False, 300, 300)  # no error


def test_guard_rejects_unknown_data_label():
    with pytest.raises(ValueError):
        check_guards("test", True, None, 10)


def test_cli_guards_exit_2_and_exploratory_runs():
    import os, tempfile
    path = os.path.join(tempfile.mkdtemp(), "pool.csv")
    _pool(n_days=30).to_csv(path, index=False)
    base = ["--pool", path, "--n-reps", "20", "--seed", "1"]
    assert main(base + ["--data-used", "dev"]) == 2                          # no --exploratory
    assert main(base + ["--data-used", "holdout"]) == 2                      # no --n-star
    assert main(base + ["--data-used", "holdout", "--n-star", "9999"]) == 2  # too few signals
    assert main(base + ["--data-used", "dev", "--signal-col", "nope"]) == 2  # bad column
    assert main(base + ["--data-used", "dev", "--exploratory"]) == 0
