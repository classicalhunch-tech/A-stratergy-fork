import math

import numpy as np
import pandas as pd

from tools.session_vs_outside import (
    cost_from_spec,
    dedupe,
    fit_window_effect,
    in_window,
    ols_cluster,
)


def ts(text):
    return pd.Timestamp(text, tz="UTC")


def test_window_edges():
    # 2026-10-07 is a Wednesday
    assert not in_window(ts("2026-10-07 12:59"))
    assert in_window(ts("2026-10-07 13:00"))
    assert in_window(ts("2026-10-07 16:59"))
    assert not in_window(ts("2026-10-07 17:00"))


def test_window_weekend_excluded():
    # 2026-10-10 is a Saturday, 2026-10-11 a Sunday
    assert not in_window(ts("2026-10-10 14:00"))
    assert not in_window(ts("2026-10-11 14:00"))


def test_window_converts_other_timezones_to_utc():
    assert in_window(pd.Timestamp("2026-10-07 15:00", tz="Europe/Berlin"))      # 13:00 UTC in summer
    assert not in_window(pd.Timestamp("2026-10-07 13:00", tz="Europe/Berlin"))  # 11:00 UTC


def test_window_naive_timestamp_taken_as_utc():
    assert in_window(pd.Timestamp("2026-10-07 14:00"))
    assert not in_window(pd.Timestamp("2026-10-07 20:00"))


def row(symbol, when, direction, net=0.0, win=False):
    return {"symbol": symbol, "entry_time": ts(when), "direction": direction,
            "net_r": net, "in_window": win}


def test_dedupe_keeps_first_same_day_symbol_direction():
    rows = [
        row("gold", "2026-10-07 15:00", "LONG"),
        row("gold", "2026-10-07 09:00", "LONG"),
    ]
    kept = dedupe(rows)
    assert len(kept) == 1
    assert kept[0]["entry_time"] == ts("2026-10-07 09:00")


def test_dedupe_keeps_different_direction_symbol_or_day():
    rows = [
        row("gold", "2026-10-07 09:00", "LONG"),
        row("gold", "2026-10-07 10:00", "SHORT"),
        row("eurusd", "2026-10-07 10:00", "LONG"),
        row("gold", "2026-10-08 09:00", "LONG"),
    ]
    assert len(dedupe(rows)) == 4


def test_ols_hand_example():
    # means 2 and 4 -> coefficient 2.0; each point its own cluster -> CR1 SE sqrt(2/3)
    y = [1, 2, 3, 3, 4, 5]
    X = [[1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1]]
    beta, se = ols_cluster(y, X, ["a", "b", "c", "d", "e", "f"])
    assert math.isclose(beta[0], 2.0)
    assert math.isclose(beta[1], 2.0)
    assert math.isclose(se[1], math.sqrt(2.0 / 3.0), rel_tol=1e-9)


def test_ols_single_cluster_gives_nan_se():
    y = [1, 2, 3, 3, 4, 5]
    X = [[1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1]]
    beta, se = ols_cluster(y, X, ["same"] * 6)
    assert math.isclose(beta[1], 2.0)
    assert np.isnan(se[1])


def test_ols_clustering_widens_se_when_days_are_shared():
    y = [1, 2, 3, 3, 4, 5]
    X = [[1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1]]
    _, se_own = ols_cluster(y, X, ["a", "b", "c", "d", "e", "f"])
    _, se_shared = ols_cluster(y, X, ["x", "x", "y", "y", "z", "z"])
    assert se_shared[1] != se_own[1]


def test_fit_window_effect_pooled_with_symbol_adjustment():
    rows = [
        row("gold", "2026-10-05 09:00", "LONG", 1.0, False),
        row("gold", "2026-10-06 14:00", "LONG", 3.0, True),
        row("gold", "2026-10-07 09:00", "LONG", 1.0, False),
        row("gold", "2026-10-08 14:00", "LONG", 3.0, True),
        row("eurusd", "2026-10-05 09:00", "LONG", 11.0, False),
        row("eurusd", "2026-10-06 14:00", "LONG", 13.0, True),
        row("eurusd", "2026-10-07 09:00", "LONG", 11.0, False),
        row("eurusd", "2026-10-08 14:00", "LONG", 13.0, True),
    ]
    beta, _ = fit_window_effect(rows)
    assert math.isclose(beta[1], 2.0, rel_tol=1e-9)   # window effect, symbol level removed


def test_cost_specs():
    df = pd.DataFrame({"open": [1, 1], "high": [11, 21], "low": [1, 1], "close": [2000.0, 4000.0]})
    assert cost_from_spec("fixed:0.5", df) == 0.5
    assert math.isclose(cost_from_spec("range", df), 0.115 * 15.0)
    assert math.isclose(cost_from_spec("scaled:6000", df), 0.5 * 3000.0 / 6000.0)
