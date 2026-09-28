"""
market_data/tests/test_causality.py

These tests exist specifically to catch a future edit that accidentally
introduces look-ahead into the feature pipeline -- the single most
important property this whole module must preserve (see NO LOOK-AHEAD
rule in the project spec).

Method: take a trade dataset, compute a feature/value at a point in time,
then APPEND MORE FUTURE TRADES to the dataset and recompute. If the
earlier value changes, something is looking into the future.

Run with: pytest market_data/tests/test_causality.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_data.features import add_rolling_imbalance, imbalance_at


def _synthetic_trades(n, seed, start_ts=1_700_000_000_000):
    rng = np.random.default_rng(seed)
    timestamps = start_ts + np.sort(rng.integers(0, 200_000, n))
    sides = rng.choice(["BUY", "SELL"], n)
    qty = rng.uniform(0.001, 1.0, n)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": 50_000.0,
            "quantity": qty,
            "side": sides,
            "trade_id": range(n),
        }
    )


def test_imbalance_at_unaffected_by_future_trades():
    df = _synthetic_trades(n=300, seed=1)
    cutoff_ts = int(df["timestamp"].iloc[150])

    result_before = imbalance_at(df, as_of_ms=cutoff_ts, trade_window=50)

    # Append 300 more trades, all timestamped after the current max --
    # these must never affect the earlier result.
    future = _synthetic_trades(n=300, seed=2, start_ts=int(df["timestamp"].max()) + 1)
    df_extended = pd.concat([df, future], ignore_index=True)

    result_after = imbalance_at(df_extended, as_of_ms=cutoff_ts, trade_window=50)

    assert result_before == result_after, (
        "imbalance_at changed after appending future trades -- look-ahead bug"
    )


def test_imbalance_at_unaffected_by_future_trades_time_window():
    df = _synthetic_trades(n=300, seed=3)
    cutoff_ts = int(df["timestamp"].iloc[150])

    result_before = imbalance_at(df, as_of_ms=cutoff_ts, time_window_ms=30_000)

    future = _synthetic_trades(n=300, seed=4, start_ts=int(df["timestamp"].max()) + 1)
    df_extended = pd.concat([df, future], ignore_index=True)

    result_after = imbalance_at(df_extended, as_of_ms=cutoff_ts, time_window_ms=30_000)

    assert result_before == result_after


def test_rolling_imbalance_row_unaffected_by_appended_future_rows():
    df = _synthetic_trades(n=300, seed=5)
    enriched_before = add_rolling_imbalance(df, trade_window=50, time_window="30s")

    mid_ts = int(df["timestamp"].iloc[150])
    row_before = enriched_before[enriched_before["timestamp"] == mid_ts]
    assert not row_before.empty
    val_before_n = row_before["imbalance_n50"].iloc[0]
    val_before_t = row_before["imbalance_t30s"].iloc[0]

    future = _synthetic_trades(n=300, seed=6, start_ts=int(df["timestamp"].max()) + 1)
    df_extended = pd.concat([df, future], ignore_index=True)
    enriched_after = add_rolling_imbalance(df_extended, trade_window=50, time_window="30s")

    row_after = enriched_after[enriched_after["timestamp"] == mid_ts]
    assert not row_after.empty
    val_after_n = row_after["imbalance_n50"].iloc[0]
    val_after_t = row_after["imbalance_t30s"].iloc[0]

    assert np.isclose(val_before_n, val_after_n, equal_nan=True), (
        "count-window imbalance changed after appending future rows -- look-ahead bug"
    )
    assert np.isclose(val_before_t, val_after_t, equal_nan=True), (
        "time-window imbalance changed after appending future rows -- look-ahead bug"
    )


def test_imbalance_at_only_uses_trades_up_to_cutoff_inclusive():
    # Construct a case where we know exactly which trades should be included.
    df = pd.DataFrame(
        {
            "timestamp": [1000, 2000, 3000, 4000, 5000],
            "price": [50_000.0] * 5,
            "quantity": [1.0, 1.0, 1.0, 1.0, 1.0],
            "side": ["BUY", "BUY", "SELL", "BUY", "SELL"],
            "trade_id": [0, 1, 2, 3, 4],
        }
    )

    # as_of exactly at timestamp 3000 with a trade-count window of 3
    # should include trades at 1000, 2000, 3000 only (not 4000, 5000).
    result = imbalance_at(df, as_of_ms=3000, trade_window=3)
    # signed sum: +1 +1 -1 = 1, volume 3 -> imbalance = 1/3
    assert result["n_trades"] == 3
    assert np.isclose(result["imbalance"], 1 / 3)