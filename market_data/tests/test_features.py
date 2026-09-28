"""
market_data/tests/test_features.py

Run with: pytest market_data/tests/test_features.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_data.features import add_rolling_imbalance, imbalance_at


def _synthetic_trades(n=200, seed=0):
    rng = np.random.default_rng(seed)
    base_ts = 1_700_000_000_000
    timestamps = base_ts + np.sort(rng.integers(0, 200_000, n))
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


def test_add_rolling_imbalance_matches_manual_calc():
    df = _synthetic_trades()
    enriched = add_rolling_imbalance(df, trade_window=50)

    last_50 = df.sort_values("timestamp").tail(50)
    signed = np.where(last_50["side"] == "BUY", last_50["quantity"], -last_50["quantity"])
    expected = signed.sum() / last_50["quantity"].sum()

    actual = enriched.sort_values("timestamp")["imbalance_n50"].iloc[-1]
    assert np.isclose(expected, actual)


def test_imbalance_bounded_between_neg1_and_1():
    df = _synthetic_trades()
    enriched = add_rolling_imbalance(df, trade_window=20, time_window="30s")
    for col in ["imbalance_n20", "imbalance_t30s"]:
        valid = enriched[col].dropna()
        assert (valid >= -1.0001).all()
        assert (valid <= 1.0001).all()


def test_all_buys_gives_imbalance_one():
    df = _synthetic_trades()
    df = df.copy()
    df["side"] = "BUY"
    enriched = add_rolling_imbalance(df, trade_window=10)
    assert np.allclose(enriched["imbalance_n10"].dropna(), 1.0)


def test_all_sells_gives_imbalance_negative_one():
    df = _synthetic_trades()
    df = df.copy()
    df["side"] = "SELL"
    enriched = add_rolling_imbalance(df, trade_window=10)
    assert np.allclose(enriched["imbalance_n10"].dropna(), -1.0)


def test_requires_at_least_one_window_type():
    df = _synthetic_trades()
    with pytest.raises(ValueError):
        add_rolling_imbalance(df)


def test_missing_columns_raises():
    df = pd.DataFrame({"timestamp": [1, 2, 3]})
    with pytest.raises(ValueError):
        add_rolling_imbalance(df, trade_window=2)


def test_imbalance_at_empty_window_returns_nan():
    df = _synthetic_trades()
    result = imbalance_at(df, as_of_ms=int(df["timestamp"].min()) - 1, trade_window=10)
    assert np.isnan(result["imbalance"])
    assert result["n_trades"] == 0


def test_imbalance_at_requires_exactly_one_window_arg():
    df = _synthetic_trades()
    with pytest.raises(ValueError):
        imbalance_at(df, as_of_ms=int(df["timestamp"].iloc[50]))
    with pytest.raises(ValueError):
        imbalance_at(df, as_of_ms=int(df["timestamp"].iloc[50]), time_window_ms=1000, trade_window=10)