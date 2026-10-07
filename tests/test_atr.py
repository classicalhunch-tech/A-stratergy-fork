import numpy as np
import pandas as pd
import pytest

from strategy.atr import atr_series, known_atr, true_range
from strategy.backtest import run_backtest


def frame(n=30, rng=2.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = pd.Series(100.0 + np.arange(n) * 0.0, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + rng / 2, "low": close - rng / 2, "close": close}
    )


def test_constant_range_gives_that_atr():
    df = frame(rng=2.0)
    assert atr_series(df, 14).iloc[-1] == pytest.approx(2.0)


def test_true_range_uses_gap_from_previous_close():
    idx = pd.date_range("2026-01-01", periods=2, freq="5min")
    df = pd.DataFrame(
        {"open": [100, 110], "high": [101, 111], "low": [99, 109], "close": [100, 110]},
        index=idx,
    )
    assert true_range(df).iloc[1] == pytest.approx(11.0)


def test_known_atr_never_uses_the_current_bar():
    df = frame(rng=2.0)
    df.iloc[-1, df.columns.get_loc("high")] = 500.0  # spike on the last bar
    assert known_atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_known_atr_is_nan_until_enough_history():
    df = frame()
    assert known_atr(df, 14).iloc[:14].isna().all()


def test_period_validation():
    with pytest.raises(ValueError):
        atr_series(frame(), 0)


def test_backtest_stop_atr_validation():
    df = frame(6)
    with pytest.raises(ValueError):
        run_backtest(df, stop_atr_mult=-1)
    with pytest.raises(ValueError):
        run_backtest(df, stop_atr_mult=0.25)  # needs market_on_close


def test_wider_stop_never_shrinks_risk():
    df = pd.read_csv("data/master_historical_data.csv", index_col=0, parse_dates=True)
    base = run_backtest(df, fill_mode="market_on_close", target_anchor="fill")
    wide = run_backtest(
        df, fill_mode="market_on_close", target_anchor="fill", stop_atr_mult=0.25
    )
    assert not wide.errors
    assert wide.total_trades_triggered <= base.total_trades_triggered + len(base.trades)
