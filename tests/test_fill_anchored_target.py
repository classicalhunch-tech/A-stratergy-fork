import pandas as pd
import pytest

from strategy.backtest import _fill_anchored_target, run_backtest
from strategy.signals import SignalType


def test_long_target_is_multiple_of_real_risk():
    # fill 100, stop 96 -> risk 4 -> 2R target 108
    assert _fill_anchored_target(SignalType.LONG, 100.0, 96.0, 2.0) == pytest.approx(108.0)
    assert _fill_anchored_target(SignalType.LONG, 100.0, 96.0, 3.0) == pytest.approx(112.0)


def test_short_target_is_multiple_of_real_risk():
    # fill 100, stop 104 -> risk 4 -> 2R target 92
    assert _fill_anchored_target(SignalType.SHORT, 100.0, 104.0, 2.0) == pytest.approx(92.0)


def test_fill_on_wrong_side_of_stop_is_rejected():
    assert _fill_anchored_target(SignalType.LONG, 95.0, 96.0, 2.0) is None
    assert _fill_anchored_target(SignalType.LONG, 96.0, 96.0, 2.0) is None
    assert _fill_anchored_target(SignalType.SHORT, 105.0, 104.0, 2.0) is None


def test_target_anchor_validation():
    df = pd.DataFrame(
        {"open": [1.0] * 6, "high": [1.1] * 6, "low": [0.9] * 6, "close": [1.0] * 6},
        index=pd.date_range("2026-01-01", periods=6, freq="5min"),
    )
    with pytest.raises(ValueError):
        run_backtest(df, target_anchor="nonsense")
    with pytest.raises(ValueError):
        run_backtest(df, target_anchor="fill")  # needs market_on_close


def test_zone_anchor_is_default_behaviour():
    df = pd.read_csv("data/master_historical_data.csv", index_col=0, parse_dates=True)
    a = run_backtest(df, fill_mode="market_on_close")
    b = run_backtest(df, fill_mode="market_on_close", target_anchor="zone")
    assert [t.take_profit for t in a.trades] == [t.take_profit for t in b.trades]


def test_fill_anchored_trades_have_exact_reward_to_risk():
    df = pd.read_csv("data/master_historical_data.csv", index_col=0, parse_dates=True)
    result = run_backtest(
        df, reward_multiple=2.0, fill_mode="market_on_close", target_anchor="fill"
    )
    assert not result.errors
    for t in result.trades:
        assert abs(t.take_profit - t.fill_price) == pytest.approx(2.0 * t.initial_risk)
