import pandas as pd
import pytest

from strategy.backtest import run_backtest
from strategy.signals import SignalType
from strategy.structure_targets import (
    StructureTargetConfig,
    StructureTargets,
    build_structure_target_fn,
)
from strategy.swings import Swing, SwingType

T = pd.Timestamp("2026-01-05 12:00")
H = pd.Timedelta(hours=1)


def swing(kind, price, confirmed_at):
    return Swing(
        swing_type=kind,
        price=price,
        formed_at=confirmed_at - H,
        confirmed_at=confirmed_at,
    )


def high(price, confirmed_at):
    return swing(SwingType.HIGH, price, confirmed_at)


def low(price, confirmed_at):
    return swing(SwingType.LOW, price, confirmed_at)


def test_long_uses_nearest_swing_high_above_fill():
    st = StructureTargets([high(110.0, T - H), high(104.0, T - 2 * H)])
    assert st.target(SignalType.LONG, 100.0, 98.0, T) == 104.0


def test_short_uses_nearest_swing_low_below_fill():
    st = StructureTargets([low(90.0, T - H), low(96.0, T - 2 * H)])
    assert st.target(SignalType.SHORT, 100.0, 102.0, T) == 96.0


def test_future_swings_are_never_used():
    st = StructureTargets([high(120.0, T + H)])
    assert st.target(SignalType.LONG, 100.0, 98.0, T) is None


def test_swing_confirmed_exactly_at_fill_time_is_usable():
    st = StructureTargets([high(105.0, T)])
    assert st.target(SignalType.LONG, 100.0, 98.0, T) == 105.0


def test_old_swings_are_ignored():
    st = StructureTargets([high(130.0, T - pd.Timedelta(days=5))])
    assert st.target(SignalType.LONG, 100.0, 98.0, T) is None


def test_min_rr_skips_by_default_and_advances_when_asked():
    swings = [high(104.0, T - 2 * H), high(110.0, T - H)]

    # risk 4: nearest swing is only 1.0R away
    skip = StructureTargets(swings)
    assert skip.target(SignalType.LONG, 100.0, 96.0, T) is None

    advance = StructureTargets(
        swings, StructureTargetConfig(advance_to_min_rr=True)
    )
    assert advance.target(SignalType.LONG, 100.0, 96.0, T) == 110.0


def test_fill_on_wrong_side_of_stop_returns_none():
    st = StructureTargets([high(110.0, T - H), low(90.0, T - H)])
    assert st.target(SignalType.LONG, 98.0, 100.0, T) is None
    assert st.target(SignalType.SHORT, 102.0, 100.0, T) is None


def test_config_validation():
    with pytest.raises(ValueError):
        StructureTargetConfig(min_rr=0)
    with pytest.raises(ValueError):
        StructureTargetConfig(max_age=pd.Timedelta(0))


def test_backtest_structure_anchor_validation():
    df = pd.DataFrame(
        {"open": [1.0] * 6, "high": [1.1] * 6, "low": [0.9] * 6, "close": [1.0] * 6},
        index=pd.date_range("2026-01-01", periods=6, freq="5min"),
    )
    with pytest.raises(ValueError):
        run_backtest(df, fill_mode="market_on_close", target_anchor="structure")
    with pytest.raises(ValueError):
        run_backtest(df, target_anchor="structure", target_fn=lambda *a: None)


def test_structure_trades_respect_min_rr():
    df = pd.read_csv("data/master_historical_data.csv", index_col=0, parse_dates=True)
    fn = build_structure_target_fn(df)
    result = run_backtest(
        df,
        fill_mode="market_on_close",
        target_anchor="structure",
        target_fn=fn,
    )
    assert not result.errors
    for t in result.trades:
        assert abs(t.take_profit - t.fill_price) / t.initial_risk >= 1.5 - 1e-6
