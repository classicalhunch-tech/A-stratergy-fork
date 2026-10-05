"""
tests/test_strict_fill_backtest.py

strict_entry_fill must never ADD trades: a setup that fills only when
price reaches the entry price is a subset of the setups that fill when
price merely touches the zone edge. Signal discovery is unaffected.
"""

from pathlib import Path

import pandas as pd

from strategy.backtest import run_backtest

DATA = Path(__file__).resolve().parents[1] / "data" / "master_historical_data.csv"


def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    df.columns = [str(column).strip().lower() for column in df.columns]
    return df.tail(5000)


def test_strict_entry_fill_never_adds_trades():
    df = _load()

    loose = run_backtest(df)
    strict = run_backtest(df, strict_entry_fill=True)

    assert strict.total_signals_generated == loose.total_signals_generated
    assert strict.total_trades_triggered <= loose.total_trades_triggered
    assert len(strict.errors) <= len(loose.errors)


def test_adapter_accepts_strict_entry_fill_option():
    from phase_03_paper.signals.adapter import StrategyAdapter

    assert StrategyAdapter()._strict_entry_fill is False
    assert StrategyAdapter(strict_entry_fill=True)._strict_entry_fill is True
