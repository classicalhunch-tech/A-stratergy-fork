"""
orderflow/research.py

Turns a list of EnrichedTrade into a flat research table (pandas
DataFrame / CSV) for baseline-vs-order-flow analysis, plus a small
descriptive comparison helper.

Nothing here filters trades, sets thresholds, or alters strategy
behavior in any way -- this module is reporting only. Per the
project rules: raw features first, thresholds/filtering only after
the relationship between order flow and outcome has been studied
out of sample.
"""

from typing import List

import pandas as pd

from orderflow.models import EnrichedTrade


def to_dataframe(trades: List[EnrichedTrade]) -> pd.DataFrame:
    """
    Flatten EnrichedTrade objects into one row per trade, matching
    the layout described in the project spec (signal, direction,
    setup/entry time, order-flow proxy values, result, R, bars held).
    """
    rows = []

    for t in trades:
        of = t.order_flow

        rows.append({
            "direction": t.direction,
            "setup_time": t.setup_time,
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "session": t.session,
            "sweep_type": t.sweep_type,
            "structure_break_type": t.structure_break_type,

            "result_status": t.result_status,
            "r_multiple": t.r_multiple,
            "bars_held": t.bars_held,

            "of_window_bars": of.window_bars if of else None,
            "of_total_volume": of.total_volume if of else None,
            "of_relative_volume": of.relative_volume if of else None,
            "of_delta_proxy": of.delta_proxy if of else None,
            "of_delta_proxy_pct": of.delta_proxy_pct if of else None,
            "of_trigger_bar_delta_proxy": (
                of.trigger_bar_delta_proxy if of else None
            ),
            "of_pressure_agrees_with_signal": (
                of.pressure_agrees_with_signal if of else None
            ),
            "of_volume_at_zone_pct": of.volume_at_zone_pct if of else None,
            "of_zone_matched": of.zone_matched if of else None,
            "of_data_source": of.data_source if of else None,
        })

    return pd.DataFrame(rows)


def save_csv(trades: List[EnrichedTrade], path: str) -> None:
    df = to_dataframe(trades)
    df.to_csv(path, index=False)


def summarize_by_pressure_agreement(trades: List[EnrichedTrade]) -> pd.DataFrame:
    """
    Descriptive baseline research view only: does result_status /
    r_multiple differ between trades where pressure_agrees_with_signal
    is True vs False vs unknown (None)?

    This is NOT a recommendation to filter on this feature -- it is
    step 3 of the project's research sequence ("analyze their
    relationship with outcomes"), before any threshold selection.
    """
    df = to_dataframe(trades)

    closed = df[df["result_status"].isin(["WIN", "LOSS"])].copy()

    if closed.empty:
        return pd.DataFrame()

    closed["pressure_bucket"] = (
        closed["of_pressure_agrees_with_signal"]
        .map({True: "agrees", False: "conflicts"})
        .fillna("unknown")
    )

    grouped = closed.groupby("pressure_bucket").agg(
        trade_count=("r_multiple", "count"),
        win_rate=("result_status", lambda s: (s == "WIN").mean() * 100.0),
        avg_r=("r_multiple", "mean"),
        total_r=("r_multiple", "sum"),
    ).reset_index()

    return grouped