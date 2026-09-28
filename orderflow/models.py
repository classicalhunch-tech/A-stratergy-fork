"""
orderflow/models.py

Data model for the order-flow / cluster confluence research layer.

This module defines pure data containers only. It has no dependency
on strategy/ internals beyond the field names it reads from
TradeResult / TradeSignal / Zone, and it computes nothing itself.

All fields documented as "proxy" or "estimated" are derived
exclusively from OHLCV data. No genuine bid/ask trade classification
is available in the current datasets, so none of these fields may be
described, logged, or reported as true order-flow delta.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class OrderFlowFeatures:
    """
    OHLCV-derived order-flow proxy features for a single trade setup,
    computed only from bars in [setup_bar_index, triggered_at_bar]
    (inclusive) -- i.e. from the liquidity sweep through the retest
    trigger candle. No bar after triggered_at_bar is ever read.
    """

    # Window bookkeeping
    window_bars: int
    window_start_time: Optional[pd.Timestamp]
    window_end_time: Optional[pd.Timestamp]

    # Volume aggregates over the window
    total_volume: float
    avg_volume: float

    # Relative volume: window avg vs a trailing lookback ending at
    # (and including) the trigger bar. Purely historical/causal.
    relative_volume: Optional[float]

    # OHLCV-derived buy/sell pressure proxy (NOT true bid/ask delta)
    est_buy_volume: float
    est_sell_volume: float
    delta_proxy: float                 # est_buy_volume - est_sell_volume
    delta_proxy_pct: Optional[float]   # delta_proxy / total_volume

    # Same proxy computed on just the retest/trigger candle
    trigger_bar_volume: Optional[float]
    trigger_bar_delta_proxy: Optional[float]

    # Does the proxy's sign agree with the signal's direction?
    pressure_agrees_with_signal: Optional[bool]

    # Share of window volume that traded while price overlapped the
    # matched zone's [price_bottom, price_top] band
    volume_at_zone_pct: Optional[float]

    # Housekeeping / provenance
    zone_matched: bool
    data_source: str = "ohlcv_proxy"   # always -- never "true_delta"


@dataclass
class EnrichedTrade:
    """
    One row of the order-flow research table: the existing strategy's
    trade outcome, unmodified, plus the order-flow proxy features
    attached alongside it. This never alters entry/SL/TP or the
    trade's result_status/r_multiple -- those are copied verbatim
    from strategy.backtest.TradeResult.
    """

    # --- identity / context, copied from TradeSignal/TradeResult ---
    direction: str                      # "LONG" / "SHORT"
    setup_time: Optional[pd.Timestamp]
    entry_time: Optional[pd.Timestamp]
    exit_time: Optional[pd.Timestamp]
    session: str
    sweep_type: Optional[str]
    structure_break_type: Optional[str]

    # --- existing strategy outcome, UNCHANGED ---
    result_status: str                  # WIN / LOSS / OPEN
    r_multiple: float
    bars_held: int

    # --- order-flow proxy features (None if window couldn't be built) ---
    order_flow: Optional[OrderFlowFeatures]