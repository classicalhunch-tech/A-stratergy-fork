"""
orderflow/attach.py

Attaches order-flow proxy features to an already-completed backtest,
without touching the existing strategy engine in any way.

This module:
    - re-derives zones from the SAME raw df + structural pipeline
      strategy/backtest.py already used (swings -> structure -> zones),
      purely to re-associate each TradeResult with its matched zone's
      price_top/price_bottom (TradeSignal only stores zone_id /
      zone_stable_key, not the boundaries themselves)
    - for each trade, restricts analysis to
      df.iloc[setup_bar_index : triggered_at_bar + 1] -- from the
      liquidity sweep through the retest trigger candle, inclusive
    - never reads a bar after triggered_at_bar for any feature
    - never modifies TradeResult, TradeSignal, Zone, or the original df

Nothing here is a second signal engine: no swings, structure breaks,
liquidity sweeps, or zones are redetected or redefined differently.
find_swings/analyze_structure/find_zones are the exact same functions
strategy/backtest.py already calls -- this is a second read-only pass
over the same causal pipeline, used only to build a lookup table, not
a parallel implementation of any of it.
"""

from typing import Dict, Hashable, List, Optional

import pandas as pd

from strategy.backtest import BacktestResult
from strategy.signals import SignalType, _stable_cache_key
from strategy.swings import find_swings
from strategy.structure import analyze_structure
from strategy.zones import Zone, find_zones

from orderflow.models import EnrichedTrade, OrderFlowFeatures
from orderflow.proxy import (
    relative_volume,
    volume_at_price_band,
    window_pressure,
)


# Trailing lookback (in bars) used for relative_volume. Purely
# historical -- ends at the trigger bar, never looks forward.
DEFAULT_RELATIVE_VOLUME_LOOKBACK = 50


def _build_zone_lookup(df: pd.DataFrame) -> Dict[Hashable, Zone]:
    """
    Re-run the same structural pipeline strategy/backtest.py already
    runs (swings -> structure -> zones) once, and index the resulting
    zones by their stable cache key so TradeSignal.zone_stable_key can
    be resolved back to a Zone with real price_top/price_bottom.
    """
    all_swings = find_swings(df)
    all_structure_breaks, _, _ = analyze_structure(df, all_swings)
    all_zones = find_zones(df, all_structure_breaks)

    lookup: Dict[Hashable, Zone] = {}

    for zone in all_zones:
        key = _stable_cache_key(zone)
        if key is not None:
            lookup[key] = zone

    return lookup


def _normalized_df(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working.columns = [str(c).strip().lower() for c in working.columns]
    working = working.sort_index()
    return working


def _compute_features(
    window_df: pd.DataFrame,
    lookback_df: pd.DataFrame,
    signal_type: SignalType,
    zone: Optional[Zone],
) -> OrderFlowFeatures:
    total_volume = float(window_df["volume"].astype(float).sum())
    avg_volume = (
        float(window_df["volume"].astype(float).mean()) if len(window_df) else 0.0
    )

    est_buy, est_sell = window_pressure(window_df)
    delta_proxy = est_buy - est_sell
    delta_proxy_pct = (delta_proxy / total_volume) if total_volume > 0 else None

    if len(window_df):
        trigger_row = window_df.iloc[-1]
        trig_buy, trig_sell = window_pressure(window_df.iloc[[-1]])
        trigger_bar_volume = float(trigger_row["volume"])
        trigger_bar_delta_proxy = trig_buy - trig_sell
    else:
        trigger_bar_volume = None
        trigger_bar_delta_proxy = None

    rel_vol = relative_volume(window_df, lookback_df)

    pressure_agrees: Optional[bool] = None
    if signal_type == SignalType.LONG:
        pressure_agrees = delta_proxy > 0
    elif signal_type == SignalType.SHORT:
        pressure_agrees = delta_proxy < 0

    zone_matched = zone is not None
    volume_at_zone_pct = None
    if zone is not None:
        volume_at_zone_pct = volume_at_price_band(
            window_df, zone.price_bottom, zone.price_top
        )

    return OrderFlowFeatures(
        window_bars=len(window_df),
        window_start_time=window_df.index[0] if len(window_df) else None,
        window_end_time=window_df.index[-1] if len(window_df) else None,
        total_volume=total_volume,
        avg_volume=avg_volume,
        relative_volume=rel_vol,
        est_buy_volume=est_buy,
        est_sell_volume=est_sell,
        delta_proxy=delta_proxy,
        delta_proxy_pct=delta_proxy_pct,
        trigger_bar_volume=trigger_bar_volume,
        trigger_bar_delta_proxy=trigger_bar_delta_proxy,
        pressure_agrees_with_signal=pressure_agrees,
        volume_at_zone_pct=volume_at_zone_pct,
        zone_matched=zone_matched,
    )


def enrich_trades(
    df: pd.DataFrame,
    backtest_result: BacktestResult,
    relative_volume_lookback: int = DEFAULT_RELATIVE_VOLUME_LOOKBACK,
) -> List[EnrichedTrade]:
    """
    Attach order-flow proxy features to every trade in an already-run
    BacktestResult.

    Parameters
    ----------
    df:
        The SAME raw OHLCV dataframe (must include a 'volume' column)
        that was passed to strategy.backtest.run_backtest().
    backtest_result:
        Output of strategy.backtest.run_backtest() -- read-only here.
    relative_volume_lookback:
        Trailing bar count used for the relative_volume feature.

    Returns
    -------
    List[EnrichedTrade]
        One entry per trade in backtest_result.trades, in the same
        order. Nothing in backtest_result is modified.
    """
    working = _normalized_df(df)

    if "volume" not in working.columns:
        raise ValueError(
            "df is missing a 'volume' column -- order-flow proxy "
            "features require volume."
        )

    zone_lookup = _build_zone_lookup(working)

    enriched: List[EnrichedTrade] = []

    for trade in backtest_result.trades:
        signal = trade.signal

        setup_idx = getattr(signal, "setup_bar_index", None)
        trigger_idx = getattr(signal, "triggered_at_bar", None)

        order_flow: Optional[OrderFlowFeatures] = None

        if setup_idx is not None and trigger_idx is not None:
            setup_idx = int(setup_idx)
            trigger_idx = int(trigger_idx)

            if 0 <= setup_idx <= trigger_idx < len(working):
                # Causal window: sweep bar through trigger bar,
                # inclusive. Never touches anything after trigger_idx.
                window_df = working.iloc[setup_idx: trigger_idx + 1]

                # Causal trailing lookback: ends at the trigger bar.
                lookback_start = max(
                    0, trigger_idx - relative_volume_lookback + 1
                )
                lookback_df = working.iloc[lookback_start: trigger_idx + 1]

                zone_key = getattr(signal, "zone_stable_key", None)
                zone = zone_lookup.get(zone_key) if zone_key is not None else None

                order_flow = _compute_features(
                    window_df=window_df,
                    lookback_df=lookback_df,
                    signal_type=signal.signal_type,
                    zone=zone,
                )

        direction = trade.direction
        direction_str = direction.value if hasattr(direction, "value") else str(direction)

        enriched.append(
            EnrichedTrade(
                direction=direction_str,
                setup_time=trade.setup_time,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                session=trade.session,
                sweep_type=getattr(signal, "sweep_type", None),
                structure_break_type=getattr(signal, "structure_break_type", None),
                result_status=trade.result_status,
                r_multiple=trade.r_multiple,
                bars_held=trade.bars_held,
                order_flow=order_flow,
            )
        )

    return enriched