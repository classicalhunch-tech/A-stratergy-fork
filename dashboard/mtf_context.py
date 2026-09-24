"""
dashboard/mtf_context.py

Multi-timeframe (MTF) resampling + alignment bridge.

Purpose
-------
Generate higher-timeframe (15m / 1H / 4H) views from the base 5-minute
OHLC data, and attach them to each 5-minute execution bar WITHOUT
lookahead bias: every execution bar only ever sees higher-timeframe
candles that had FULLY CLOSED at or before that execution bar's evaluation time.

This is a data-layer bridge only. It does not touch strategy/backtest.py,
strategy/signals.py, strategy/structure.py, etc. Those keep evaluating
one DataFrame at a time -- they just now also receive optional HTF
context objects to check confluence against.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class MTFConfig:
    """Which higher timeframes to generate, in your top-down order."""
    macro_tf: str = "4h"        # narrative / extreme zones
    internal_tf: str = "15min"  # internal structure
    base_interval: str = "5min" # explicit base timeframe spacing
    origin: str = "start_day"


def resample_ohlc(df_5m: pd.DataFrame, rule: str, origin: str = "start_day") -> pd.DataFrame:
    """
    Resample 5-min open-timestamped OHLC into a higher timeframe using
    clean session/calendar boundaries.

    Using label='right' and closed='left' ensures the bucket covers [00:00, 04:00)
    and gets labeled at 04:00 (the close time), precisely capturing all 5-minute
    open timestamps from 00:00:00 through 03:55:00 while excluding 04:00:00.
    """
    if not isinstance(df_5m.index, pd.DatetimeIndex):
        raise TypeError("df_5m must have a DatetimeIndex.")

    required = {"open", "high", "low", "close"}
    missing = required - set(df_5m.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df_5m.columns:
        agg["volume"] = "sum"

    htf = (
        df_5m.resample(rule, label="right", closed="left", origin=origin)
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"])
    )
    htf.index.name = "close_time"
    return htf


def attach_htf_context(
    df_5m: pd.DataFrame,
    htf_df: pd.DataFrame,
    prefix: str,
    timestamp_is_open: bool = True,
    base_interval: str = "5min",
) -> pd.DataFrame:
    """
    Lookahead-safe join of a higher-timeframe DataFrame onto the base
    5-minute DataFrame.

    Parameters:
    - timestamp_is_open: True if your 5m CSV timestamps represent candle OPEN times.
      If True, we shift the match perspective to candle CLOSE times (timestamp + base_interval)
      so a candle starting at 10:00 only gains visibility to an HTF close once it actually completes.
    - base_interval: Explicit string duration representing base candle spacing (e.g., '5min').
    """
    if not isinstance(df_5m.index, pd.DatetimeIndex):
        raise TypeError("df_5m must have a DatetimeIndex.")

    if not isinstance(htf_df.index, pd.DatetimeIndex):
        raise TypeError("htf_df must have a DatetimeIndex.")

    if htf_df.empty:
        raise ValueError("htf_df has no rows -- nothing to attach.")

    # Reset by position, not by a hardcoded expected index name. Relying on
    # htf_df's index literally being named "close_time" only works when htf_df
    # comes straight from resample_ohlc; any other index name would make the
    # rename silently no-op and the merge_asof below raise a confusing KeyError
    # instead of working or failing clearly.
    base = df_5m.reset_index()
    base = base.rename(columns={base.columns[0]: "time"})

    htf = htf_df.reset_index()
    htf = htf.rename(columns={htf.columns[0]: "htf_close_time"})

    # If the base data timestamps represent OPEN times, the bar's economic completion
    # and availability for signal action happens at the close of that base interval.
    if timestamp_is_open:
        base["evaluation_time"] = base["time"] + pd.Timedelta(base_interval)
    else:
        base["evaluation_time"] = base["time"]

    base = base.sort_values("evaluation_time", kind="mergesort")  # stable sort
    htf = htf.sort_values("htf_close_time", kind="mergesort")

    merged = pd.merge_asof(
        base,
        htf,
        left_on="evaluation_time",
        right_on="htf_close_time",
        direction="backward",  # only match HTF candles already closed
        suffixes=("", f"_{prefix}"),
    )

    # Clean up column collisions safely using robust suffix mapping.
    # "volume" is included alongside OHLC -- an HTF volume column collides with the
    # base's own volume column during merge_asof and, without this, was left as the
    # pandas-generated "volume_<prefix>" instead of the intended "<prefix>_volume"
    # naming convention every other HTF column follows. Harmless to include when
    # volume isn't present -- the "if suffixed in merged.columns" guard skips it.
    for col in ["open", "high", "low", "close", "volume"]:
        suffixed = f"{col}_{prefix}"
        if suffixed in merged.columns:
            merged = merged.rename(columns={suffixed: f"{prefix}_{col}"})
        elif col in merged.columns and col not in df_5m.columns:
            merged = merged.rename(columns={col: f"{prefix}_{col}"})

    merged = merged.rename(columns={"htf_close_time": f"{prefix}_close_time"})

    # Drop temporary evaluation helper and restore clean index mapping
    if "evaluation_time" in merged.columns:
        merged = merged.drop(columns=["evaluation_time"])

    merged = merged.set_index("time")
    merged.index.name = df_5m.index.name

    return merged


def build_mtf_dataset(
    df_5m: pd.DataFrame,
    config: Optional[MTFConfig] = None,
    timestamp_is_open: bool = True,
) -> pd.DataFrame:
    """
    One-call entry point: takes the base 5-min OHLC DataFrame and
    returns it enriched with lookahead-safe macro_tf and internal_tf
    context columns.
    """
    config = config or MTFConfig()

    macro_htf = resample_ohlc(df_5m, config.macro_tf, origin=config.origin)
    internal_htf = resample_ohlc(df_5m, config.internal_tf, origin=config.origin)

    out = attach_htf_context(
        df_5m,
        macro_htf,
        prefix="macro",
        timestamp_is_open=timestamp_is_open,
        base_interval=config.base_interval
    )
    out = attach_htf_context(
        out,
        internal_htf,
        prefix="internal",
        timestamp_is_open=timestamp_is_open,
        base_interval=config.base_interval
    )

    return out