"""
market_data/features.py

Trade-flow imbalance features computed from normalized TRADE events
(see market_data/models.py, market_data/binance_trades.py).

TWO WAYS TO USE THIS MODULE
----------------------------

1. add_rolling_imbalance(df, ...)
   Adds imbalance columns to every row of a trade DataFrame. Useful for
   exploring how imbalance behaves generally -- descriptive research,
   plotting, distribution checks.

2. imbalance_at(df, as_of_ms, ...)
   Computes imbalance as of one specific point in time (e.g. a strategy's
   entry_time). This is the function that matters for attaching a feature
   to a real trade signal without look-ahead: it only ever looks at trades
   with timestamp <= as_of_ms.

CAUSALITY
---------
Both window modes are backward-looking (trailing) by construction:

- TIME windows: pandas' `.rolling(window="60s")` on a sorted, time-indexed
  series is right-closed and trailing -- the window for row i covers
  (t_i - 60s, t_i], never anything after t_i.

- COUNT windows: `.rolling(window=N)` over row position is likewise
  trailing -- covers the N most recent rows up to and including row i.

Neither mode uses information from trades that occur after the point being
evaluated. This must remain true after any future edits to this file.

DEFINITIONS
-----------
signed_qty  = +quantity if side == BUY, -quantity if side == SELL
delta       = sum(signed_qty) over the window
volume      = sum(quantity) over the window          (unsigned, total)
imbalance   = delta / volume                          (range -1..+1)
              +1 = all aggressor buying, -1 = all aggressor selling, 0 = balanced

This is a genuine aggressor-side calculation (real exchange-reported side),
not an OHLCV-derived proxy -- see binance_trades.py docstring.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _add_signed_qty(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signed_qty"] = np.where(df["side"] == "BUY", df["quantity"], -df["quantity"])
    return df


def add_rolling_imbalance(
    df: pd.DataFrame,
    time_window: Optional[str] = None,
    trade_window: Optional[int] = None,
    min_trades: int = 1,
) -> pd.DataFrame:
    """
    Add rolling imbalance columns to a normalized trades DataFrame.

    Parameters
    ----------
    df : DataFrame with at least ['timestamp' (epoch ms), 'quantity', 'side']
    time_window : e.g. "60s", "5min" -- pandas offset alias. If given, adds
        columns suffixed _t{time_window}: delta_t60s, volume_t60s, imbalance_t60s
    trade_window : e.g. 500 -- number of trailing trades. If given, adds
        columns suffixed _n{trade_window}: delta_n500, volume_n500, imbalance_n500
    min_trades : minimum number of trades required in the window for the
        imbalance to be reported; below this, imbalance is set to NaN rather
        than a possibly-meaningless +-1 from a tiny sample. Default 1 means
        no filtering (matches prior behavior). Set this higher (e.g. 20-30)
        on bursty data where many windows contain very few trades -- check
        the n_trades_* column's distribution first to pick a sensible value.

    At least one of time_window / trade_window must be given. Both can be
    given at once to compute both simultaneously.

    Returns a new DataFrame (does not mutate the input), sorted by timestamp.
    """
    if time_window is None and trade_window is None:
        raise ValueError("Provide time_window and/or trade_window.")

    required = {"timestamp", "quantity", "side"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input trades DataFrame missing columns: {missing}")

    work = _add_signed_qty(df)
    work = work.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    if time_window is not None:
        dt_index = pd.to_datetime(work["timestamp"], unit="ms", utc=True)
        indexed = work.set_index(dt_index)

        delta = indexed["signed_qty"].rolling(time_window).sum()
        volume = indexed["quantity"].rolling(time_window).sum()
        n_trades = indexed["quantity"].rolling(time_window).count()

        suffix = f"t{time_window}"
        work[f"delta_{suffix}"] = delta.to_numpy()
        work[f"volume_{suffix}"] = volume.to_numpy()
        work[f"n_trades_{suffix}"] = n_trades.to_numpy()
        work[f"imbalance_{suffix}"] = np.where(
            (work[f"volume_{suffix}"] > 0) & (work[f"n_trades_{suffix}"] >= min_trades),
            work[f"delta_{suffix}"] / work[f"volume_{suffix}"],
            np.nan,
        )

    if trade_window is not None:
        delta = work["signed_qty"].rolling(trade_window, min_periods=1).sum()
        volume = work["quantity"].rolling(trade_window, min_periods=1).sum()
        n_trades = work["quantity"].rolling(trade_window, min_periods=1).count()

        suffix = f"n{trade_window}"
        work[f"delta_{suffix}"] = delta
        work[f"volume_{suffix}"] = volume
        work[f"n_trades_{suffix}"] = n_trades
        work[f"imbalance_{suffix}"] = np.where(
            (work[f"volume_{suffix}"] > 0) & (work[f"n_trades_{suffix}"] >= min_trades),
            work[f"delta_{suffix}"] / work[f"volume_{suffix}"],
            np.nan,
        )

    return work.drop(columns=["signed_qty"])


def imbalance_at(
    df: pd.DataFrame,
    as_of_ms: int,
    time_window_ms: Optional[int] = None,
    trade_window: Optional[int] = None,
    min_trades: int = 1,
) -> dict:
    """
    Causal, point-in-time imbalance lookup: "what was the imbalance
    immediately before/at as_of_ms?"

    Only trades with timestamp <= as_of_ms are used -- this is the function
    to call when attaching a feature to a real strategy signal's entry_time,
    since it can never see a trade that happens after the decision point.

    Parameters
    ----------
    df : normalized trades DataFrame, ['timestamp', 'quantity', 'side']
    as_of_ms : epoch ms cutoff (e.g. a strategy trade's entry_time)
    time_window_ms : look back this many milliseconds from as_of_ms
    trade_window : OR look back this many trades (trailing count), whichever
        window type is given. Exactly one of time_window_ms / trade_window
        must be provided.
    min_trades : minimum trades required in the window; below this,
        imbalance is NaN rather than a possibly-meaningless +-1 (see
        add_rolling_imbalance's docstring for why this matters on bursty
        data).

    Returns
    -------
    dict with: delta, volume, n_trades, imbalance (NaN if no trades in window
    or fewer than min_trades trades in window)
    """
    if (time_window_ms is None) == (trade_window is None):
        raise ValueError("Provide exactly one of time_window_ms or trade_window.")

    causal = df[df["timestamp"] <= as_of_ms]
    if causal.empty:
        return {"delta": np.nan, "volume": np.nan, "n_trades": 0, "imbalance": np.nan}

    causal = _add_signed_qty(causal.sort_values("timestamp", kind="mergesort"))

    if time_window_ms is not None:
        lower_bound = as_of_ms - time_window_ms
        window = causal[causal["timestamp"] > lower_bound]
    else:
        window = causal.tail(trade_window)

    if window.empty:
        return {"delta": np.nan, "volume": np.nan, "n_trades": 0, "imbalance": np.nan}

    delta = float(window["signed_qty"].sum())
    volume = float(window["quantity"].sum())
    n_trades = int(len(window))
    if volume > 0 and n_trades >= min_trades:
        imbalance = delta / volume
    else:
        imbalance = np.nan

    return {"delta": delta, "volume": volume, "n_trades": n_trades, "imbalance": imbalance}