"""
strategy/swings.py

Swing detection: identifies confirmed structural swing highs and lows
from OHLC data, used as the foundation for market structure analysis.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd


class SwingType(Enum):
    HIGH = "HIGH"
    LOW = "LOW"


@dataclass
class Swing:
    swing_type: SwingType
    price: float
    formed_at: pd.Timestamp
    confirmed_at: pd.Timestamp


def find_swings(df: pd.DataFrame) -> list:
    """
    Detect confirmed structural swing highs and lows.
    Accepts DataFrame with either capitalized or lower-case OHLC columns.

    Raises:
        ValueError: if required OHLC columns are missing, if the index
            is not a DatetimeIndex, or if NaNs are present in high/low/close.

    NOTE: This function is unchanged. It remains the correctness
    reference that SwingState (below) is validated against. Do not
    modify this function as part of any incremental-state work --
    if the two ever disagree, this function is assumed correct and
    SwingState has a bug.
    """
    if df.empty or len(df) < 2:
        return []

    df_copy = df.copy()
    df_copy.columns = [col.lower() for col in df_copy.columns]

    required = {"open", "high", "low", "close"}
    if not required.issubset(df_copy.columns):
        raise ValueError(
            f"Missing required columns: {required - set(df_copy.columns)}"
        )

    if not isinstance(df_copy.index, pd.DatetimeIndex):
        raise ValueError(
            "find_swings requires a DatetimeIndex -- got "
            f"{type(df_copy.index).__name__}. Swing.formed_at/confirmed_at "
            "depend on real timestamps for downstream structure analysis."
        )

    if df_copy[["high", "low", "close"]].isna().any().any():
        raise ValueError(
            "find_swings received NaN values in high/low/close columns. "
            "Clean or interpolate the input data before swing detection."
        )

    # sort_index() already returns a fresh copy -- no need for an extra
    # .copy() before it (avoids a redundant full-DataFrame allocation).
    df_copy = df_copy.sort_index()

    swings = []

    # Convert once to raw NumPy arrays -- avoids all pandas per-call
    # accessor overhead.
    index_vals = df_copy.index
    highs = df_copy["high"].to_numpy()
    lows = df_copy["low"].to_numpy()
    closes = df_copy["close"].to_numpy()

    first_high = highs[0]
    first_low = lows[0]

    candidate_high = first_high
    candidate_high_time = index_vals[0]
    candidate_low = first_low
    candidate_low_time = index_vals[0]

    mode = "initial"
    pullback_low = None
    pullback_low_time = None
    pullback_high = None
    pullback_high_time = None

    n = len(df_copy)

    for i in range(1, n):
        ts = index_vals[i]
        high = highs[i]
        low = lows[i]
        close = closes[i]

        if mode == "initial":
            if close > first_high:
                candidate_low = low
                candidate_low_time = ts
                pullback_high = None
                pullback_high_time = None
                mode = "seeking_low"
                continue
            if close < first_low:
                candidate_high = high
                candidate_high_time = ts
                pullback_low = None
                pullback_low_time = None
                mode = "seeking_high"
                continue
            continue

        if mode == "seeking_high":
            if pullback_low is not None and close < pullback_low:
                swings.append(
                    Swing(
                        swing_type=SwingType.HIGH,
                        price=candidate_high,
                        formed_at=candidate_high_time,
                        confirmed_at=ts,
                    )
                )
                candidate_low = low
                candidate_low_time = ts
                pullback_high = None
                pullback_high_time = None
                mode = "seeking_low"
                continue
            if high > candidate_high:
                candidate_high = high
                candidate_high_time = ts
                pullback_low = None
                pullback_low_time = None
            else:
                if pullback_low is None or low < pullback_low:
                    pullback_low = low
                    pullback_low_time = ts
            continue

        if mode == "seeking_low":
            if pullback_high is not None and close > pullback_high:
                swings.append(
                    Swing(
                        swing_type=SwingType.LOW,
                        price=candidate_low,
                        formed_at=candidate_low_time,
                        confirmed_at=ts,
                    )
                )
                candidate_high = high
                candidate_high_time = ts
                pullback_low = None
                pullback_low_time = None
                mode = "seeking_high"
                continue
            if low < candidate_low:
                candidate_low = low
                candidate_low_time = ts
                pullback_high = None
                pullback_high_time = None
            else:
                if pullback_high is None or high > pullback_high:
                    pullback_high = high
                    pullback_high_time = ts
            continue

    return swings


# ============================================================
# INCREMENTAL SWING STATE (new -- added, does not replace
# find_swings() above)
# ============================================================
#
# SwingState carries forward exactly the loop-local variables
# find_swings() re-derives from scratch on every call. find_swings()
# is a pure forward-only state machine: at row i it only ever reads
# state carried from row i-1, never an earlier row directly, never a
# later row. That makes feeding it one new candle at a time through
# .step() provably equivalent to re-running the whole loop -- not a
# guess, a direct reading of the existing loop body, reproduced
# unchanged below inside step().
#
# WARNING: This class intentionally skips the whole-DataFrame
# validation find_swings() performs (missing columns, DatetimeIndex
# check, NaN check). Callers feed it individual, already-validated
# candle fields. If per-candle validation is needed, add it at the
# call site, not inside step().
#
# CORRECTNESS REQUIREMENT: verified by
# phase_02_optimization/test_swing_state_equivalence.py before any
# adapter change is made. Do not assume equivalence from reading the
# code alone.
# ============================================================


class SwingState:
    """
    Incremental, causal swing detector.

    Feed candles one at a time via step(). Internally reproduces the
    find_swings() loop body exactly, carrying state across calls
    instead of rebuilding it from the full history each time.
    """

    def __init__(self) -> None:
        self._initialized = False

        self.mode = "initial"

        self.first_high: Optional[float] = None
        self.first_low: Optional[float] = None

        self.candidate_high: Optional[float] = None
        self.candidate_high_time = None
        self.candidate_low: Optional[float] = None
        self.candidate_low_time = None

        self.pullback_high: Optional[float] = None
        self.pullback_high_time = None
        self.pullback_low: Optional[float] = None
        self.pullback_low_time = None

    def step(
        self,
        ts: pd.Timestamp,
        high: float,
        low: float,
        close: float,
    ) -> Optional[Swing]:
        """
        Process exactly one new causal candle.

        Returns a newly confirmed Swing if this candle confirmed one,
        otherwise None. Mirrors find_swings()'s loop body exactly --
        at most one Swing can be confirmed per candle.
        """

        if not self._initialized:
            self.first_high = high
            self.first_low = low
            self.candidate_high = high
            self.candidate_high_time = ts
            self.candidate_low = low
            self.candidate_low_time = ts
            self._initialized = True
            return None

        if self.mode == "initial":
            if close > self.first_high:
                self.candidate_low = low
                self.candidate_low_time = ts
                self.pullback_high = None
                self.pullback_high_time = None
                self.mode = "seeking_low"
                return None
            if close < self.first_low:
                self.candidate_high = high
                self.candidate_high_time = ts
                self.pullback_low = None
                self.pullback_low_time = None
                self.mode = "seeking_high"
                return None
            return None

        if self.mode == "seeking_high":
            if (
                self.pullback_low is not None
                and close < self.pullback_low
            ):
                confirmed = Swing(
                    swing_type=SwingType.HIGH,
                    price=self.candidate_high,
                    formed_at=self.candidate_high_time,
                    confirmed_at=ts,
                )
                self.candidate_low = low
                self.candidate_low_time = ts
                self.pullback_high = None
                self.pullback_high_time = None
                self.mode = "seeking_low"
                return confirmed

            if high > self.candidate_high:
                self.candidate_high = high
                self.candidate_high_time = ts
                self.pullback_low = None
                self.pullback_low_time = None
            else:
                if (
                    self.pullback_low is None
                    or low < self.pullback_low
                ):
                    self.pullback_low = low
                    self.pullback_low_time = ts
            return None

        if self.mode == "seeking_low":
            if (
                self.pullback_high is not None
                and close > self.pullback_high
            ):
                confirmed = Swing(
                    swing_type=SwingType.LOW,
                    price=self.candidate_low,
                    formed_at=self.candidate_low_time,
                    confirmed_at=ts,
                )
                self.candidate_high = high
                self.candidate_high_time = ts
                self.pullback_low = None
                self.pullback_low_time = None
                self.mode = "seeking_high"
                return confirmed

            if low < self.candidate_low:
                self.candidate_low = low
                self.candidate_low_time = ts
                self.pullback_high = None
                self.pullback_high_time = None
            else:
                if (
                    self.pullback_high is None
                    or high > self.pullback_high
                ):
                    self.pullback_high = high
                    self.pullback_high_time = ts
            return None

        # Unreachable: mode is always one of the three above.
        return None