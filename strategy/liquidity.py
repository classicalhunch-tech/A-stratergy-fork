"""
strategy/liquidity.py

Liquidity Engine:
    Detects and manages Sell-Side Liquidity (SSL) and Buy-Side Liquidity (BSL)
    levels, tracking their lifecycle from formation to sweep.
"""

from enum import Enum
from typing import List, Optional
import pandas as pd

from strategy.swings import Swing, SwingType


class LiquidityType(Enum):
    SELL_SIDE = "sell_side"  # SSL (formed at swing lows)
    BUY_SIDE = "buy_side"    # BSL (formed at swing highs)


class LiquidityStatus(Enum):
    ACTIVE = "active"
    SWEPT = "swept"
    INVALIDATED = "invalidated"


class LiquidityLevel:
    """Represents a structural liquidity pool in price action."""

    def __init__(
        self,
        liquidity_type: LiquidityType,
        price: float,
        formed_at: pd.Timestamp,
        swing: Swing,
        status: LiquidityStatus = LiquidityStatus.ACTIVE,
        sweep_bar_index: Optional[int] = None,
        swept_at: Optional[pd.Timestamp] = None,
    ):
        self.liquidity_type = liquidity_type
        self.price = price
        self.formed_at = formed_at
        self.swing = swing
        self.status = status
        self.sweep_bar_index = sweep_bar_index
        self.swept_at = swept_at

    def __repr__(self) -> str:
        return (
            f"LiquidityLevel(type={self.liquidity_type.value}, "
            f"price={self.price}, status={self.status.value}, "
            f"formed_at={self.formed_at})"
        )


def detect_liquidity_levels(
    df: pd.DataFrame,
    swings: List[Swing],
    lookback_bars: int = 50,
) -> List[LiquidityLevel]:
    """
    Detects liquidity levels from confirmed swing points.

    Args:
        df: OHLC DataFrame indexed by timestamp (DatetimeIndex), same
            convention as swings.py / structure.py / zones.py.
        swings: List of Swing objects detected on the chart.
        lookback_bars: Maximum historical window to track active liquidity.

    Returns:
        List of LiquidityLevel objects.
    """
    liquidity_levels = []

    for swing in swings:
        if swing.swing_type == SwingType.LOW:
            # Swing Lows act as Sell-Side Liquidity (SSL) pools.
            # formed_at uses confirmed_at, not formed_at on the Swing itself --
            # a liquidity pool shouldn't exist until the swing that created it
            # is actually confirmed (same causal principle as the rest of the
            # pipeline: nothing is "real" before its own confirmation point).
            liquidity_levels.append(
                LiquidityLevel(
                    liquidity_type=LiquidityType.SELL_SIDE,
                    price=swing.price,
                    formed_at=swing.confirmed_at,
                    swing=swing,
                    status=LiquidityStatus.ACTIVE,
                )
            )
        elif swing.swing_type == SwingType.HIGH:
            # Swing Highs act as Buy-Side Liquidity (BSL) pools
            liquidity_levels.append(
                LiquidityLevel(
                    liquidity_type=LiquidityType.BUY_SIDE,
                    price=swing.price,
                    formed_at=swing.confirmed_at,
                    swing=swing,
                    status=LiquidityStatus.ACTIVE,
                )
            )

    return liquidity_levels


def update_liquidity_sweeps(
    df: pd.DataFrame,
    liquidity_levels: List[LiquidityLevel],
) -> List[LiquidityLevel]:
    """
    Evaluates price action against active liquidity levels to determine
    if a sweep has occurred.

    Args:
        df: OHLC DataFrame indexed by timestamp (DatetimeIndex), same
            convention as swings.py / structure.py / zones.py.
        liquidity_levels: List of existing LiquidityLevel objects.

    Returns:
        Updated list of LiquidityLevel objects.

    PERFORMANCE NOTE (fixed 2026-09):
        The previous implementation used `for ts, row in sub_df.iterrows()`
        to find the first bar that swept each level. Because this function
        is called fresh every bar from run_backtest() on the full visible
        history, and iterrows() creates a new pandas Series object per row,
        this made the sweep-detection step alone scale roughly O(n^3) across
        a full backtest and made real datasets (thousands of candles)
        effectively never finish, despite passing on small unit-test
        fixtures.

        This version finds the exact same "first bar strictly after
        formed_at where the sweep condition is true" using a vectorized
        boolean mask + Series.idxmax(), which returns the index of the
        first occurrence of the max value in a boolean Series (True > False,
        so the first True wins) -- identical semantics to the old
        break-on-first-match loop, computed in C rather than row-by-row
        Python. The `mask.any()` guard is required because idxmax() on an
        all-False (or empty) mask would otherwise silently return the first
        index even though no sweep occurred.

        No behavioral change: for every level, this returns the same
        status, sweep_bar_index, and swept_at as the original loop would.
    """
    df_copy = df.copy()
    df_copy.columns = [c.lower() for c in df_copy.columns]
    df_copy = df_copy.sort_index()

    # Map each timestamp to its positional bar index, matching the same
    # pattern zones.py uses for index_pos -- sweep_bar_index must be an
    # integer bar position, not a raw Timestamp.
    index_pos = {ts: pos for pos, ts in enumerate(df_copy.index)}

    for level in liquidity_levels:
        if level.status != LiquidityStatus.ACTIVE:
            continue

        # Only bars strictly after the level was formed can sweep it --
        # a level can't be swept by the same bar (or an earlier bar) that
        # formed it.
        sub_df = df_copy[df_copy.index > level.formed_at]

        if sub_df.empty:
            continue

        if level.liquidity_type == LiquidityType.SELL_SIDE:
            # SSL is swept if price drops below the swing low.
            mask = sub_df["low"] < level.price
        elif level.liquidity_type == LiquidityType.BUY_SIDE:
            # BSL is swept if price exceeds the swing high.
            mask = sub_df["high"] > level.price
        else:
            continue

        if not mask.any():
            continue

        # First True in a boolean Series == first bar where the sweep
        # condition holds, same bar the original loop would have broken on.
        first_sweep_ts = mask.idxmax()

        level.status = LiquidityStatus.SWEPT
        level.sweep_bar_index = index_pos[first_sweep_ts]
        level.swept_at = first_sweep_ts

    return liquidity_levels