"""
strategy/structure.py

Consumes confirmed Swings from strategy/swings.py and OHLC candles to track:
1. Swing Classifications: HH, HL, LH, LL
2. Trend Direction: BULLISH, BEARISH, UNKNOWN
3. Structural Breaks: INITIAL_BREAK, BOS (Continuation), and CHOCH (Reversal)
   with explicit protected swing boundaries and active BOS levels.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Set

import pandas as pd

from strategy.swings import SwingType, Swing


class Trend(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    UNKNOWN = "unknown"


class BreakType(Enum):
    INITIAL_BREAK = "initial_break"
    BOS = "bos"
    CHOCH = "choch"


class SwingLabel(Enum):
    INITIAL_HIGH = "INITIAL_HIGH"
    INITIAL_LOW = "INITIAL_LOW"
    HH = "HH"
    LH = "LH"
    HL = "HL"
    LL = "LL"


@dataclass
class ClassifiedSwing:
    swing: Swing
    label: SwingLabel

    def __repr__(self):
        return (
            f"{self.label.value}("
            f"{self.swing.swing_type.value}, "
            f"price={self.swing.price})"
        )


@dataclass
class StructureBreak:
    break_type: BreakType
    direction: Trend
    broken_swing: Swing
    broken_at: pd.Timestamp
    break_price: float

    def __repr__(self):
        return (
            f"StructureBreak("
            f"{self.break_type.value}, "
            f"direction={self.direction.value}, "
            f"level={self.broken_swing.price}, "
            f"broken_at={self.broken_at.strftime('%m-%d %H:%M')})"
        )


def analyze_structure(
    df: pd.DataFrame,
    swings: list,
) -> Tuple[List[StructureBreak], List[ClassifiedSwing], Trend]:
    """
    Evaluates candle data against confirmed swings using explicit structural
    boundaries, tracking active BOS levels and protected CHOCH reversal levels.
    """

    if df.empty or not swings:
        return [], [], Trend.UNKNOWN

    df_copy = df.copy()
    df_copy.columns = [c.lower() for c in df_copy.columns]
    df_copy = df_copy.sort_index()

    breaks: List[StructureBreak] = []
    classified_swings: List[ClassifiedSwing] = []
    current_trend = Trend.UNKNOWN

    # Explicit structural boundary markers.
    active_bos_swing: Optional[Swing] = None
    protected_swing: Optional[Swing] = None

    last_high: Optional[Swing] = None
    last_low: Optional[Swing] = None

    swings_sorted = sorted(swings, key=lambda s: s.confirmed_at)
    next_swing_idx = 0

    consumed_swings: Set[
        Tuple[SwingType, float, pd.Timestamp]
    ] = set()

    def get_swing_key(sw: Swing):
        return (
            sw.swing_type,
            sw.price,
            sw.confirmed_at,
        )

    def record_break(
        break_type,
        direction,
        broken_swing,
        ts,
        close,
    ):
        """Append a StructureBreak and mark its level as consumed."""

        breaks.append(
            StructureBreak(
                break_type=break_type,
                direction=direction,
                broken_swing=broken_swing,
                broken_at=ts,
                break_price=close,
            )
        )

        consumed_swings.add(
            get_swing_key(broken_swing)
        )

    # PERFORMANCE OPTIMIZATION:
    # Use itertuples() instead of iterrows() to reduce pandas Series
    # overhead during large backtest scans.
    for row in df_copy.itertuples():

        ts = row.Index
        close = row.close

        # ------------------------------------------------------------
        # 1. Unlock and classify confirmed swings.
        # ------------------------------------------------------------
        while (
            next_swing_idx < len(swings_sorted)
            and swings_sorted[next_swing_idx].confirmed_at <= ts
        ):
            sw = swings_sorted[next_swing_idx]

            if sw.swing_type == SwingType.HIGH:

                if last_high is None:
                    lbl = SwingLabel.INITIAL_HIGH
                elif sw.price > last_high.price:
                    lbl = SwingLabel.HH
                else:
                    lbl = SwingLabel.LH

                last_high = sw

                # In bearish structure, only tighten the protected
                # ceiling. A higher high must not weaken it.
                if current_trend == Trend.BEARISH:
                    if (
                        protected_swing is None
                        or sw.price < protected_swing.price
                    ):
                        protected_swing = sw

            else:

                if last_low is None:
                    lbl = SwingLabel.INITIAL_LOW
                elif sw.price < last_low.price:
                    lbl = SwingLabel.LL
                else:
                    lbl = SwingLabel.HL

                last_low = sw

                # In bullish structure, only raise the protected
                # floor. A lower low must not weaken it.
                if current_trend == Trend.BULLISH:
                    if (
                        protected_swing is None
                        or sw.price > protected_swing.price
                    ):
                        protected_swing = sw

            classified_swings.append(
                ClassifiedSwing(
                    swing=sw,
                    label=lbl,
                )
            )

            next_swing_idx += 1

            # Track the furthest structural extreme in the current
            # trend direction as the active BOS target.
            if (
                current_trend == Trend.BULLISH
                and sw.swing_type == SwingType.HIGH
            ):
                if (
                    active_bos_swing is None
                    or sw.price > active_bos_swing.price
                ):
                    active_bos_swing = sw

            elif (
                current_trend == Trend.BEARISH
                and sw.swing_type == SwingType.LOW
            ):
                if (
                    active_bos_swing is None
                    or sw.price < active_bos_swing.price
                ):
                    active_bos_swing = sw

        # ------------------------------------------------------------
        # 2. Evaluate this candle for a structural break.
        # ------------------------------------------------------------
        if current_trend == Trend.UNKNOWN:

            # First bullish structural break.
            if (
                last_high
                and get_swing_key(last_high)
                not in consumed_swings
                and close > last_high.price
            ):
                record_break(
                    BreakType.INITIAL_BREAK,
                    Trend.BULLISH,
                    last_high,
                    ts,
                    close,
                )

                current_trend = Trend.BULLISH
                active_bos_swing = last_high
                protected_swing = last_low

            # First bearish structural break.
            elif (
                last_low
                and get_swing_key(last_low)
                not in consumed_swings
                and close < last_low.price
            ):
                record_break(
                    BreakType.INITIAL_BREAK,
                    Trend.BEARISH,
                    last_low,
                    ts,
                    close,
                )

                current_trend = Trend.BEARISH
                active_bos_swing = last_low
                protected_swing = last_high

        elif current_trend == Trend.BULLISH:

            # CHOCH:
            # Close below the protected bullish swing.
            if (
                protected_swing
                and get_swing_key(protected_swing)
                not in consumed_swings
                and close < protected_swing.price
            ):
                record_break(
                    BreakType.CHOCH,
                    Trend.BEARISH,
                    protected_swing,
                    ts,
                    close,
                )

                current_trend = Trend.BEARISH
                active_bos_swing = last_low
                protected_swing = last_high

            # BOS:
            # Close above the active bullish structural high.
            elif (
                active_bos_swing
                and get_swing_key(active_bos_swing)
                not in consumed_swings
                and close > active_bos_swing.price
            ):
                record_break(
                    BreakType.BOS,
                    Trend.BULLISH,
                    active_bos_swing,
                    ts,
                    close,
                )

                # IMPORTANT:
                # Do not reassign active_bos_swing to last_high here.
                # The broken swing is consumed. The next confirmed
                # structural high must promote the BOS target naturally.

        elif current_trend == Trend.BEARISH:

            # CHOCH:
            # Close above the protected bearish swing.
            if (
                protected_swing
                and get_swing_key(protected_swing)
                not in consumed_swings
                and close > protected_swing.price
            ):
                record_break(
                    BreakType.CHOCH,
                    Trend.BULLISH,
                    protected_swing,
                    ts,
                    close,
                )

                current_trend = Trend.BULLISH
                active_bos_swing = last_high
                protected_swing = last_low

            # BOS:
            # Close below the active bearish structural low.
            elif (
                active_bos_swing
                and get_swing_key(active_bos_swing)
                not in consumed_swings
                and close < active_bos_swing.price
            ):
                record_break(
                    BreakType.BOS,
                    Trend.BEARISH,
                    active_bos_swing,
                    ts,
                    close,
                )

                # IMPORTANT:
                # Do not reassign active_bos_swing to last_low here.
                # The next confirmed structural low must promote the
                # target naturally.

    return breaks, classified_swings, current_trend


# ============================================================
# INCREMENTAL STRUCTURE STATE
# ============================================================
#
# StructureState: incremental, causal structure tracker.
#
# analyze_structure() above is UNCHANGED and remains the
# correctness reference. StructureState re-implements its exact
# loop body, but keeps state (current_trend, active_bos_swing,
# protected_swing, last_high, last_low, next_swing_idx,
# consumed_swings, classified_swings, breaks) persisted across
# step() calls instead of rebuilt from scratch on every call --
# same pattern as SwingState in strategy/swings.py.
#
# StructureState.step() is only proven correct when:
#   1. Rows are fed in strictly increasing chronological order.
#   2. Each row is fed exactly once.
#   3. New confirmed swings are fed via add_swings() in
#      confirmed_at order, at or before the row whose timestamp
#      reaches them -- matching how analyze_structure() unlocks
#      swings via its own next_swing_idx loop above.
#
# Equivalence to analyze_structure() is proven by
# phase_02_optimization/test_structure_state_equivalence.py.
# ============================================================


def _get_swing_key(sw: Swing) -> Tuple[SwingType, float, pd.Timestamp]:
    return (sw.swing_type, sw.price, sw.confirmed_at)


@dataclass
class StructureState:
    """
    Incremental equivalent of analyze_structure(). Feed rows one at a
    time via step(); feed newly confirmed swings via add_swings()
    before the step() call for the row that should unlock them.
    """

    current_trend: Trend = Trend.UNKNOWN

    active_bos_swing: Optional[Swing] = None
    protected_swing: Optional[Swing] = None

    last_high: Optional[Swing] = None
    last_low: Optional[Swing] = None

    # Growing, confirmed_at-ordered list of swings seen so far.
    # New swings are expected to be appended in non-decreasing
    # confirmed_at order (guaranteed by SwingState's causal feed),
    # so this never needs re-sorting.
    _swings_sorted: List[Swing] = field(default_factory=list)
    _next_swing_idx: int = 0

    _consumed_swings: Set[Tuple] = field(default_factory=set)

    # Accumulated outputs, growing across the state's lifetime --
    # same accumulate-and-return-the-whole-list pattern as
    # StrategyAdapter._confirmed_swings / self._zones.
    breaks: List[StructureBreak] = field(default_factory=list)
    classified_swings: List[ClassifiedSwing] = field(default_factory=list)

    def add_swings(self, new_swings: List[Swing]) -> None:
        """
        Append newly confirmed swings. Must be called with swings in
        confirmed_at order, and each swing added exactly once --
        same contract as SwingState's step() output being fed once.
        """
        self._swings_sorted.extend(new_swings)

    def _record_break(
        self,
        break_type: BreakType,
        direction: Trend,
        broken_swing: Swing,
        ts: pd.Timestamp,
        close: float,
    ) -> None:
        self.breaks.append(
            StructureBreak(
                break_type=break_type,
                direction=direction,
                broken_swing=broken_swing,
                broken_at=ts,
                break_price=close,
            )
        )
        self._consumed_swings.add(_get_swing_key(broken_swing))

    def step(self, ts: pd.Timestamp, close: float) -> None:
        """
        Process exactly one new candle. Mirrors the body of
        analyze_structure()'s per-row loop exactly, operating on
        persisted state instead of locals rebuilt every call.
        """

        # ------------------------------------------------------------
        # 1. Unlock and classify any swings confirmed at or before ts.
        # ------------------------------------------------------------
        while (
            self._next_swing_idx < len(self._swings_sorted)
            and self._swings_sorted[self._next_swing_idx].confirmed_at <= ts
        ):
            sw = self._swings_sorted[self._next_swing_idx]

            if sw.swing_type == SwingType.HIGH:
                if self.last_high is None:
                    lbl = SwingLabel.INITIAL_HIGH
                elif sw.price > self.last_high.price:
                    lbl = SwingLabel.HH
                else:
                    lbl = SwingLabel.LH

                self.last_high = sw

                if self.current_trend == Trend.BEARISH:
                    if (
                        self.protected_swing is None
                        or sw.price < self.protected_swing.price
                    ):
                        self.protected_swing = sw

            else:
                if self.last_low is None:
                    lbl = SwingLabel.INITIAL_LOW
                elif sw.price < self.last_low.price:
                    lbl = SwingLabel.LL
                else:
                    lbl = SwingLabel.HL

                self.last_low = sw

                if self.current_trend == Trend.BULLISH:
                    if (
                        self.protected_swing is None
                        or sw.price > self.protected_swing.price
                    ):
                        self.protected_swing = sw

            self.classified_swings.append(
                ClassifiedSwing(swing=sw, label=lbl)
            )

            self._next_swing_idx += 1

            if (
                self.current_trend == Trend.BULLISH
                and sw.swing_type == SwingType.HIGH
            ):
                if (
                    self.active_bos_swing is None
                    or sw.price > self.active_bos_swing.price
                ):
                    self.active_bos_swing = sw

            elif (
                self.current_trend == Trend.BEARISH
                and sw.swing_type == SwingType.LOW
            ):
                if (
                    self.active_bos_swing is None
                    or sw.price < self.active_bos_swing.price
                ):
                    self.active_bos_swing = sw

        # ------------------------------------------------------------
        # 2. Evaluate this candle for a structural break.
        # ------------------------------------------------------------
        if self.current_trend == Trend.UNKNOWN:

            if (
                self.last_high
                and _get_swing_key(self.last_high) not in self._consumed_swings
                and close > self.last_high.price
            ):
                self._record_break(
                    BreakType.INITIAL_BREAK, Trend.BULLISH, self.last_high, ts, close
                )
                self.current_trend = Trend.BULLISH
                self.active_bos_swing = self.last_high
                self.protected_swing = self.last_low

            elif (
                self.last_low
                and _get_swing_key(self.last_low) not in self._consumed_swings
                and close < self.last_low.price
            ):
                self._record_break(
                    BreakType.INITIAL_BREAK, Trend.BEARISH, self.last_low, ts, close
                )
                self.current_trend = Trend.BEARISH
                self.active_bos_swing = self.last_low
                self.protected_swing = self.last_high

        elif self.current_trend == Trend.BULLISH:

            if (
                self.protected_swing
                and _get_swing_key(self.protected_swing) not in self._consumed_swings
                and close < self.protected_swing.price
            ):
                self._record_break(
                    BreakType.CHOCH, Trend.BEARISH, self.protected_swing, ts, close
                )
                self.current_trend = Trend.BEARISH
                self.active_bos_swing = self.last_low
                self.protected_swing = self.last_high

            elif (
                self.active_bos_swing
                and _get_swing_key(self.active_bos_swing) not in self._consumed_swings
                and close > self.active_bos_swing.price
            ):
                self._record_break(
                    BreakType.BOS, Trend.BULLISH, self.active_bos_swing, ts, close
                )
                # Do not reassign active_bos_swing here -- matches
                # analyze_structure()'s explicit comment/behavior.

        elif self.current_trend == Trend.BEARISH:

            if (
                self.protected_swing
                and _get_swing_key(self.protected_swing) not in self._consumed_swings
                and close > self.protected_swing.price
            ):
                self._record_break(
                    BreakType.CHOCH, Trend.BULLISH, self.protected_swing, ts, close
                )
                self.current_trend = Trend.BULLISH
                self.active_bos_swing = self.last_high
                self.protected_swing = self.last_low

            elif (
                self.active_bos_swing
                and _get_swing_key(self.active_bos_swing) not in self._consumed_swings
                and close < self.active_bos_swing.price
            ):
                self._record_break(
                    BreakType.BOS, Trend.BEARISH, self.active_bos_swing, ts, close
                )
                # Do not reassign active_bos_swing here -- matches
                # analyze_structure()'s explicit comment/behavior.