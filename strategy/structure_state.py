"""
StructureState: incremental, causal structure tracker.

Mirrors the SwingState pattern already proven in strategy/swings.py
and used by phase_03_paper/signals/adapter.py's _advance_swing_state
and _advance_zone_state.

analyze_structure(df, swings) in this same file is UNCHANGED and
remains the correctness reference. StructureState re-implements its
exact loop body, but keeps state (current_trend, active_bos_swing,
protected_swing, last_high, last_low, next_swing_idx, consumed_swings,
classified_swings, breaks, rows processed) persisted across step()
calls instead of rebuilt from scratch on every call.

Like SwingState.step(), StructureState.step() is only proven correct
when:
  1. Rows are fed in strictly increasing chronological order.
  2. Each row is fed exactly once.
  3. New confirmed swings are fed via add_swings() in confirmed_at
     order, at or before the row whose timestamp reaches them --
     matching how the original analyze_structure() unlocks swings
     via its "while next_swing_idx < len(swings_sorted) and
     swings_sorted[next_swing_idx].confirmed_at <= ts" loop.

Equivalence to analyze_structure() must be proven the same way
SwingState was: a dedicated test feeding the same swings/candles
through both analyze_structure(df, swings) and StructureState
row-by-row, on staged real/synthetic datasets, asserting identical
breaks, classified_swings, and final trend.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import pandas as pd

from strategy.swings import Swing, SwingType
from strategy.structure import (
    BreakType,
    ClassifiedSwing,
    StructureBreak,
    SwingLabel,
    Trend,
)


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