"""
strategy/structure_targets.py

Structural take-profit: the nearest CONFIRMED 15M swing high above the fill
(swing low below, for shorts), taken from the same find_swings() the rest of
the system uses.

Look-ahead safety: a swing is only usable if its confirmed_at is at or before
the fill time (the open of the fill bar). 15M bar labels are close times, so
a swing confirmed at label C was fully known once the clock reached C.

Rules:
    - nearest swing beyond the fill price, within max_age
    - reward to that swing must be >= min_rr x real risk, otherwise None
      (the backtest skips the trade)
    - advance_to_min_rr=True is an A/B variant: instead of skipping, use the
      nearest swing that does give >= min_rr
"""

from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import pandas as pd

from dashboard.mtf_context import resample_ohlc
from strategy.signals import SignalType
from strategy.swings import Swing, SwingType, find_swings

EPSILON = 1e-8


@dataclass(frozen=True)
class StructureTargetConfig:
    min_rr: float = 1.5
    max_age: pd.Timedelta = pd.Timedelta(days=3)
    advance_to_min_rr: bool = False

    def __post_init__(self) -> None:
        if self.min_rr <= 0:
            raise ValueError("min_rr must be > 0")
        if self.max_age <= pd.Timedelta(0):
            raise ValueError("max_age must be > 0")


class StructureTargets:
    def __init__(
        self,
        swings: Sequence[Swing],
        config: Optional[StructureTargetConfig] = None,
    ) -> None:
        self.config = config or StructureTargetConfig()

        ordered = sorted(swings, key=lambda s: s.confirmed_at)

        self._highs = [s for s in ordered if s.swing_type == SwingType.HIGH]
        self._lows = [s for s in ordered if s.swing_type == SwingType.LOW]
        self._high_conf = [s.confirmed_at for s in self._highs]
        self._low_conf = [s.confirmed_at for s in self._lows]

    def target(
        self,
        direction: SignalType,
        fill_price: float,
        stop_loss: float,
        fill_time: pd.Timestamp,
    ) -> Optional[float]:

        risk = abs(fill_price - stop_loss)

        if risk <= EPSILON:
            return None

        fill_time = pd.Timestamp(fill_time)

        if direction == SignalType.LONG:
            if fill_price <= stop_loss + EPSILON:
                return None
            pool, conf = self._highs, self._high_conf
        elif direction == SignalType.SHORT:
            if fill_price >= stop_loss - EPSILON:
                return None
            pool, conf = self._lows, self._low_conf
        else:
            return None

        visible = bisect_right(conf, fill_time)
        oldest = fill_time - self.config.max_age

        prices = []

        for j in range(visible - 1, -1, -1):
            if conf[j] < oldest:
                break

            price = float(pool[j].price)

            if direction == SignalType.LONG and price > fill_price + EPSILON:
                prices.append(price)
            elif direction == SignalType.SHORT and price < fill_price - EPSILON:
                prices.append(price)

        if not prices:
            return None

        # nearest first: ascending for longs, descending for shorts
        prices.sort(reverse=(direction == SignalType.SHORT))

        for price in prices:
            reward_to_risk = abs(price - fill_price) / risk

            if reward_to_risk >= self.config.min_rr - EPSILON:
                return price

            if not self.config.advance_to_min_rr:
                return None

        return None


def build_structure_target_fn(
    df_5m: pd.DataFrame,
    tf: str = "15min",
    origin: str = "start_day",
    config: Optional[StructureTargetConfig] = None,
) -> Callable[[SignalType, float, float, pd.Timestamp], Optional[float]]:
    """Resample to tf, run find_swings, return target(direction, fill, stop, time)."""

    htf = resample_ohlc(df_5m, tf, origin=origin)
    swings = find_swings(htf)
    return StructureTargets(swings, config).target
