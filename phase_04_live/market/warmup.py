"""
phase_04_live/market/warmup.py

Startup warm-up for the live strategy adapter.

StrategyAdapter starts empty. Its multi-timeframe filter needs history
before it approves anything, and the 1h structure needs more than the
minimum to be trustworthy. warm_up_adapter() feeds recent historical
candles into the adapter, in order, before the live loop starts.

SAFETY: any signal that triggers while replaying history is DISCARDED
and only counted. Those triggers are in the past, so they must never
become orders. Setups still pending when the warm-up ends stay pending,
exactly as they would if the bot had been running continuously.

This module has no MetaTrader5 dependency, so it can be tested anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from phase_03_paper.market.engine import Candle as PaperCandle


# About one trading week of 5-minute candles. Costs roughly half a
# minute at startup.
DEFAULT_WARMUP_CANDLES = 2000


@dataclass(frozen=True)
class WarmupResult:
    candles_fed: int
    signals_discarded: int
    first_timestamp: Optional[datetime]
    last_timestamp: Optional[datetime]
    adapter_errors: int


def to_paper_candle(candle: Any) -> PaperCandle:
    """Convert a live candle (or any candle-like object) to a Phase 3 Candle."""

    timestamp = candle.timestamp

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)

    return PaperCandle(
        timestamp=timestamp,
        open=float(candle.open),
        high=float(candle.high),
        low=float(candle.low),
        close=float(candle.close),
        volume=float(getattr(candle, "volume", 0.0) or 0.0),
    )


def warm_up_adapter(adapter: Any, candles: Sequence[Any]) -> WarmupResult:
    """
    Feed historical candles (oldest first) into the adapter.

    All candles are validated and converted BEFORE the first one is fed,
    so a bad candle can never leave the adapter half-loaded.

    Raises ValueError when a candle is invalid or the candles are not in
    strictly increasing time order.
    """

    if not candles:
        return WarmupResult(0, 0, None, None, 0)

    converted = []
    previous: Optional[datetime] = None

    for index, candle in enumerate(candles):
        try:
            paper = to_paper_candle(candle)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                f"Warm-up candle {index} is invalid: {exc}"
            ) from exc

        if previous is not None and paper.timestamp <= previous:
            raise ValueError(
                "Warm-up candles must be in strictly increasing time "
                f"order: candle {index} at {paper.timestamp} is not "
                f"after {previous}."
            )

        converted.append(paper)
        previous = paper.timestamp

    errors_before = len(adapter.errors)
    discarded = 0

    for paper in converted:
        events = adapter.on_candle(paper)
        discarded += len(events)

    return WarmupResult(
        candles_fed=len(converted),
        signals_discarded=discarded,
        first_timestamp=converted[0].timestamp,
        last_timestamp=converted[-1].timestamp,
        adapter_errors=len(adapter.errors) - errors_before,
    )


__all__ = [
    "DEFAULT_WARMUP_CANDLES",
    "WarmupResult",
    "to_paper_candle",
    "warm_up_adapter",
]
