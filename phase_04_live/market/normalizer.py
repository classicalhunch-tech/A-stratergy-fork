
"""
phase_04_live/market/normalizer.py

Converts an accepted phase_04_live Candle into the raw dict shape
phase_03_paper.market.engine.MarketDataEngine expects from any
MarketDataSource.

MarketDataSource.stream() must yield raw dicts, not Candle objects --
normalization and validation happen exclusively inside
MarketDataEngine. This module only produces the dict shape
MarketDataEngine._normalize() expects: timestamp, open, high, low,
close, volume.

symbol/timeframe are dropped here since Phase 3 Candle has no fields
for them (Phase 3 was built single-symbol/single-timeframe). If
Phase 4 ever needs multi-symbol routing, handle it via which
MarketDataEngine instance a symbol is routed to -- do not add
symbol/timeframe fields to phase_03_paper.market.engine.Candle to
work around this.
"""

from phase_04_live.market.models import Candle


def to_market_data_engine_dict(candle: Candle) -> dict:
    """
    Convert a phase_04_live Candle (already quality-filtered) into
    the raw dict shape MarketDataEngine._normalize() expects.
    """
    return {
        "timestamp": candle.timestamp,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
    }
