"""
phase_04_live/market/models.py

Data model for live market candles.

Matches the existing project convention (see pull_mt5_90000.py):
    timestamp, open, high, low, close, volume

Extended here with symbol/timeframe, since Phase 3's CSVs were
single-symbol/single-timeframe and didn't need to carry that context
per row -- a live feed does, so callers know what each candle belongs to.

Note: phase_03_paper/models.py has no candle/OHLCV model (it only
defines session/notification/audit models), so this is not a
duplicate of anything -- it's the first candle model in the project.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str

    def as_dict(self) -> dict:
        """Matches the project's existing CSV column convention."""
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class QualityIssueType(str, Enum):
    ZERO_OR_NEGATIVE_PRICE = "ZERO_OR_NEGATIVE_PRICE"
    INVALID_OHLC = "INVALID_OHLC"
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    TIMEFRAME_GAP = "TIMEFRAME_GAP"


@dataclass(frozen=True)
class QualityIssue:
    """
    Raised whenever quality.py drops a candle. Carries enough context
    for monitoring/alerts.py to raise a real alert later -- this class
    only describes the problem, it does not decide what to do about it.
    """
    issue_type: QualityIssueType
    symbol: str
    timeframe: str
    detail: str
    candle: Candle | None
    detected_at: datetime
