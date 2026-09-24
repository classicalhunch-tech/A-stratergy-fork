"""
phase_03_paper/market/engine.py

Phase 3.3 — Market Data

Provides the runtime with a controlled, chronological stream of
market data.

Responsibilities:
    - receive raw market data
    - normalize timestamps
    - normalize numeric values
    - validate OHLC candles
    - validate volume
    - enforce chronological ordering
    - expose latest market state
    - expose market-data health
    - provide incremental one-candle-at-a-time access

Explicitly does NOT:
    - make trading decisions
    - enforce session rules
    - execute trades
    - manage positions
    - render dashboard/UI
    - construct notifications

Concrete sources (mock, historical replay, live MT5) live in
sibling modules and all implement MarketDataSource below.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candle:
    """A single normalized, validated OHLC candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self):
        if self.timestamp.tzinfo is None:
            raise ValueError("Candle.timestamp must be timezone-aware")

        values = {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"Candle.{name} must be finite, got {value}"
                )

        if self.volume < 0:
            raise ValueError(
                f"Candle.volume cannot be negative, got {self.volume}"
            )

        if not (
            self.low <= self.open <= self.high
            and self.low <= self.close <= self.high
        ):
            raise ValueError(
                f"Invalid OHLC candle at {self.timestamp}: "
                f"O={self.open} H={self.high} "
                f"L={self.low} C={self.close}"
            )


# ---------------------------------------------------------------------------
# Market data source abstraction
# ---------------------------------------------------------------------------

class MarketDataSource(ABC):
    """
    Minimal contract any market data source must satisfy.

    Sources produce RAW records only.

    They must NOT:
        - validate candles
        - normalize timestamps
        - make trading decisions
        - apply strategy rules

    Those responsibilities belong to MarketDataEngine.
    """

    @abstractmethod
    def stream(self) -> Iterator[dict]:
        """Yield raw candle records in source order."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Market data status
# ---------------------------------------------------------------------------

class MarketDataStatus(Enum):
    HEALTHY = "HEALTHY"
    STALE = "STALE"
    ERROR = "ERROR"
    NOT_STARTED = "NOT_STARTED"


@dataclass(frozen=True)
class MarketDataStatusReport:
    """Health/status snapshot for monitoring and dashboard consumers."""

    status: MarketDataStatus
    candles_received: int
    latest_timestamp: Optional[datetime]
    last_error: Optional[str]


# ---------------------------------------------------------------------------
# Market Data Engine
# ---------------------------------------------------------------------------

class MarketDataEngine:
    """
    Single authoritative source of the runtime's current market state.

    Responsibilities:
        - receive raw candles
        - normalize timestamps
        - validate candles
        - enforce chronological ordering
        - expose latest candle
        - expose candle count
        - expose health/status
        - provide incremental candle access

    Explicitly NOT responsible for:
        - trading decisions
        - session rules
        - trade execution
        - position management
        - dashboard rendering
        - notification wording
    """

    def __init__(self, source: MarketDataSource):
        self._source = source

        self._latest: Optional[Candle] = None
        self._last_timestamp: Optional[datetime] = None
        self._candle_count: int = 0

        self._status: MarketDataStatus = MarketDataStatus.NOT_STARTED
        self._last_error: Optional[str] = None

        # Persistent iterator used by next_candle().
        self._stream_iter: Optional[Iterator[Candle]] = None

    # -----------------------------------------------------------------------
    # Normalization
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize(raw: dict) -> Candle:
        """
        Convert a raw market-data record into a validated Candle.

        Expected raw keys:
            timestamp
            open
            high
            low
            close
            volume (optional)

        Timestamp may be:
            - datetime
            - ISO-8601 string

        Naive timestamps are assumed to be UTC.

        Timezone-aware timestamps are converted explicitly to UTC.
        """

        ts = raw["timestamp"]

        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        if not isinstance(ts, datetime):
            raise TypeError(
                "timestamp must be a datetime or ISO-8601 string"
            )

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        return Candle(
            timestamp=ts,
            open=float(raw["open"]),
            high=float(raw["high"]),
            low=float(raw["low"]),
            close=float(raw["close"]),
            volume=float(raw.get("volume", 0.0)),
        )

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def stream(self) -> Iterator[Candle]:
        """
        Pull from the source, normalize records, validate candles,
        enforce chronological ordering, and yield Candle objects.

        This is the authoritative ingestion path for market data.
        """

        for raw in self._source.stream():

            # ---------------------------------------------------------------
            # Normalize and validate
            # ---------------------------------------------------------------

            try:
                candle = self._normalize(raw)

            except (KeyError, ValueError, TypeError) as exc:
                self._status = MarketDataStatus.ERROR
                self._last_error = str(exc)

                logger.error(
                    "Market data normalization failed: %s",
                    exc,
                )

                raise

            # ---------------------------------------------------------------
            # Chronological ordering
            # ---------------------------------------------------------------

            if (
                self._last_timestamp is not None
                and candle.timestamp <= self._last_timestamp
            ):
                self._status = MarketDataStatus.ERROR

                self._last_error = (
                    f"Out-of-order candle: {candle.timestamp} "
                    f"<= previous {self._last_timestamp}"
                )

                logger.error(self._last_error)

                raise ValueError(self._last_error)

            # ---------------------------------------------------------------
            # Update authoritative state
            # ---------------------------------------------------------------

            self._latest = candle
            self._last_timestamp = candle.timestamp
            self._candle_count += 1
            self._status = MarketDataStatus.HEALTHY
            self._last_error = None

            # ---------------------------------------------------------------
            # Release candle to downstream runtime
            # ---------------------------------------------------------------

            yield candle

    # -----------------------------------------------------------------------
    # Incremental runtime access
    # -----------------------------------------------------------------------

    def next_candle(self) -> Optional[Candle]:
        """
        Advance the underlying market-data stream by exactly one candle.

        Returns:
            Candle: the next successfully ingested candle.
            None: when the underlying source is exhausted.

        Runtime components such as RuntimeCoordinator should use this
        method for one-tick-at-a-time processing instead of consuming
        the entire stream directly.
        """

        if self._stream_iter is None:
            self._stream_iter = self.stream()

        try:
            return next(self._stream_iter)

        except StopIteration:
            return None

    # -----------------------------------------------------------------------
    # State accessors
    # -----------------------------------------------------------------------

    def latest(self) -> Optional[Candle]:
        """
        Return the most recently ingested candle.

        Returns None if no candle has arrived yet.
        """

        return self._latest

    def candle_count(self) -> int:
        """Return the total number of successfully ingested candles."""

        return self._candle_count

    def status(self) -> MarketDataStatusReport:
        """
        Return a health/status snapshot for monitoring
        and dashboard consumers.
        """

        return MarketDataStatusReport(
            status=self._status,
            candles_received=self._candle_count,
            latest_timestamp=self._last_timestamp,
            last_error=self._last_error,
        )