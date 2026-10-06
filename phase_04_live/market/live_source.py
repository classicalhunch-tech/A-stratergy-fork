"""
phase_04_live/market/live_source.py

Live MT5 market data source, implementing phase_03_paper's
MarketDataSource contract so it can be handed directly to
MarketDataEngine (per phase_03_paper/market/engine.py: "Concrete
sources (mock, historical replay, live MT5) live in sibling modules
and all implement MarketDataSource below.").

Connection pattern (initialize/symbol_select/copy_rates_from_pos,
column renaming time->timestamp, tick_volume->volume) is copied
directly from pull_mt5_90000.py to stay consistent with how this
project already talks to MT5.

stream() polls MT5, quality-filters each new candle (drop-and-flag
policy), and yields ONLY accepted candles as raw dicts. Rejected
candles never reach MarketDataEngine -- they are only surfaced via
the on_quality_issue callback. This is the layer that prevents
MarketDataEngine.stream()'s hard-raise-on-bad-data behavior (correct
for historical replay) from crashing a live feed on a single bad tick.
"""

import time
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional

import MetaTrader5 as mt5

from phase_03_paper.market.engine import MarketDataSource
from phase_04_live.market.models import Candle, QualityIssue
from phase_04_live.market.normalizer import to_market_data_engine_dict
from phase_04_live.market.quality import QualityFilter


MT5_TERMINAL_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}


class LiveMarketSource(MarketDataSource):
    """
    MarketDataSource implementation backed by live MT5 polling.

    connect()/disconnect()/fetch_latest() are unchanged from the
    original implementation. stream() is the MarketDataSource
    contract method -- it is what MarketDataEngine actually calls.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        terminal_path: str = MT5_TERMINAL_PATH,
        poll_interval_seconds: float = 5.0,
        on_quality_issue: Optional[Callable[[QualityIssue], None]] = None,
    ):
        if timeframe not in TIMEFRAME_MAP:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {list(TIMEFRAME_MAP.keys())}"
            )

        self.symbol = symbol
        self.timeframe_str = timeframe
        self.timeframe = TIMEFRAME_MAP[timeframe]
        self.terminal_path = terminal_path
        self.poll_interval_seconds = poll_interval_seconds

        self._connected = False
        self._quality_filter = QualityFilter(on_issue=on_quality_issue, reject_gaps=False)
        self._last_seen_timestamp: Optional[datetime] = None

    def connect(self) -> None:
        if not mt5.initialize(path=self.terminal_path):
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

        if not mt5.symbol_select(self.symbol, True):
            mt5.shutdown()
            raise RuntimeError(
                f"symbol_select() failed for '{self.symbol}': {mt5.last_error()}"
            )

        self._connected = True

    def disconnect(self) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False

    def mark_history_loaded(self, last_candle: Candle) -> None:
        """
        Call after warming up the strategy with historical candles.

        Treats every candle up to and including last_candle as already
        processed, so stream() does not deliver them again, and seeds the
        quality filter so ordering and duplicate checks continue from there.
        """
        self._last_seen_timestamp = last_candle.timestamp
        self._quality_filter.seed(last_candle)

    def fetch_latest(self, count: int = 1) -> list[Candle]:
        """
        Returns the most recent `count` CLOSED candles (position 1
        onward -- position 0 is the currently-forming, unclosed
        candle and is deliberately excluded).
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 1, count)

        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos() failed: {mt5.last_error()}")

        candles = []
        for row in rates:
            candles.append(
                Candle(
                    timestamp=datetime.fromtimestamp(row["time"], tz=timezone.utc),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["tick_volume"]),
                    symbol=self.symbol,
                    timeframe=self.timeframe_str,
                )
            )

        return candles

    def poll_new_candles(self, last_seen_timestamp, lookback: int = 10) -> list[Candle]:
        """
        Fetches the last `lookback` closed candles and returns only
        those strictly newer than `last_seen_timestamp`, in
        chronological order.
        """
        recent = self.fetch_latest(lookback)

        if last_seen_timestamp is None:
            return recent

        return [c for c in recent if c.timestamp > last_seen_timestamp]

    def stream(self) -> Iterator[dict]:
        """
        MarketDataSource contract method.

        Polls MT5 forever, quality-filters each newly seen candle,
        and yields ONLY accepted candles, as raw dicts, in
        chronological order. Rejected candles update
        _last_seen_timestamp (so they are never re-considered) but
        are never yielded -- only surfaced via the QualityFilter's
        on_issue callback.

        This is an infinite generator by design -- MarketDataEngine
        drives it one candle at a time via next_candle(), the same
        pattern already used for historical/mock sources.
        """
        if not self._connected:
            self.connect()

        while True:
            new_candles = self.poll_new_candles(self._last_seen_timestamp)

            for candle in new_candles:
                accepted, _issue = self._quality_filter.check(candle)
                self._last_seen_timestamp = candle.timestamp

                if accepted:
                    yield to_market_data_engine_dict(candle)

            time.sleep(self.poll_interval_seconds)
