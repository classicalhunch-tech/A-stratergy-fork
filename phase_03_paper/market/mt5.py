"""
phase_03_paper/market/mt5.py

Phase 4 — Future MetaTrader 5 Market Data Source

Defines the interface placeholder for a future live MetaTrader 5
market-data source.

IMPORTANT
---------
This module is intentionally NOT implemented during Phase 3.

Phase 3 uses historical CSV replay for paper-trading validation.

The future MT5 source will implement the same MarketDataSource
interface used by CsvReplayMarketDataSource. This allows the
MarketDataEngine and downstream strategy components to remain
unchanged when live market data is introduced.

RESPONSIBILITIES
----------------
This class will eventually:

    1. Connect to MetaTrader 5.
    2. Subscribe/poll for market data.
    3. Convert MT5 candle data into raw candle records.
    4. Yield those records through stream().

It will NOT:

    - validate OHLC data
    - normalize timestamps
    - enforce chronological ordering
    - generate trading signals
    - execute orders
    - contain strategy logic

Those responsibilities belong to other Phase 3/4 components.
"""

from __future__ import annotations

from typing import Iterator

from phase_03_paper.market.engine import MarketDataSource


class MT5MarketDataSource(MarketDataSource):
    """
    Future live MetaTrader 5 market-data source.

    STATUS:
        Placeholder only.

    The implementation will be added during the future live-data /
    execution integration phase.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
    ) -> None:
        self._symbol: str = symbol
        self._timeframe: str = timeframe

    def stream(self) -> Iterator[dict]:
        """
        Yield raw market-data records from MetaTrader 5.

        Not implemented during Phase 3.
        """
        raise NotImplementedError(
            "MT5MarketDataSource is reserved for future live-data integration"
        )


__all__ = [
    "MT5MarketDataSource",
]