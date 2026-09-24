"""
phase_03_paper/replay/csv_source.py

CsvMarketDataSource — historical CSV-backed MarketDataSource.

PURPOSE
-------
Implements the MarketDataSource contract defined in
phase_03_paper/market/engine.py, which explicitly anticipates this:

    "Concrete sources (mock, historical replay, live MT5) live in
    sibling modules and all implement MarketDataSource below."

This module does NOT duplicate MarketDataEngine's responsibilities.
Per that contract, a MarketDataSource must yield RAW records only —
it must NOT validate candles, normalize timestamps, or make any
trading decisions. All of that remains MarketDataEngine's job.

CsvMarketDataSource's only job is: read a CSV file, and yield each
row as a raw dict with the keys MarketDataEngine._normalize() expects
(timestamp, open, high, low, close, volume).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List

from phase_03_paper.market.engine import MarketDataSource


class CsvMarketDataSource(MarketDataSource):
    """
    Read historical OHLC(V) candles from a CSV file and yield them as
    raw records, in file order.

    The CSV is read once into memory at construction time. Historical
    replay datasets are expected to be finite and bounded, and reading
    fully up front lets ReplayRunner know the total candle count and
    each candle's timestamp ahead of driving the runtime — see
    replay_runner.py for why that matters.

    Column names are matched case-insensitively. Defaults assume
    columns named timestamp/open/high/low/close/volume; pass the
    *_col parameters to map differently-named columns (e.g. a MetaTrader
    export using "Time"/"Open"/"High"/"Low"/"Close"/"Volume").

    The volume column is optional — if absent, the "volume" key is
    simply omitted from each record, and MarketDataEngine._normalize()
    already defaults missing volume to 0.0.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        timestamp_col: str = "timestamp",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
    ) -> None:
        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self._timestamp_col = timestamp_col
        self._open_col = open_col
        self._high_col = high_col
        self._low_col = low_col
        self._close_col = close_col
        self._volume_col = volume_col

        self._rows: List[Dict[str, str]] = self._read_rows(path)

    def _read_rows(self, path: Path) -> List[Dict[str, str]]:
        """Read the CSV once, matching required columns case-insensitively."""

        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)

            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header row: {path}")

            # Case-insensitive header lookup.
            header_map = {
                name.strip().lower(): name for name in reader.fieldnames
            }

            required = {
                self._timestamp_col: True,
                self._open_col: True,
                self._high_col: True,
                self._low_col: True,
                self._close_col: True,
            }

            for expected_col in required:
                if expected_col.lower() not in header_map:
                    raise ValueError(
                        f"CSV file {path} is missing required column "
                        f"'{expected_col}' (available columns: "
                        f"{reader.fieldnames})"
                    )

            has_volume = self._volume_col.lower() in header_map

            rows: List[Dict[str, str]] = []

            for csv_row in reader:
                row: Dict[str, str] = {
                    "timestamp": csv_row[header_map[self._timestamp_col.lower()]],
                    "open": csv_row[header_map[self._open_col.lower()]],
                    "high": csv_row[header_map[self._high_col.lower()]],
                    "low": csv_row[header_map[self._low_col.lower()]],
                    "close": csv_row[header_map[self._close_col.lower()]],
                }

                if has_volume:
                    row["volume"] = csv_row[header_map[self._volume_col.lower()]]

                rows.append(row)

            return rows

    def stream(self) -> Iterator[dict]:
        """Yield each CSV row as a raw record, in file order."""

        for row in self._rows:
            yield dict(row)

    def __len__(self) -> int:
        """Total number of candle rows available in this source."""

        return len(self._rows)


__all__ = ["CsvMarketDataSource"]