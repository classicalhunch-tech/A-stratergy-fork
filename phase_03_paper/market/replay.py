"""
phase_03_paper/market/replay.py

Phase 3.3 — Historical CSV Replay Market Data Source

Reads historical OHLC candle data from a CSV file and exposes it
through the Phase 3 MarketDataSource interface.

IMPORTANT:
    This source produces RAW records only.

    Validation, timestamp normalization, OHLC validation, volume
    validation, and chronological enforcement remain the responsibility
    of MarketDataEngine.

This source therefore remains interchangeable with future live sources
such as MT5 without changing MarketDataEngine or downstream components.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from phase_03_paper.market.engine import MarketDataSource


class CsvReplayMarketDataSource(MarketDataSource):
    """
    Historical CSV replay source.

    The CSV must contain these columns:

        timestamp
        open
        high
        low
        close

    Optional:

        volume

    The source does not validate candle values. That is deliberately
    handled by MarketDataEngine.
    """

    REQUIRED_COLUMNS = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }

    OPTIONAL_COLUMNS = {
        "volume",
    }

    def __init__(self, csv_path: str | Path):
        self._csv_path = Path(csv_path)

    def stream(self) -> Iterator[dict]:
        """
        Read the CSV and yield raw candle dictionaries in file order.

        MarketDataEngine is responsible for:
            - timestamp parsing
            - UTC normalization
            - OHLC validation
            - volume validation
            - chronological ordering
        """

        if not self._csv_path.exists():
            raise FileNotFoundError(
                f"Historical market data file not found: {self._csv_path}"
            )

        if not self._csv_path.is_file():
            raise ValueError(
                f"Historical market data path is not a file: "
                f"{self._csv_path}"
            )

        with self._csv_path.open(
            mode="r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError(
                    f"CSV file has no header row: {self._csv_path}"
                )

            columns = {
                column.strip().lower()
                for column in reader.fieldnames
                if column is not None
            }

            missing = self.REQUIRED_COLUMNS - columns

            if missing:
                raise ValueError(
                    "CSV is missing required columns: "
                    + ", ".join(sorted(missing))
                )

            for row_number, row in enumerate(reader, start=2):
                yield self._normalize_row(row, row_number)

    @staticmethod
    def _normalize_row(
        row: dict[str, str | None],
        row_number: int,
    ) -> dict:
        """
        Convert CSV column names into the raw schema expected by
        MarketDataEngine.

        Values remain strings intentionally. MarketDataEngine performs
        the final numeric conversion and validation.
        """

        normalized = {
            str(key).strip().lower(): value
            for key, value in row.items()
            if key is not None
        }

        required = (
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        )

        for column in required:
            value = normalized.get(column)

            if value is None or not str(value).strip():
                raise ValueError(
                    f"Missing value for '{column}' "
                    f"at CSV row {row_number}"
                )

        record = {
            "timestamp": str(normalized["timestamp"]).strip(),
            "open": str(normalized["open"]).strip(),
            "high": str(normalized["high"]).strip(),
            "low": str(normalized["low"]).strip(),
            "close": str(normalized["close"]).strip(),
        }

        volume = normalized.get("volume")

        if volume is not None and str(volume).strip():
            record["volume"] = str(volume).strip()

        return record


__all__ = [
    "CsvReplayMarketDataSource",
]