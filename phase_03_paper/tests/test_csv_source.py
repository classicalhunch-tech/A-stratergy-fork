"""
phase_03_paper/tests/test_csv_source.py

CsvMarketDataSource — historical CSV source tests.

PURPOSE
-------
Verify CsvMarketDataSource reads a CSV into raw records suitable for
MarketDataEngine._normalize(), handles case-insensitive headers,
optional volume, missing required columns, and preserves file order.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phase_03_paper.replay.csv_source import CsvMarketDataSource


def _write_csv(tmpdir: str, filename: str, content: str) -> str:
    path = Path(tmpdir) / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestCsvMarketDataSource(unittest.TestCase):
    """Test suite for CsvMarketDataSource."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

    def test_reads_standard_columns_in_order(self) -> None:
        """Rows are yielded as raw dicts, in file order."""

        content = (
            "timestamp,open,high,low,close,volume\n"
            "2026-01-05T09:00:00+00:00,1.1000,1.1050,1.0950,1.1010,100\n"
            "2026-01-05T09:05:00+00:00,1.1010,1.1060,1.0990,1.1020,120\n"
        )
        path = _write_csv(self._tmpdir.name, "data.csv", content)

        source = CsvMarketDataSource(path)
        rows = list(source.stream())

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["open"], "1.1000")
        self.assertEqual(rows[1]["close"], "1.1020")
        self.assertEqual(len(source), 2)

    def test_case_insensitive_headers(self) -> None:
        """Headers like 'Timestamp'/'Open' are matched case-insensitively."""

        content = (
            "Timestamp,Open,High,Low,Close,Volume\n"
            "2026-01-05T09:00:00+00:00,1.1,1.2,1.0,1.15,50\n"
        )
        path = _write_csv(self._tmpdir.name, "data.csv", content)

        source = CsvMarketDataSource(path)
        rows = list(source.stream())

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["open"], "1.1")

    def test_volume_optional_when_column_absent(self) -> None:
        """A CSV without a volume column still yields valid rows, without
        a 'volume' key (MarketDataEngine defaults it downstream)."""

        content = (
            "timestamp,open,high,low,close\n"
            "2026-01-05T09:00:00+00:00,1.1,1.2,1.0,1.15\n"
        )
        path = _write_csv(self._tmpdir.name, "data.csv", content)

        source = CsvMarketDataSource(path)
        rows = list(source.stream())

        self.assertEqual(len(rows), 1)
        self.assertNotIn("volume", rows[0])

    def test_missing_required_column_raises(self) -> None:
        """A CSV missing a required column (e.g. 'close') raises clearly."""

        content = "timestamp,open,high,low\n2026-01-05T09:00:00+00:00,1.1,1.2,1.0\n"
        path = _write_csv(self._tmpdir.name, "data.csv", content)

        with self.assertRaises(ValueError):
            CsvMarketDataSource(path)

    def test_missing_file_raises_file_not_found(self) -> None:
        """A nonexistent path raises FileNotFoundError, not a generic error."""

        missing_path = str(Path(self._tmpdir.name) / "does_not_exist.csv")

        with self.assertRaises(FileNotFoundError):
            CsvMarketDataSource(missing_path)

    def test_custom_column_names(self) -> None:
        """Custom column-name mappings are honored (e.g. MT5-style export)."""

        content = (
            "Time,O,H,L,C,V\n"
            "2026-01-05T09:00:00+00:00,1.1,1.2,1.0,1.15,10\n"
        )
        path = _write_csv(self._tmpdir.name, "data.csv", content)

        source = CsvMarketDataSource(
            path,
            timestamp_col="Time",
            open_col="O",
            high_col="H",
            low_col="L",
            close_col="C",
            volume_col="V",
        )
        rows = list(source.stream())

        self.assertEqual(rows[0]["open"], "1.1")
        self.assertEqual(rows[0]["volume"], "10")

    def test_stream_can_be_called_multiple_times(self) -> None:
        """stream() yields a fresh iterator each call (rows read once at init)."""

        content = (
            "timestamp,open,high,low,close\n"
            "2026-01-05T09:00:00+00:00,1.1,1.2,1.0,1.15\n"
        )
        path = _write_csv(self._tmpdir.name, "data.csv", content)

        source = CsvMarketDataSource(path)

        first_pass = list(source.stream())
        second_pass = list(source.stream())

        self.assertEqual(len(first_pass), 1)
        self.assertEqual(len(second_pass), 1)


if __name__ == "__main__":
    unittest.main()