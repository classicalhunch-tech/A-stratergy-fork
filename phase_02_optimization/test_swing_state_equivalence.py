"""
phase_02_optimization/test_swing_state_equivalence.py

Equivalence test: find_swings() (reference) vs SwingState.step()
(incremental candidate).

This file imports the real implementations from strategy/swings.py.
It does NOT redefine Swing, SwingType, find_swings, or SwingState --
duplicating them here would mean this test could pass even if the
real strategy/swings.py were broken, which defeats its purpose.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from strategy.swings import Swing, SwingState, find_swings


def _generate_synthetic_ohlc(n: int, seed: int = 42) -> pd.DataFrame:
    """
    Build a reproducible, causally-ordered synthetic OHLC DataFrame
    with a DatetimeIndex -- deliberately volatile, so it actually
    exercises seeking_high/seeking_low transitions rather than
    drifting flat.
    """

    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        "2020-01-01", periods=n, freq="5min", tz="UTC"
    )

    steps = rng.normal(loc=0.0, scale=1.0, size=n)
    closes = 100.0 + np.cumsum(steps)

    intrabar_range = rng.uniform(0.1, 1.0, size=n)
    opens = closes - steps
    highs = np.maximum(opens, closes) + intrabar_range
    lows = np.minimum(opens, closes) - intrabar_range

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        },
        index=timestamps,
    )


def _run_incremental(df: pd.DataFrame) -> List[Swing]:
    """
    Drive one SwingState through the DataFrame one row at a time,
    after applying the SAME normalization find_swings() applies
    internally (lower-case columns, sort_index), so both paths
    process candles in an identical order.
    """

    df_copy = df.copy()
    df_copy.columns = [col.lower() for col in df_copy.columns]
    df_copy = df_copy.sort_index()

    state = SwingState()
    swings: List[Swing] = []

    for ts, row in df_copy.iterrows():
        confirmed = state.step(
            ts=ts,
            high=row["high"],
            low=row["low"],
            close=row["close"],
        )
        if confirmed is not None:
            swings.append(confirmed)

    return swings


class SwingStateEquivalenceTestCase(unittest.TestCase):
    """Shared assertion logic used by every sized/staged test below."""

    def assert_swings_equivalent(
        self,
        reference: List[Swing],
        candidate: List[Swing],
        context: str,
    ) -> None:
        self.assertEqual(
            len(reference),
            len(candidate),
            f"[{context}] Swing count differs: "
            f"find_swings()={len(reference)} vs SwingState={len(candidate)}",
        )

        for i, (ref, cand) in enumerate(zip(reference, candidate)):
            self.assertEqual(
                ref.swing_type,
                cand.swing_type,
                f"[{context}] Swing #{i}: swing_type differs",
            )
            self.assertEqual(
                ref.price,
                cand.price,
                f"[{context}] Swing #{i}: price differs "
                f"({ref.price} vs {cand.price})",
            )
            self.assertEqual(
                ref.formed_at,
                cand.formed_at,
                f"[{context}] Swing #{i}: formed_at differs "
                f"({ref.formed_at} vs {cand.formed_at})",
            )
            self.assertEqual(
                ref.confirmed_at,
                cand.confirmed_at,
                f"[{context}] Swing #{i}: confirmed_at differs "
                f"({ref.confirmed_at} vs {cand.confirmed_at})",
            )

    def run_equivalence(self, n: int, seed: int = 42) -> None:
        df = _generate_synthetic_ohlc(n, seed=seed)

        reference = find_swings(df)
        candidate = _run_incremental(df)

        self.assert_swings_equivalent(
            reference, candidate, context=f"{n} synthetic candles"
        )

        print(
            f"[OK] {n} candles: find_swings()={len(reference)} swings, "
            f"SwingState()={len(candidate)} swings -- identical"
        )


class TestSwingStateEquivalenceStaged(SwingStateEquivalenceTestCase):
    """
    Staged synthetic equivalence tests, smallest to largest, per the
    required testing order: 500 -> 1,000 -> 2,000 -> 5,000.
    """

    def test_equivalence_500_candles(self):
        self.run_equivalence(500)

    def test_equivalence_1000_candles(self):
        self.run_equivalence(1000)

    def test_equivalence_2000_candles(self):
        self.run_equivalence(2000)

    def test_equivalence_5000_candles(self):
        self.run_equivalence(5000)

    def test_equivalence_holds_across_multiple_seeds(self):
        """
        A single seed proving equivalence once is weaker evidence
        than several independent random sequences agreeing.
        """
        for seed in (1, 7, 99, 12345):
            with self.subTest(seed=seed):
                self.run_equivalence(500, seed=seed)


class TestSwingStateEquivalenceRealDataset(SwingStateEquivalenceTestCase):
    """
    Bonus confirmation against the real historical dataset, if
    present. Skips gracefully rather than failing the suite when the
    file isn't available.
    """

    DATA_FILE = "your_data_file.csv"

    def test_equivalence_on_real_dataset(self):
        path = Path(self.DATA_FILE)
        if not path.exists():
            self.skipTest(f"{self.DATA_FILE!r} not found; skipping real-data check.")

        raw = pd.read_csv(path)
        raw.columns = [str(c).strip().lower() for c in raw.columns]

        if "timestamp" not in raw.columns:
            self.skipTest(
                f"{self.DATA_FILE!r} has no 'timestamp' column; "
                "skipping real-data check."
            )

        raw["timestamp"] = pd.to_datetime(
            raw["timestamp"], errors="coerce", utc=True
        )
        raw = raw.dropna(subset=["timestamp"])
        df = (
            raw.set_index("timestamp")
            .sort_index()[["open", "high", "low", "close"]]
            .dropna()
        )

        reference = find_swings(df)
        candidate = _run_incremental(df)

        self.assert_swings_equivalent(
            reference,
            candidate,
            context=f"real dataset ({len(df)} candles)",
        )


if __name__ == "__main__":
    unittest.main()