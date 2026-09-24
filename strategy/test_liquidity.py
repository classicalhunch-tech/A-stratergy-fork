"""
tests/test_liquidity.py

Unit tests for strategy/liquidity.py (detect_liquidity_levels,
update_liquidity_sweeps).
"""

import unittest
import pandas as pd

from strategy.swings import Swing, SwingType
from strategy.liquidity import (
    detect_liquidity_levels,
    update_liquidity_sweeps,
    LiquidityType,
    LiquidityStatus,
)


class TestLiquidityDetectionAndSweep(unittest.TestCase):

    def setUp(self):
        self.high_swing = Swing(
            swing_type=SwingType.HIGH,
            price=120.0,
            formed_at=pd.Timestamp("2026-01-01 06:00", tz="UTC"),
            confirmed_at=pd.Timestamp("2026-01-01 08:00", tz="UTC"),
        )

        self.low_swing = Swing(
            swing_type=SwingType.LOW,
            price=60.0,
            formed_at=pd.Timestamp("2026-01-01 06:00", tz="UTC"),
            confirmed_at=pd.Timestamp("2026-01-01 08:00", tz="UTC"),
        )

        # df must be indexed by timestamp (DatetimeIndex), matching the
        # convention used across swings.py / structure.py / zones.py /
        # liquidity.py. Bars start strictly after confirmed_at (08:00),
        # since a level can't be swept by the bar that formed it.
        self.df = pd.DataFrame(
            {
                "open":  [105.0, 115.0, 90.0],
                "high":  [110.0, 125.0, 100.0],
                "low":   [70.0,  95.0,  55.0],
                "close": [100.0, 118.0, 90.0],
            },
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2026-01-01 09:00", tz="UTC"),
                    pd.Timestamp("2026-01-01 10:00", tz="UTC"),
                    pd.Timestamp("2026-01-01 11:00", tz="UTC"),
                ],
                name="timestamp",
            ),
        )

    def test_detection_creates_active_bsl_and_ssl(self):
        levels = detect_liquidity_levels(
            self.df,
            [self.high_swing, self.low_swing],
        )

        self.assertEqual(len(levels), 2)

        bsl = next(l for l in levels if l.liquidity_type == LiquidityType.BUY_SIDE)
        ssl = next(l for l in levels if l.liquidity_type == LiquidityType.SELL_SIDE)

        self.assertEqual(bsl.price, 120.0)
        self.assertEqual(ssl.price, 60.0)
        self.assertEqual(bsl.status, LiquidityStatus.ACTIVE)
        self.assertEqual(ssl.status, LiquidityStatus.ACTIVE)
        # formed_at comes from confirmed_at, not the swing's own formed_at.
        self.assertEqual(bsl.formed_at, self.high_swing.confirmed_at)
        self.assertEqual(ssl.formed_at, self.low_swing.confirmed_at)

    def test_sweep_evaluation_marks_levels_swept_at_correct_bar(self):
        levels = detect_liquidity_levels(
            self.df,
            [self.high_swing, self.low_swing],
        )

        updated = update_liquidity_sweeps(self.df, levels)

        bsl = next(l for l in updated if l.liquidity_type == LiquidityType.BUY_SIDE)
        ssl = next(l for l in updated if l.liquidity_type == LiquidityType.SELL_SIDE)

        # BSL (120.0) is swept when high > 120 -> the 10:00 bar (high=125).
        self.assertEqual(bsl.status, LiquidityStatus.SWEPT)
        self.assertEqual(
            bsl.swept_at, pd.Timestamp("2026-01-01 10:00", tz="UTC")
        )
        self.assertEqual(bsl.sweep_bar_index, 1)

        # SSL (60.0) is swept when low < 60 -> the 11:00 bar (low=55).
        self.assertEqual(ssl.status, LiquidityStatus.SWEPT)
        self.assertEqual(
            ssl.swept_at, pd.Timestamp("2026-01-01 11:00", tz="UTC")
        )
        self.assertEqual(ssl.sweep_bar_index, 2)

    def test_level_not_swept_remains_active(self):
        # Tighter df where price never reaches either level.
        calm_df = self.df.copy()
        calm_df["high"] = [108.0, 112.0, 109.0]
        calm_df["low"] = [95.0, 96.0, 94.0]

        levels = detect_liquidity_levels(
            calm_df,
            [self.high_swing, self.low_swing],
        )

        updated = update_liquidity_sweeps(calm_df, levels)

        for level in updated:
            self.assertEqual(level.status, LiquidityStatus.ACTIVE)
            self.assertIsNone(level.swept_at)
            self.assertIsNone(level.sweep_bar_index)


if __name__ == "__main__":
    unittest.main()