"""
tests/test_signals.py

Comprehensive unit tests for strategy/signals.py
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd

from strategy.signals import (
    SignalType,
    SignalStatus,
    TradeSignal,
    generate_flip_zone_signals,
)
from strategy.structure import (
    StructureBreak,
    BreakType,
    Trend,
)
from strategy.zones import (
    Zone,
    ZoneType,
)
from strategy.liquidity import (
    LiquidityLevel,
    LiquidityType,
    LiquidityStatus,
)
from strategy.swings import (
    Swing,
    SwingType,
)


class TestSignalsEngineMasterclass(unittest.TestCase):
    def setUp(self):
        base_time = datetime(2026, 1, 1, 9, 0)
        data = []
        price = 100.0

        for i in range(30):
            timestamp = base_time + timedelta(minutes=5 * i)
            data.append(
                {
                    "timestamp": timestamp,
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price + 0.2,
                    "volume": 1000,
                }
            )
            price += 0.2

        self.df = pd.DataFrame(data)

    def tearDown(self):
        del self.df

    def _make_swing(
        self,
        price: float,
        bar_index: int,
        swing_type: SwingType,
    ):
        timestamp = self.df.loc[bar_index, "timestamp"]
        return Swing(
            swing_type=swing_type,
            price=price,
            formed_at=timestamp,
            confirmed_at=timestamp,
        )

    def _build_default_long_components(self):
        self.df.loc[5, "low"] = 94.0

        swing = self._make_swing(
            price=95.0,
            bar_index=3,
            swing_type=SwingType.LOW,
        )

        liquidity = LiquidityLevel(
            liquidity_type=LiquidityType.SELL_SIDE,
            price=95.0,
            formed_at=self.df.loc[3, "timestamp"],
            swing=swing,
            status=LiquidityStatus.SWEPT,
            sweep_bar_index=5,
            swept_at=self.df.loc[5, "timestamp"],
        )

        broken_swing = self._make_swing(
            price=101.0,
            bar_index=6,
            swing_type=SwingType.HIGH,
        )

        structure_break = StructureBreak(
            break_type=BreakType.CHOCH,
            direction=Trend.BULLISH,
            broken_swing=broken_swing,
            broken_at=self.df.loc[8, "timestamp"],
            break_price=102.0,
        )

        zone = Zone(
            zone_type=ZoneType.DEMAND,
            price_top=98.0,
            price_bottom=96.0,
            origin_start=self.df.loc[6, "timestamp"],
            origin_end=self.df.loc[7, "timestamp"],
            formed_from_break_at=self.df.loc[8, "timestamp"],
            mitigated=False,
        )

        return liquidity, structure_break, zone

    def _build_default_short_components(self):
        self.df.loc[5, "high"] = 106.0

        swing = self._make_swing(
            price=105.0,
            bar_index=3,
            swing_type=SwingType.HIGH,
        )

        liquidity = LiquidityLevel(
            liquidity_type=LiquidityType.BUY_SIDE,
            price=105.0,
            formed_at=self.df.loc[3, "timestamp"],
            swing=swing,
            status=LiquidityStatus.SWEPT,
            sweep_bar_index=5,
            swept_at=self.df.loc[5, "timestamp"],
        )

        broken_swing = self._make_swing(
            price=99.0,
            bar_index=6,
            swing_type=SwingType.LOW,
        )

        structure_break = StructureBreak(
            break_type=BreakType.CHOCH,
            direction=Trend.BEARISH,
            broken_swing=broken_swing,
            broken_at=self.df.loc[8, "timestamp"],
            break_price=98.0,
        )

        zone = Zone(
            zone_type=ZoneType.SUPPLY,
            price_top=104.0,
            price_bottom=102.0,
            origin_start=self.df.loc[6, "timestamp"],
            origin_end=self.df.loc[7, "timestamp"],
            formed_from_break_at=self.df.loc[8, "timestamp"],
            mitigated=False,
        )

        return liquidity, structure_break, zone

    def test_empty_inputs_return_empty_list(self):
        signals = generate_flip_zone_signals(
            liquidity_levels=[],
            structure_breaks=[],
            zones=[],
            df=self.df,
        )
        self.assertEqual(signals, [])

    def test_invalid_configuration_parameters_raise_value_error(self):
        invalid_configurations = [
            {"reward_multiple": 0},
            {"stop_buffer": -0.1},
            {"entry_mode": "invalid_mode"},
            {"max_bars_to_retest": 0},
        ]

        for config in invalid_configurations:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    generate_flip_zone_signals(
                        liquidity_levels=[],
                        structure_breaks=[],
                        zones=[],
                        df=self.df,
                        **config,
                    )

    def test_long_signal_pipeline_comprehensive(self):
        liquidity, structure_break, zone = self._build_default_long_components()

        # Trigger condition for LONG is simply: does `low` dip back
        # into the zone (<= price_top, 98)? `high` isn't checked by
        # the trigger logic; 101.0 here just keeps the bar's OHLC
        # roughly plausible. Baseline lows never reach <= 98 before
        # this override (they trend upward from setUp), so no bar-9
        # suppression is needed here the way the short test needs it.
        self.df.loc[10, "low"] = 97.0
        self.df.loc[10, "high"] = 101.0

        signals = generate_flip_zone_signals(
            liquidity_levels=[liquidity],
            structure_breaks=[structure_break],
            zones=[zone],
            df=self.df,
            entry_mode="midpoint",
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, SignalType.LONG)

    def test_short_signal_pipeline_comprehensive(self):
        liquidity, structure_break, zone = self._build_default_short_components()

        # The retest loop starts at break_idx + 1 = 9 and fires on the
        # FIRST bar where high >= price_bottom (102). Because setUp's
        # synthetic prices rise ~0.2/bar from 100, bar 9's baseline
        # high is already 102.8 -- which would trigger the signal
        # before bar 10 is ever reached, making the values below
        # meaningless without this override. Suppress bar 9 so bar 10
        # is the bar that actually exercises the retest.
        self.df.loc[9, "high"] = 100.0
        self.df.loc[10, "high"] = 103.5
        self.df.loc[10, "low"] = 102.5

        signals = generate_flip_zone_signals(
            liquidity_levels=[liquidity],
            structure_breaks=[structure_break],
            zones=[zone],
            df=self.df,
            entry_mode="midpoint",
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, SignalType.SHORT)


if __name__ == "__main__":
    unittest.main()