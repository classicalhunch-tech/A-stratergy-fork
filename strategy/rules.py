import pandas as pd
from datetime import time

from strategy.structure import Trend


class TradingRules:
    def __init__(
        self,
        start_hour_london=7,
        end_hour_london=16,
        start_hour_ny=12,
        end_hour_ny=21,
        stop_loss_buffer=2.0,  # named/configurable, not buried as a magic number
    ):
        # Allowed trading hours (UTC).
        self.london_open = time(start_hour_london, 0)
        self.london_close = time(end_hour_london, 0)
        self.ny_open = time(start_hour_ny, 0)
        self.ny_close = time(end_hour_ny, 0)
        self.stop_loss_buffer = stop_loss_buffer

    def is_valid_session(self, timestamp: pd.Timestamp) -> bool:
        """Check if the current candle falls within London or New York sessions.

        Assumes timestamp is UTC (or tz-naive UTC) — comparison silently
        strips tzinfo, so mismatched timezones will not raise an error.
        """
        t = timestamp.time()
        in_london = self.london_open <= t <= self.london_close
        in_ny = self.ny_open <= t <= self.ny_close
        return in_london or in_ny

    def _closest_zone(self, candle: pd.Series, zones: list, zone_type: str):
        """Return the unmitigated zone of zone_type nearest to the current
        close, or None. Nearest = smallest distance from close to the zone's
        near edge, so the freshest relevant zone wins over a farther one
        earlier in the list."""
        candidates = [z for z in zones if z["type"] == zone_type and not z["mitigated"]]
        if not candidates:
            return None

        def distance(zone):
            if zone_type == "DEMAND":
                return abs(candle["close"] - zone["price_top"])
            return abs(candle["close"] - zone["price_bottom"])

        return min(candidates, key=distance)

    def evaluate_setup(self, candle: pd.Series, current_trend: Trend, active_zones: list) -> dict:
        """
        Evaluates whether a trade setup is valid based on Session Time,
        Market Structure Trend, and Supply/Demand zone mitigation.
        """
        timestamp = candle.name

        # Rule 1: Session Filter (London & New York Only)
        if not self.is_valid_session(timestamp):
            return {"action": "HOLD", "reason": "Outside London/NY Session"}

        # Rule 2: Trend Alignment & Supply/Demand Zone Interaction
        if current_trend == Trend.BULLISH:
            zone = self._closest_zone(candle, active_zones, "DEMAND")
            if zone and candle["low"] <= zone["price_top"] and candle["close"] >= zone["price_bottom"]:
                zone["mitigated"] = True
                return {
                    "action": "BUY",
                    "entry": candle["close"],
                    "stop_loss": zone["price_bottom"] - self.stop_loss_buffer,
                    "reason": "Bullish Trend + Demand Zone Mitigation in active session",
                }

        elif current_trend == Trend.BEARISH:
            zone = self._closest_zone(candle, active_zones, "SUPPLY")
            if zone and candle["high"] >= zone["price_bottom"] and candle["close"] <= zone["price_top"]:
                zone["mitigated"] = True
                return {
                    "action": "SELL",
                    "entry": candle["close"],
                    "stop_loss": zone["price_top"] + self.stop_loss_buffer,
                    "reason": "Bearish Trend + Supply Zone Mitigation in active session",
                }

        return {"action": "HOLD", "reason": "No setup criteria met"}