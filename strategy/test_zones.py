import sys
sys.path.insert(0, ".")

import pandas as pd

from strategy.swings import find_swings
from strategy.structure import analyze_structure
from strategy.zones import find_zones, update_mitigation, ZoneType
from strategy.test_structure import generate_bullish_sequence_data, generate_bearish_sequence_data


def test_bullish_zone_detection():
    print("\n--- TESTING ZONE DETECTION (BULLISH BASE) ---")
    df = generate_bullish_sequence_data()
    swings = find_swings(df)
    breaks, classified, final_trend = analyze_structure(df, swings)
    zones = find_zones(df, breaks)

    print(f"Zones Detected ({len(zones)}):")
    for z in zones:
        print(f"  {z}")

    demand_zones = [z for z in zones if z.zone_type == ZoneType.DEMAND]
    assert len(demand_zones) >= 1, "Expected at least one DEMAND zone"

    z = demand_zones[0]
    assert z.price_top == 125.0, f"Expected zone top 125.0, got {z.price_top}"
    assert z.price_bottom == 80.0, f"Expected zone bottom 80.0, got {z.price_bottom}"
    print("✅ ZONE DETECTION TEST PASSED!")


def test_bearish_zone_detection():
    print("\n--- TESTING ZONE DETECTION (BEARISH BASE) ---")
    df = generate_bearish_sequence_data()
    swings = find_swings(df)
    breaks, classified, final_trend = analyze_structure(df, swings)
    zones = find_zones(df, breaks)

    print(f"Zones Detected ({len(zones)}):")
    for z in zones:
        print(f"  {z}")

    assert len(zones) == 2, f"Expected exactly 2 zones, got {len(zones)}"

    supply = zones[0]
    assert supply.zone_type == ZoneType.SUPPLY, f"Expected first zone to be SUPPLY, got {supply.zone_type}"
    assert supply.price_top == 90.0, f"Expected SUPPLY top 90.0, got {supply.price_top}"
    assert supply.price_bottom == 74.0, f"Expected SUPPLY bottom 74.0, got {supply.price_bottom}"

    demand = zones[1]
    assert demand.zone_type == ZoneType.DEMAND, f"Expected second zone to be DEMAND, got {demand.zone_type}"
    assert demand.price_top == 85.0, f"Expected DEMAND top 85.0, got {demand.price_top}"
    assert demand.price_bottom == 55.0, f"Expected DEMAND bottom 55.0, got {demand.price_bottom}"

    print("✅ BEARISH ZONE DETECTION TEST PASSED!")


def test_mitigation_tracking():
    print("\n--- TESTING MITIGATION TRACKING ---")

    # Bullish sequence: walk every candle forward, updating mitigation.
    df = generate_bullish_sequence_data()
    swings = find_swings(df)
    breaks, classified, final_trend = analyze_structure(df, swings)
    zones = find_zones(df, breaks)

    for ts, row in df.iterrows():
        update_mitigation(zones, row, ts)

    zone1, zone2, zone3 = zones
    assert zone1.mitigated is True, "Zone 1 (80-125) should be mitigated"
    assert zone1.mitigated_at == pd.Timestamp("2026-01-01 10:00"), \
        f"Expected zone 1 mitigated at 10:00, got {zone1.mitigated_at}"

    assert zone2.mitigated is True, "Zone 2 (100-140) should be mitigated"
    assert zone2.mitigated_at == pd.Timestamp("2026-01-01 16:00"), \
        f"Expected zone 2 mitigated at 16:00, got {zone2.mitigated_at}"

    assert zone3.mitigated is False, "Zone 3 (105-143) should remain unmitigated (no candles after it forms)"

    print("✅ BULLISH MITIGATION TEST PASSED!")

    # Bearish sequence: same walk-forward check.
    df_b = generate_bearish_sequence_data()
    swings_b = find_swings(df_b)
    breaks_b, classified_b, final_trend_b = analyze_structure(df_b, swings_b)
    zones_b = find_zones(df_b, breaks_b)

    for ts, row in df_b.iterrows():
        update_mitigation(zones_b, row, ts)

    supply_zone, demand_zone = zones_b
    assert supply_zone.mitigated is True, "SUPPLY zone (74-90) should be mitigated"
    assert supply_zone.mitigated_at == pd.Timestamp("2026-01-01 17:00"), \
        f"Expected SUPPLY zone mitigated at 17:00, got {supply_zone.mitigated_at}"

    assert demand_zone.mitigated is False, "DEMAND zone (55-85) should remain unmitigated (no candles after it forms)"

    print("✅ BEARISH MITIGATION TEST PASSED!")


if __name__ == "__main__":
    test_bullish_zone_detection()
    test_bearish_zone_detection()
    test_mitigation_tracking()
    print("\n✅ ALL ZONE DETECTION TESTS PASSED!")