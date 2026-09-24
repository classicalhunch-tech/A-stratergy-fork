import sys
sys.path.insert(0, ".")

import pandas as pd
from strategy.swings import find_swings
from strategy.structure import analyze_structure
from strategy.zones import find_zones, update_mitigation
from strategy.test_structure import (
    generate_bullish_sequence_data,
    generate_bearish_sequence_data,
)


def test_mitigation_tracking():
    print("\n--- TESTING MITIGATION TRACKING ---")

    # 1. Bullish sequence test (Walk-forward event loop)
    df = generate_bullish_sequence_data()
    swings = find_swings(df)
    breaks, classified, final_trend = analyze_structure(df, swings)
    zones = find_zones(df, breaks)

    for ts, row in df.iterrows():
        update_mitigation(zones, row, ts)

    assert len(zones) == 3, f"Expected exactly 3 zones, got {len(zones)}"
    zone1, zone2, zone3 = zones[0], zones[1], zones[2]

    # Verify Zone 1 mitigation
    assert zone1.mitigated is True, "Zone 1 (80-125) should be mitigated"
    assert zone1.mitigated_at == pd.Timestamp("2026-01-01 10:00"), \
        f"Expected zone 1 mitigated at 10:00, got {zone1.mitigated_at}"

    # Verify Zone 2 mitigation
    assert zone2.mitigated is True, "Zone 2 (100-140) should be mitigated"
    assert zone2.mitigated_at == pd.Timestamp("2026-01-01 16:00"), \
        f"Expected zone 2 mitigated at 16:00, got {zone2.mitigated_at}"

    # Verify Zone 3 unmitigated state
    assert zone3.mitigated is False, "Zone 3 (105-143) should remain unmitigated"

    print("✅ BULLISH MITIGATION TEST PASSED!")

    # 2. Bearish sequence test (Walk-forward event loop)
    df_b = generate_bearish_sequence_data()
    swings_b = find_swings(df_b)
    breaks_b, classified_b, final_trend_b = analyze_structure(df_b, swings_b)
    zones_b = find_zones(df_b, breaks_b)

    for ts, row in df_b.iterrows():
        update_mitigation(zones_b, row, ts)

    assert len(zones_b) == 2, f"Expected exactly 2 zones, got {len(zones_b)}"
    supply_zone, demand_zone = zones_b[0], zones_b[1]

    # Verify Supply Zone mitigation
    assert supply_zone.mitigated is True, "SUPPLY zone (74-90) should be mitigated"
    assert supply_zone.mitigated_at == pd.Timestamp("2026-01-01 17:00"), \
        f"Expected SUPPLY zone mitigated at 17:00, got {supply_zone.mitigated_at}"

    # Verify Demand Zone unmitigated state
    assert demand_zone.mitigated is False, "DEMAND zone (55-85) should remain unmitigated"

    print("✅ BEARISH MITIGATION TEST PASSED!")
    print("\n✅ ALL MITIGATION TESTS PASSED!")


if __name__ == "__main__":
    test_mitigation_tracking()