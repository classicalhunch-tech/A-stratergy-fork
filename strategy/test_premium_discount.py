import sys
sys.path.insert(0, ".")

from dataclasses import dataclass
from strategy.swings import find_swings
from strategy.structure import analyze_structure
from strategy.zones import find_zones, ZoneType
from strategy.premium_discount import calculate_equilibrium, filter_zones, filter_zone_by_premium_discount
from strategy.test_structure import generate_bullish_sequence_data


# Lightweight mock zone for testing the filter logic independently
@dataclass
class MockZone:
    price_top: float
    price_bottom: float
    zone_type: ZoneType


def test_premium_discount_filter():
    print("\n--- TESTING PREMIUM / DISCOUNT FILTER (STRICT) ---")
    
    df = generate_bullish_sequence_data()
    swings = find_swings(df)
    breaks, classified, final_trend = analyze_structure(df, swings)
    zones = find_zones(df, breaks)

    # Active dealing range for our bullish sequence dataset
    swing_low = 80.0
    swing_high = 140.0

    eq = calculate_equilibrium(swing_high, swing_low)
    print(f"Active Range: Low={swing_low}, High={swing_high} | Equilibrium (50%) = {eq}")

    filtered_zones = filter_zones(zones, swing_high, swing_low)
    print(f"Zones before filter: {len(zones)}, Zones after filter: {len(filtered_zones)}")

    # Assertions
    assert eq == 110.0, f"Expected equilibrium 110.0, got {eq}"
    
    # Test individual mock zones using our clean MockZone dataclass
    valid_demand = MockZone(price_top=100.0, price_bottom=90.0, zone_type=ZoneType.DEMAND)
    invalid_demand = MockZone(price_top=120.0, price_bottom=115.0, zone_type=ZoneType.DEMAND)
    
    valid_supply = MockZone(price_top=135.0, price_bottom=120.0, zone_type=ZoneType.SUPPLY)
    invalid_supply = MockZone(price_top=105.0, price_bottom=95.0, zone_type=ZoneType.SUPPLY)

    assert filter_zone_by_premium_discount(valid_demand, swing_high, swing_low) is True, "Valid discount demand should pass"
    assert filter_zone_by_premium_discount(invalid_demand, swing_high, swing_low) is False, "Premium demand should fail strict filter"
    
    assert filter_zone_by_premium_discount(valid_supply, swing_high, swing_low) is True, "Valid premium supply should pass"
    assert filter_zone_by_premium_discount(invalid_supply, swing_high, swing_low) is False, "Discount supply should fail strict filter"

    print("✅ STRICT PREMIUM / DISCOUNT FILTER TEST PASSED!")


if __name__ == "__main__":
    test_premium_discount_filter()