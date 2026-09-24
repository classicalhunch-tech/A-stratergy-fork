import sys
import pandas as pd

# Ensure python can import from strategy package when running from project root
sys.path.insert(0, ".")

from strategy.swings import find_swings, SwingType


def generate_multi_swing_data() -> pd.DataFrame:
    """
    Controlled OHLC dataset testing complete state alternation:
    HIGH -> LOW -> HIGH -> LOW
    """
    data = [
        # Open, High, Low, Close
        # --- BOOTSTRAP: Close 90 < 95 -> enter seeking_high cleanly ---
        [100, 105, 95, 90],    # 00:00 - First candle range (95-105)
        [90,  94,  88, 89],    # 01:00 - Close 89 < 95 -> confirms seeking_high state baseline
        
        # --- SWING HIGH 1 (Target: 130.0) ---
        [89,  120, 89, 115],   # 02:00 - Push up
        [115, 130, 110, 125],  # 03:00 - Candidate High = 130.0
        [125, 125, 112, 114],  # 04:00 - Pullback Low = 112.0 (Protection)
        [114, 114, 100, 105],  # 05:00 - Close 105 < 112 -> CONFIRM HIGH @ 130.0 -> seek low

        # --- SWING LOW 1 (Target: 80.0) ---
        [105, 106, 80,  82],   # 06:00 - Candidate Low = 80.0
        [82,  95,  82,  92],   # 07:00 - Pullback High = 95.0 (Protection)
        [92,  102, 91,  100],  # 08:00 - Close 100 > 95 -> CONFIRM LOW @ 80.0 -> seek high

        # --- SWING HIGH 2 (Target: 140.0) ---
        [100, 140, 99,  135],  # 09:00 - Candidate High = 140.0
        [135, 135, 120, 122],  # 10:00 - Pullback Low = 120.0 (Protection)
        [122, 122, 110, 115],  # 11:00 - Close 115 < 120 -> CONFIRM HIGH @ 140.0 -> seek low

        # --- SWING LOW 2 (Target: 70.0) ---
        [115, 116, 70,  72],   # 12:00 - Candidate Low = 70.0
        [72,  85,  72,  82],   # 13:00 - Pullback High = 85.0 (Protection)
        [82,  92,  81,  90],   # 14:00 - Close 90 > 85 -> CONFIRM LOW @ 70.0 -> seek high
    ]
    
    dates = pd.date_range(start="2026-01-01 00:00", periods=len(data), freq="1h")
    return pd.DataFrame(data, columns=["open", "high", "low", "close"], index=dates)


def run_test():
    df = generate_multi_swing_data()
    print("=== RUNNING MULTI-SWING ALTERNATION TEST ===")
    
    swings = find_swings(df)
    
    print(f"\nTotal swings detected: {len(swings)}\n")
    for s in swings:
        print(
            f"Type: {s.swing_type.value.upper():<4} | "
            f"Price: {s.price:<6.1f} | "
            f"Formed At: {s.formed_at.strftime('%H:%M')} | "
            f"Confirmed At: {s.confirmed_at.strftime('%H:%M')}"
        )

    expected_types = [SwingType.HIGH, SwingType.LOW, SwingType.HIGH, SwingType.LOW]
    expected_prices = [130.0, 80.0, 140.0, 70.0]

    assert len(swings) == 4, f"Expected 4 swings, got {len(swings)}"
    
    for idx, (s, exp_type, exp_price) in enumerate(zip(swings, expected_types, expected_prices)):
        assert s.swing_type == exp_type, f"Swing {idx} type mismatch: expected {exp_type}, got {s.swing_type}"
        assert s.price == exp_price, f"Swing {idx} price mismatch: expected {exp_price}, got {s.price}"
    
    print("\n✅ MULTI-SWING ALTERNATION TEST PASSED PERFECTLY!")


if __name__ == "__main__":
    run_test()