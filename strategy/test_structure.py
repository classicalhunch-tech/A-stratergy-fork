import sys
import pandas as pd

sys.path.insert(0, ".")

from strategy.swings import find_swings
from strategy.structure import analyze_structure, Trend, BreakType, SwingLabel


def generate_bullish_sequence_data() -> pd.DataFrame:
    """
    Tests Bullish Sequence:
    1. High 1 (130) & Low 1 (80) established.
    2. Close 135 > 130 -> INITIAL_BREAK (BULLISH) [Consumes High 130].
    3. High 2 (140) & Low 2 (100) confirmed.
    4. Close 142 > 140 -> BULLISH BOS! [Consumes High 140, Protected Low = 100].
    5. Close 143 > 140 on next candle -> MUST NOT TRIGGER EXTRA BOS (140 is consumed).
    6. Close 92 < 100 -> BEARISH CHOCH! Trend flips to BEARISH.
    """
    data = [
        # --- BASELINE ---
        [100, 105, 95, 90],     # 00:00
        [90,  94,  88, 89],     # 01:00

        # --- HIGH 1 (130) -> INITIAL_HIGH ---
        [89,  120, 89, 115],    # 02:00
        [115, 130, 110, 125],   # 03:00 - High 130.0
        [125, 125, 112, 114],   # 04:00
        [114, 114, 100, 105],   # 05:00 - Confirm High 130.0

        # --- LOW 1 (80) -> INITIAL_LOW ---
        [105, 106, 80,  82],    # 06:00 - Low 80.0
        [82,  95,  82,  92],    # 07:00
        [92,  102, 91,  100],   # 08:00 - Confirm Low 80.0

        # --- BULLISH INITIAL BREAK (Close 135 > High 130) ---
        [100, 136, 99,  135],   # 09:00 - INITIAL_BREAK (BULLISH), consumes 130.0

        # --- HIGH 2 (140) -> HH ---
        [135, 140, 120, 122],   # 10:00 - High 140.0
        [122, 122, 110, 115],   # 11:00 - Confirm High 140.0 (HH)

        # --- LOW 2 (100) -> HL ---
        [115, 116, 100, 105],   # 12:00 - Low 100.0
        [105, 118, 105, 115],   # 13:00
        [115, 120, 112, 118],   # 14:00 - Confirm Low 100.0 (HL)

        # --- BULLISH BOS (Close 142 > High 140) ---
        [118, 142, 117, 142],   # 15:00 - BULLISH BOS! Consumes 140.0

        # --- CONSECUTIVE CANDLE ABOVE 140 (Should NOT trigger double BOS) ---
        [142, 143, 138, 143],   # 16:00 - Close 143 (No new BOS because 140 is consumed)

        # --- BEARISH CHOCH (Close 92 < Protected Low 100) ---
        [143, 143, 90,  92],    # 17:00 - BEARISH CHOCH!
    ]

    dates = pd.date_range(start="2026-01-01 00:00", periods=len(data), freq="1h")
    return pd.DataFrame(data, columns=["open", "high", "low", "close"], index=dates)


def generate_bearish_sequence_data() -> pd.DataFrame:
    """
    Tests Bearish Sequence:
    1. Bootstrap commits to seeking_high first (close at 01:00 < first_low),
       so the early Low-70 wick (candle 2) is never tracked as a candidate —
       this is the documented bootstrap limitation, not a bug.
    2. High 1 (120) confirms first -> INITIAL_HIGH.
    3. Low 2 (60) confirms next -> INITIAL_LOW (first low ever tracked).
    4. High 2 (90) confirms -> LH.
    5. Close 55 < 60 -> INITIAL_BREAK (BEARISH) [Consumes Low 60].
    6. Close 98 > 90 -> BULLISH CHOCH! Trend flips to BULLISH.
    (No BOS occurs in this dataset — only one confirmed low exists
    before the reversal, so there's no second lower-low to break.)
    """
    data = [
        # --- BASELINE ---
        [100, 105, 95, 96],     # 00:00
        [96,  98,  90, 91],     # 01:00

        # --- Low 70 wick here is NEVER tracked (bootstrap commits to seeking_high) ---
        [91,  92,  70, 75],     # 02:00
        [75,  85,  76, 82],     # 03:00
        [82,  88,  80, 85],     # 04:00
        [85,  95,  84, 90],     # 05:00

        # --- HIGH 1 (120) -> INITIAL_HIGH ---
        [90,  120, 89, 115],    # 06:00 - Swing High 120.0
        [115, 115, 100, 102],   # 07:00 - High 115 < 120
        [102, 104, 98,  99],    # 08:00 - Confirms High 120.0

        [99,  100, 62, 65],     # 09:00

        # --- LOW 2 (60) -> INITIAL_LOW ---
        [65,  75,  60, 72],     # 10:00 - Swing Low 60.0
        [72,  78,  68, 76],     # 11:00
        [76,  80,  70, 75],     # 12:00
        [75,  90,  74, 82],     # 13:00 - Confirms Low 60.0

        # --- HIGH 2 (90) -> LH ---
        [82,  85,  70, 72],     # 14:00
        [72,  73,  68, 69],     # 15:00 - Confirms High 90.0 (LH)

        # --- BEARISH INITIAL_BREAK (Close 55 < Low 60) ---
        [69,  70,  55, 55],     # 16:00

        # --- BULLISH CHOCH (Close 98 > Protected High 90) ---
        [55,  98,  54, 98],     # 17:00
    ]

    dates = pd.date_range(start="2026-01-01 00:00", periods=len(data), freq="1h")
    return pd.DataFrame(data, columns=["open", "high", "low", "close"], index=dates)


def test_bullish_sequence():
    print("\n--- 1. TESTING BULLISH SEQUENCE (INITIAL -> BOS -> CHOCH) ---")
    df = generate_bullish_sequence_data()
    swings = find_swings(df)
    breaks, classified, final_trend = analyze_structure(df, swings)

    print(f"Swings Classified ({len(classified)}):")
    for c in classified:
        print(f"  {c.label.value:<14} | Type: {c.swing.swing_type.value.upper():<4} | Price: {c.swing.price}")

    print(f"\nStructure Breaks Detected ({len(breaks)}):")
    for b in breaks:
        print(f"  {b}")

    # --- STRICT ASSERTIONS ---
    assert len(breaks) == 3, f"Expected exactly 3 breaks, got {len(breaks)}"

    # Break 1: INITIAL_BREAK at 130.0
    assert breaks[0].break_type == BreakType.INITIAL_BREAK
    assert breaks[0].direction == Trend.BULLISH
    assert breaks[0].broken_swing.price == 130.0
    assert breaks[0].break_price == 135.0

    # Break 2: BULLISH BOS at 140.0
    assert breaks[1].break_type == BreakType.BOS
    assert breaks[1].direction == Trend.BULLISH
    assert breaks[1].broken_swing.price == 140.0
    assert breaks[1].break_price == 142.0

    # Break 3: BEARISH CHOCH at 100.0
    assert breaks[2].break_type == BreakType.CHOCH
    assert breaks[2].direction == Trend.BEARISH
    assert breaks[2].broken_swing.price == 100.0
    assert breaks[2].break_price == 92.0

    assert final_trend == Trend.BEARISH
    print("✅ BULLISH SEQUENCE STRICT ASSERTIONS PASSED PERFECTLY!")


def test_bearish_sequence():
    print("\n--- 2. TESTING BEARISH SEQUENCE (INITIAL -> CHOCH) ---")
    df = generate_bearish_sequence_data()
    swings = find_swings(df)
    breaks, classified, final_trend = analyze_structure(df, swings)

    print(f"Swings Classified ({len(classified)}):")
    for c in classified:
        print(f"  {c.label.value:<14} | Type: {c.swing.swing_type.value.upper():<4} | Price: {c.swing.price}")

    print(f"\nStructure Breaks Detected ({len(breaks)}):")
    for b in breaks:
        print(f"  {b}")

    # --- STRICT ASSERTIONS ---
    assert len(breaks) == 2, f"Expected exactly 2 breaks, got {len(breaks)}"

    # Break 1: BEARISH INITIAL_BREAK at 60.0
    assert breaks[0].break_type == BreakType.INITIAL_BREAK
    assert breaks[0].direction == Trend.BEARISH
    assert breaks[0].broken_swing.price == 60.0
    assert breaks[0].break_price == 55.0

    # Break 2: BULLISH CHOCH at 90.0
    assert breaks[1].break_type == BreakType.CHOCH
    assert breaks[1].direction == Trend.BULLISH
    assert breaks[1].broken_swing.price == 90.0
    assert breaks[1].break_price == 98.0

    assert final_trend == Trend.BULLISH
    print("✅ BEARISH SEQUENCE STRICT ASSERTIONS PASSED PERFECTLY!")


if __name__ == "__main__":
    print("=== RUNNING DETERMINISTIC SMC STRUCTURE STATE MACHINE TESTS ===")
    test_bullish_sequence()
    test_bearish_sequence()
    print("\n✅ ALL DUAL-DIRECTION STRUCTURE TESTS PASSED WITH ABSOLUTE PRECISION!")