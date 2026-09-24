"""
compile_master.py

Phase 1 historical-data compiler.

Purpose:
    Convert your raw historical OHLC CSV into one clean,
    canonical dataset for the trading research pipeline.

Input:
    your_data_file.csv

Output:
    data/master_historical_data.csv

This script cleans and validates data.
It does NOT modify the original source CSV.
"""

from pathlib import Path
import os

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_FILE = Path("your_data_file.csv")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "master_historical_data.csv"


# ============================================================
# HELPERS
# ============================================================

def find_column(columns, keywords, exclude=None, used=None):
    """
    Find the column that best matches one of the supplied keywords.

    BUG FIX: the original version did a plain substring search with
    no regard for word boundaries, exact matches, or columns already
    claimed by another field. That silently breaks on very common
    exchange export formats (e.g. Binance klines: open_time, open,
    high, low, close, volume, close_time, ...):

      - find_column(["open"])  would match "open_time" (the FIRST
        column containing the substring "open") instead of the real
        "open" price column.
      - find_column(["close"]) would match "close_time" instead of
        "close".
      - find_column(["timestamp","...,"time"]) could just as easily
        grab "open_time" instead of an intended "datetime" column.

    That means the compiler could silently write epoch timestamps
    into the open/close PRICE columns with no error raised — a
    serious silent data-corruption bug. This version:

      1. Prefers an EXACT (case-insensitive) column-name match.
      2. Falls back to substring matching, but skips columns that
         look like a different field (e.g. skips "*_time"/"*_date"
         columns when hunting for a price field).
      3. Never returns a column that's already been claimed by
         another field (via `used`).
    """

    exclude = exclude or []
    used = used or set()

    columns_by_lower = {
        str(column).strip().lower(): column
        for column in columns
    }

    # Pass 1: exact match.
    for keyword in keywords:
        candidate = columns_by_lower.get(keyword)

        if candidate is not None and candidate not in used:
            return candidate

    # Pass 2: substring match, skipping excluded/used columns.
    for column in columns:

        if column in used:
            continue

        column_lower = str(column).strip().lower()

        if any(bad in column_lower for bad in exclude):
            continue

        for keyword in keywords:
            if keyword in column_lower:
                return column

    return None


def detect_timestamp_series(raw_series):
    """
    Parse a raw timestamp column into tz-naive pandas datetimes.

    BUG FIX: the original code always called
    `pd.to_datetime(raw, errors="coerce")` with no unit hint. Many
    exchange/broker exports store timestamps as raw epoch integers
    (seconds, milliseconds, or nanoseconds since 1970). Feeding those
    straight into pd.to_datetime without a `unit=` produces silently
    wrong dates (e.g. an epoch-ms value gets read as nanoseconds and
    lands somewhere in the 1970s, a few minutes after the epoch,
    instead of the correct date) — no exception, no NaT, just wrong
    data that passes every downstream check.

    This inspects the magnitude of numeric timestamp values and picks
    the matching unit (s / ms / ns) before parsing. String/ISO
    timestamps are left to pandas' normal parser.

    Also normalizes tz-aware timestamps to tz-naive (UTC-based) so
    downstream tooling never has to reconcile tz-aware vs tz-naive
    values in the same series (a real crash we hit and fixed in the
    dashboard's sort logic).
    """

    if pd.api.types.is_numeric_dtype(raw_series):

        sample = raw_series.dropna()

        if sample.empty:
            return pd.to_datetime(raw_series, errors="coerce")

        magnitude = sample.abs().median()

        if magnitude > 1e14:
            unit = "ns"
        elif magnitude > 1e11:
            unit = "ms"
        elif magnitude > 1e8:
            unit = "s"
        else:
            # Doesn't look like a recognizable epoch scale (too
            # small) — fall back to pandas' default numeric-to-date
            # interpretation rather than guessing.
            unit = None

        if unit is not None:
            parsed = pd.to_datetime(
                raw_series,
                unit=unit,
                errors="coerce",
            )
        else:
            parsed = pd.to_datetime(
                raw_series,
                errors="coerce",
            )

    else:
        parsed = pd.to_datetime(
            raw_series,
            errors="coerce",
        )

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)

    return parsed


def print_separator():
    print("-" * 70)


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("🧪 PHASE 1 — HISTORICAL DATA COMPILER")
print("=" * 70)
print()

print(f"📂 Source: {SOURCE_FILE}")
print(f"📁 Output: {OUTPUT_FILE}")
print()

if OUTPUT_FILE.exists():
    print(f"⚠️  Output file already exists and will be OVERWRITTEN.")
    print()

# ============================================================
# LOAD SOURCE DATA
# ============================================================

if not SOURCE_FILE.exists():
    print(f"❌ ERROR: Source file not found:")
    print(f"   {SOURCE_FILE.resolve()}")
    raise SystemExit(1)

try:
    df = pd.read_csv(SOURCE_FILE)
except Exception as exc:
    print(f"❌ ERROR: Could not read source CSV.")
    print(f"   {type(exc).__name__}: {exc}")
    raise SystemExit(1)

print(f"✅ Loaded source file: {len(df):,} rows")

if df.empty:
    print("❌ ERROR: Source CSV is empty.")
    raise SystemExit(1)

# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

df.columns = [
    str(column).strip().lower()
    for column in df.columns
]

print()
print("📋 Detected columns:")
print("   " + ", ".join(df.columns))

# ============================================================
# IDENTIFY REQUIRED COLUMNS
# ============================================================
#
# `used_columns` tracks columns already claimed so the same source
# column (e.g. "close_time") can never accidentally be assigned to
# two different destination fields (e.g. both "timestamp" AND
# swallowed into "close" price data). See find_column() docstring.

used_columns = set()

timestamp_col = find_column(
    df.columns,
    ["timestamp", "datetime", "date", "time"],
    used=used_columns,
)

if timestamp_col is not None:
    used_columns.add(timestamp_col)

# Price/volume columns explicitly exclude anything that looks like a
# *_time / *_date companion column (e.g. Binance's open_time,
# close_time) so they can never be mistaken for a price field.
price_exclude = ["time", "date"]

open_col = find_column(
    df.columns,
    ["open"],
    exclude=price_exclude,
    used=used_columns,
)

if open_col is not None:
    used_columns.add(open_col)

high_col = find_column(
    df.columns,
    ["high"],
    exclude=price_exclude,
    used=used_columns,
)

if high_col is not None:
    used_columns.add(high_col)

low_col = find_column(
    df.columns,
    ["low"],
    exclude=price_exclude,
    used=used_columns,
)

if low_col is not None:
    used_columns.add(low_col)

close_col = find_column(
    df.columns,
    ["close"],
    exclude=price_exclude,
    used=used_columns,
)

if close_col is not None:
    used_columns.add(close_col)

volume_col = find_column(
    df.columns,
    ["volume", "vol", "tick_volume"],
    exclude=price_exclude,
    used=used_columns,
)

if volume_col is not None:
    used_columns.add(volume_col)

print()
print("🔎 Column mapping:")
print(f"   Timestamp → {timestamp_col}")
print(f"   Open      → {open_col}")
print(f"   High      → {high_col}")
print(f"   Low       → {low_col}")
print(f"   Close     → {close_col}")
print(f"   Volume    → {volume_col}")

required = {
    "timestamp": timestamp_col,
    "open": open_col,
    "high": high_col,
    "low": low_col,
    "close": close_col,
}

missing = [
    name
    for name, column in required.items()
    if column is None
]

if missing:
    print()
    print("❌ ERROR: Missing required columns:")
    for name in missing:
        print(f"   - {name}")

    print()
    print("Available columns:")
    for column in df.columns:
        print(f"   - {column}")

    raise SystemExit(1)

# ============================================================
# BUILD MASTER DATAFRAME
# ============================================================

master = pd.DataFrame()

master["timestamp"] = detect_timestamp_series(
    df[timestamp_col]
)

master["open"] = pd.to_numeric(
    df[open_col],
    errors="coerce"
)

master["high"] = pd.to_numeric(
    df[high_col],
    errors="coerce"
)

master["low"] = pd.to_numeric(
    df[low_col],
    errors="coerce"
)

master["close"] = pd.to_numeric(
    df[close_col],
    errors="coerce"
)

# ============================================================
# VOLUME HANDLING
# ============================================================

if volume_col is not None:
    master["volume"] = pd.to_numeric(
        df[volume_col],
        errors="coerce"
    )

    volume_source = "SOURCE CSV"

else:
    # We deliberately do NOT invent fake volume values.
    master["volume"] = pd.NA

    volume_source = "NOT PROVIDED"

print()
print(f"📊 Volume source: {volume_source}")

# ============================================================
# INITIAL DATA QUALITY CHECK
# ============================================================

source_rows = len(master)

invalid_timestamp = master["timestamp"].isna().sum()

invalid_ohlc = master[
    [
        "open",
        "high",
        "low",
        "close",
    ]
].isna().any(axis=1).sum()

invalid_timestamp_rows = int(invalid_timestamp)
invalid_ohlc_rows = int(invalid_ohlc)

print()
print("🧹 Initial data-quality scan:")
print(f"   Invalid timestamps : {invalid_timestamp_rows:,}")
print(f"   Invalid OHLC rows  : {invalid_ohlc_rows:,}")

# ============================================================
# REMOVE INVALID ROWS
# ============================================================

before_cleaning = len(master)

master = master.dropna(
    subset=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]
).copy()

removed_invalid = before_cleaning - len(master)

print(f"   Removed invalid rows: {removed_invalid:,}")

if master.empty:
    print()
    print(
        "❌ ERROR: No valid candles remain after removing rows "
        "with unparseable timestamps or prices. Double-check the "
        "column mapping above — if it looks wrong, your source "
        "CSV likely uses column names this script doesn't recognize."
    )
    raise SystemExit(1)

# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

master = master.sort_values(
    "timestamp",
    kind="mergesort",
).reset_index(drop=True)

# ============================================================
# DUPLICATE TIMESTAMP CHECK
# ============================================================

duplicate_count = int(
    master["timestamp"].duplicated(keep="first").sum()
)

print()
print(f"🔁 Duplicate timestamps: {duplicate_count:,}")

if duplicate_count > 0:
    master = master.drop_duplicates(
        subset=["timestamp"],
        keep="first",
    ).reset_index(drop=True)

    print(
        f"   Removed duplicates: {duplicate_count:,}"
    )

# ============================================================
# OHLC STRUCTURAL VALIDATION
# ============================================================

invalid_price_logic = (
    (master["high"] < master["low"])
    |
    (master["high"] < master["open"])
    |
    (master["high"] < master["close"])
    |
    (master["low"] > master["open"])
    |
    (master["low"] > master["close"])
)

invalid_price_logic_count = int(
    invalid_price_logic.sum()
)

print()
print(
    f"📐 Invalid OHLC relationships: "
    f"{invalid_price_logic_count:,}"
)

if invalid_price_logic_count > 0:
    master = master.loc[
        ~invalid_price_logic
    ].reset_index(drop=True)

    print(
        f"   Removed invalid candles: "
        f"{invalid_price_logic_count:,}"
    )

# ============================================================
# POSITIVE PRICE VALIDATION
# ============================================================

negative_or_zero_price = (
    (master["open"] <= 0)
    |
    (master["high"] <= 0)
    |
    (master["low"] <= 0)
    |
    (master["close"] <= 0)
)

invalid_price_count = int(
    negative_or_zero_price.sum()
)

print(
    f"💰 Non-positive price rows: "
    f"{invalid_price_count:,}"
)

if invalid_price_count > 0:
    master = master.loc[
        ~negative_or_zero_price
    ].reset_index(drop=True)

# ============================================================
# VOLUME SANITY CHECK
# ============================================================
#
# We flag (but don't remove) negative volume — some brokers/feeds
# report signed "delta volume" in this field, so a negative value
# isn't automatically corrupt data the way a negative price is.

if volume_col is not None:

    negative_volume = (
        master["volume"] < 0
    ).fillna(False)

    negative_volume_count = int(negative_volume.sum())

    if negative_volume_count > 0:
        print(
            f"⚠️  Negative volume values: {negative_volume_count:,} "
            "(kept as-is — verify this is expected for your source)"
        )

# ============================================================
# TIMESTAMP GAP ANALYSIS
# ============================================================

gap_count = 0
largest_gap = None

if len(master) >= 2:
    timestamp_diff = master["timestamp"].diff()

    positive_diffs = timestamp_diff[
        timestamp_diff > pd.Timedelta(0)
    ]

    if not positive_diffs.empty:
        largest_gap = positive_diffs.max()

        # We report unusually large gaps rather than
        # deleting them. Weekends/session closures may
        # legitimately create gaps in real market data.
        median_interval = positive_diffs.median()

        if median_interval > pd.Timedelta(0):
            gap_threshold = median_interval * 10

            gap_count = int(
                (
                    positive_diffs
                    > gap_threshold
                ).sum()
            )

print()
print("⏱️ Timestamp continuity:")
print(f"   Suspected large gaps: {gap_count:,}")

if largest_gap is not None:
    print(
        f"   Largest observed gap: "
        f"{largest_gap}"
    )

# ============================================================
# FINAL EMPTY CHECK
# ============================================================

if master.empty:
    print()
    print(
        "❌ ERROR: No valid candles remain "
        "after cleaning."
    )
    raise SystemExit(1)

# ============================================================
# FINAL COLUMN ORDER
# ============================================================

master = master[
    [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
].copy()

# ============================================================
# FINAL VALIDATION
# ============================================================

assert not master.empty

assert master[
    [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]
].notna().all().all()

assert master["timestamp"].is_monotonic_increasing

assert (
    master["high"] >= master["low"]
).all()

assert (
    master["high"] >= master["open"]
).all()

assert (
    master["high"] >= master["close"]
).all()

assert (
    master["low"] <= master["open"]
).all()

assert (
    master["low"] <= master["close"]
).all()

# ============================================================
# SAVE MASTER DATASET
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    master.to_csv(
        OUTPUT_FILE,
        index=False
    )
except Exception as exc:
    print()
    print("❌ ERROR: Could not save master dataset.")
    print(f"   {type(exc).__name__}: {exc}")
    raise SystemExit(1)

# ============================================================
# FINAL REPORT
# ============================================================

print()
print_separator()

print("✅ MASTER DATASET CREATED SUCCESSFULLY")
print_separator()

print(f"Source rows:       {source_rows:,}")
print(f"Final candles:     {len(master):,}")
print(f"Rows removed:      {source_rows - len(master):,}")

print()

print(
    f"First timestamp:   "
    f"{master['timestamp'].iloc[0]}"
)

print(
    f"Last timestamp:    "
    f"{master['timestamp'].iloc[-1]}"
)

print()

print(
    f"Output file:       "
    f"{OUTPUT_FILE}"
)

print(
    f"Output size:       "
    f"{OUTPUT_FILE.stat().st_size:,} bytes"
)

print()

print("📊 FINAL COLUMNS:")
for column in master.columns:
    print(f"   ✓ {column}")

print()

print("🔐 INTEGRITY CHECKS:")
print("   ✓ Chronologically sorted")
print("   ✓ Duplicate timestamps removed")
print("   ✓ Invalid timestamps removed")
print("   ✓ Invalid OHLC rows removed")
print("   ✓ OHLC relationships validated")
print("   ✓ Positive prices validated")
print("   ✓ Timestamps normalized to tz-naive")
print("   ✓ Original source file preserved")

if volume_col is None:
    print(
        "   ⚠ Volume not supplied by source "
        "→ stored as empty"
    )
else:
    print("   ✓ Source volume preserved")

print()
print("=" * 70)
print("🏁 COMPILATION COMPLETE")
print("=" * 70)
print()