"""
Verification script for the retest_engine.py refactor.

Purpose:
- Run the current backtest against your_data_file.csv.
- Print the statistics needed to compare against pre-refactor results.
- Verify trade accounting consistency.
- Do NOT modify strategy logic.
"""

from __future__ import annotations

import pandas as pd

from strategy.backtest import run_backtest


DATA_FILE = "your_data_file.csv"


def normalize_status(status) -> str:
    """Return a consistent uppercase status string."""
    if hasattr(status, "value"):
        status = status.value

    return str(status).upper()


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

df.columns = [str(col).strip().lower() for col in df.columns]

required_columns = {"timestamp", "open", "high", "low", "close"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

before_count = len(df)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
)

df = df.dropna(subset=["timestamp"])

dropped_count = before_count - len(df)
if dropped_count:
    print(f"WARNING: dropped {dropped_count} row(s) with unparseable timestamps")

df = (
    df.set_index("timestamp")
    .sort_index()
    [["open", "high", "low", "close"]]
)

if df.empty:
    raise ValueError("Dataset is empty after loading.")

print(f"Loaded {len(df):,} candles from {DATA_FILE}")
print(f"Range: {df.index[0]} -> {df.index[-1]}")
print("-" * 60)


# ------------------------------------------------------------
# RUN BACKTEST
# ------------------------------------------------------------

print("Running backtest...")
print("-" * 60)

result = run_backtest(df)


# ------------------------------------------------------------
# DERIVE TRADE STATISTICS
# ------------------------------------------------------------

wins = sum(
    1
    for trade in result.trades
    if normalize_status(
        getattr(trade, "result_status", None)
    ) == "WIN"
)

losses = sum(
    1
    for trade in result.trades
    if normalize_status(
        getattr(trade, "result_status", None)
    ) == "LOSS"
)

open_trades = sum(
    1
    for trade in result.trades
    if normalize_status(
        getattr(trade, "result_status", None)
    ) == "OPEN"
)


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

print()
print("=" * 60)
print("BACKTEST VERIFICATION RESULTS")
print("=" * 60)

print(
    f"Total signals generated : "
    f"{result.total_signals_generated}"
)

print(
    f"Total trades triggered  : "
    f"{result.total_trades_triggered}"
)

print(
    f"Total invalidated       : "
    f"{result.total_invalidated}"
)

print(
    f"Total expired           : "
    f"{result.total_expired}"
)

print(f"Wins                    : {wins}")
print(f"Losses                  : {losses}")
print(f"Open (unresolved) trades: {open_trades}")
print(f"Win rate                : {result.win_rate}")
print(f"Expectancy              : {result.expectancy}")

print(f"Errors ({len(result.errors)}):")

for error in result.errors:
    print(f"  - {error}")


# ------------------------------------------------------------
# TRADE ACCOUNTING CHECK
# ------------------------------------------------------------

print("-" * 60)
print("TRADE ACCOUNTING CHECK")

triggered = result.total_trades_triggered
returned = len(result.trades)

print(f"Triggered trades : {triggered}")
print(f"Returned trades  : {returned}")

if triggered == returned:
    print("[PASS] Triggered count matches result.trades length.")
else:
    print(
        "[FAIL] Triggered count does NOT match "
        "result.trades length."
    )


# ------------------------------------------------------------
# STATUS CHECK
# ------------------------------------------------------------

valid_statuses = {"WIN", "LOSS", "OPEN"}

unexpected_statuses = {
    normalize_status(getattr(trade, "result_status", None))
    for trade in result.trades
    if normalize_status(
        getattr(trade, "result_status", None)
    ) not in valid_statuses
}

if unexpected_statuses:
    print(
        f"[FAIL] Unexpected trade statuses: "
        f"{sorted(unexpected_statuses)}"
    )
else:
    print("[PASS] All trade statuses are valid.")


print("-" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)