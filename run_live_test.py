import pandas as pd
from strategy.backtest import run_backtest

print("Loading historical data...")
df = pd.read_csv(
    "your_data_file.csv",
    parse_dates=True,
    index_col=0,
)
print(f"Loaded {len(df):,} rows")
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")

# Quick sanity check before burning time on a full run
required = {"open", "high", "low", "close"}
missing = required - {c.lower() for c in df.columns}
if missing:
    raise ValueError(f"CSV is missing required OHLC columns: {missing}")

print("\nRunning backtest over historical data...")
result = run_backtest(
    df=df,
    max_bars_to_retest=20,
    reward_multiple=2.0,
    stop_buffer=0.5,
    entry_mode="midpoint",
)

print("\n" + "=" * 40)
print("--- BACKTEST PERFORMANCE SUMMARY ---")
print("=" * 40)
print(f"Total Signals Generated : {result.total_signals_generated}")
print(f"Total Trades Triggered  : {result.total_trades_triggered}")
print(f"Total Invalidated       : {result.total_invalidated}")
print(f"Total Expired           : {result.total_expired}")

closed_trades = [t for t in result.trades if t.result_status in {"WIN", "LOSS"}]
open_trades = [t for t in result.trades if t.result_status == "OPEN"]
wins = [t for t in closed_trades if t.result_status == "WIN"]
losses = [t for t in closed_trades if t.result_status == "LOSS"]
total_r = sum(t.r_multiple for t in closed_trades)

print(f"Closed Trades           : {len(closed_trades)} (Wins: {len(wins)}, Losses: {len(losses)})")
print(f"Still OPEN at dataset end: {len(open_trades)}")
print(f"Win Rate                : {result.win_rate:.2f}%")
print(f"Expectancy              : {result.expectancy:+.2f}R per trade")
print(f"Total Cumulative R      : {total_r:+.2f}R")
print(f"Errors Encountered      : {len(result.errors)}")

if result.errors:
    print("\n" + "-" * 40)
    print("--- FIRST 5 ERRORS ---")
    print("-" * 40)
    for error in result.errors[:5]:
        print(error)

# ============================================================
# ROBUST POSITION OVERLAP DIAGNOSTIC
# ============================================================
print("\n" + "=" * 40)
print("--- PORTFOLIO CONCURRENCY DIAGNOSTIC ---")
print("=" * 40)

max_concurrent_positions = 0
valid_trades = [t for t in result.trades if t.entry_time is not None]

if valid_trades:
    events = []
    for t in valid_trades:
        entry_t = t.entry_time
        exit_t = t.exit_time if t.exit_time is not None else pd.Timestamp.max
        events.append((entry_t, 1))
        events.append((exit_t, -1))

    # Sort chronologically; on a tie, process exits (-1) before entries (+1)
    # so a trade closing and another opening on the same bar doesn't
    # double-count as 2 concurrent positions.
    events.sort(key=lambda x: (x[0], x[1]))

    current_open = 0
    for timestamp, action in events:
        current_open += action
        max_concurrent_positions = max(max_concurrent_positions, current_open)

print(f"Max concurrent open positions observed: {max_concurrent_positions}")
if max_concurrent_positions > 1:
    print("NOTE: engine currently allows overlapping positions —")
    print("this backtest is NOT single-position-at-a-time.")
print("=" * 40)