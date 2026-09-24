
"""
Phase 3 Strategy Adapter Smoke Test
===================================

Purpose:
- Load real OHLC data.
- Feed candles one at a time.
- Verify StrategyAdapter processes candles causally.
- Verify persistent adapter state survives across candles.
- Report triggered signals, pending signals, and adapter errors.

This test does NOT:
- execute paper trades
- connect to a broker
- modify strategy logic
- modify the dataset
"""

from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd

from phase_03_paper.market.engine import Candle
from phase_03_paper.signals.adapter import StrategyAdapter


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE: str = "your_data_file.csv"

# Start with 500 candles for a fast smoke test.
MAX_CANDLES: int = 500


def main() -> None:
    print("=" * 70)
    print("PHASE 3 STRATEGY ADAPTER SMOKE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    file_path = Path(DATA_FILE)

    if not file_path.exists():
        print(
            f"\nFATAL: Dataset file not found at:"
            f"\n{file_path.resolve()}"
        )
        print(
            "\nPlease check DATA_FILE and make sure the CSV exists."
        )
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(
            f"\nFATAL: Failed to read CSV file: {exc}"
        )
        traceback.print_exc()
        return

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required - set(df.columns)

    if missing:
        print(
            f"\nFATAL: Missing required columns:"
            f" {sorted(missing)}"
        )
        return

    # --------------------------------------------------------
    # Validate timestamps
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )

    invalid_timestamp_count = int(
        df["timestamp"].isna().sum()
    )

    if invalid_timestamp_count:
        print(
            f"\nWARNING: Dropping "
            f"{invalid_timestamp_count} rows "
            f"with invalid timestamps."
        )

        df = df.dropna(
            subset=["timestamp"]
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df.sort_values("timestamp")
        .head(MAX_CANDLES)
        .reset_index(drop=True)
    )

    if df.empty:
        print(
            "\nFATAL: Dataset is empty after processing."
        )
        return

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print("\nDataset loaded successfully.")

    print(
        f"Dataset Range : "
        f"{df['timestamp'].iloc[0]} -> "
        f"{df['timestamp'].iloc[-1]}"
    )

    print(
        f"Candles target: {len(df):,}"
    )

    print("-" * 70)

    # --------------------------------------------------------
    # Create adapter
    # --------------------------------------------------------

    adapter = StrategyAdapter()

    triggered_count = 0
    processing_failed = False

    # --------------------------------------------------------
    # Causal one-candle-at-a-time replay
    # --------------------------------------------------------

    for row in df.itertuples(index=False):

        try:
            candle = Candle(
                timestamp=pd.Timestamp(
                    row.timestamp
                ),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
            )

            events = adapter.on_candle(candle)

            if events:
                triggered_count += len(events)

                for event in events:

                    signal_type = getattr(
                        event.signal,
                        "signal_type",
                        "N/A",
                    )

                    if hasattr(
                        signal_type,
                        "value",
                    ):
                        signal_type = signal_type.value

                    print(
                        f"TRIGGERED | "
                        f"time={candle.timestamp} | "
                        f"type={signal_type} | "
                        f"fill={event.fill_price:.5f} | "
                        f"risk={event.initial_risk:.5f}"
                    )

        except Exception as exc:

            processing_failed = True

            print(
                "\nCRITICAL ERROR while processing "
                f"candle {row.timestamp}: {exc}"
            )

            traceback.print_exc()
            break

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST RESULT SUMMARY")
    print("=" * 70)

    print(
        f"Candles supplied  : {len(df):,}"
    )

    print(
        f"Candles processed : "
        f"{adapter.candle_count:,}"
    )

    print(
        f"Triggered events  : "
        f"{triggered_count:,}"
    )

    print(
        f"Pending signals   : "
        f"{adapter.pending_count:,}"
    )

    print(
        f"Adapter errors    : "
        f"{len(adapter.errors):,}"
    )

    # --------------------------------------------------------
    # Error report
    # --------------------------------------------------------

    if adapter.errors:
        print()
        print(
            "ADAPTER ERROR LOG "
            "(First 10):"
        )

        for error in adapter.errors[:10]:
            print(f"  - {error}")

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print()

    if processing_failed:
        print(
            "[FAIL]: Candle processing stopped "
            "because of a runtime exception."
        )
        return

    if adapter.candle_count != len(df):
        print(
            "[FAIL]: Candle count mismatch."
        )
        print(
            f"Expected: {len(df)}"
        )
        print(
            f"Processed: {adapter.candle_count}"
        )
        return

    if adapter.errors:
        print(
            "[FAIL]: StrategyAdapter reported "
            "internal errors."
        )
        return

    print(
        "[PASS]: StrategyAdapter processed "
        "all candles causally with zero errors."
    )


if __name__ == "__main__":
    main()

