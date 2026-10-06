"""
tools/check_data_sessions.py

Shows how a data file lines up with the real gold market schedule: its
columns, and for each month when the market reopens after the weekend and
when the daily break starts and ends, in the timestamps stored in the file.

How to read it: if the file is in true UTC and your broker's gold schedule
follows New York time, the weekend reopen moves by ONE HOUR when the US
clocks change (2 Nov 2025 and 8 Mar 2026 in this data). A reopen time that
never moves all year suggests a fixed offset was applied to a server clock
that changes with daylight saving.

Usage:
    python tools/check_data_sessions.py --data data/real_gold_data_5m.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def mode_text(series: pd.Series) -> str:
    if len(series) == 0:
        return "-"
    return str(series.mode().iloc[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    df = pd.read_csv(Path(args.data), index_col=0, parse_dates=True)
    df.columns = [str(c).strip().lower() for c in df.columns]

    print("columns:", list(df.columns))

    if "spread" in df.columns:
        spread = pd.to_numeric(df["spread"], errors="coerce")
        print(
            "spread column (broker points): "
            f"median {spread.median():.1f}, 90th percentile {spread.quantile(0.9):.1f}, "
            f"max {spread.max():.1f}"
        )
    else:
        print("no spread column in this file (spread costs are assumed, not measured)")

    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    index = index.sort_values()

    times = pd.Series(index)
    step = times.diff()
    previous = times.shift(1)

    weekend = step > pd.Timedelta(hours=40)
    daily = (step > pd.Timedelta(minutes=30)) & (step <= pd.Timedelta(hours=6))

    weekend_reopen = times[weekend]
    weekend_last = previous[weekend]
    daily_reopen = times[daily]
    daily_last = previous[daily]

    months = sorted(set(times.dt.to_period("M")))

    print()
    print(
        f"{'month':8} {'weekend last candle':20} {'weekend reopen':16} "
        f"{'daily last candle':18} {'daily reopen':14}"
    )

    for month in months:
        weekend_mask = weekend_reopen.dt.to_period("M") == month
        daily_mask = daily_reopen.dt.to_period("M") == month

        print(
            f"{str(month):8} "
            f"{mode_text(weekend_last[weekend_mask].dt.strftime('%a %H:%M')):20} "
            f"{mode_text(weekend_reopen[weekend_mask].dt.strftime('%a %H:%M')):16} "
            f"{mode_text(daily_last[daily_mask].dt.strftime('%H:%M')):18} "
            f"{mode_text(daily_reopen[daily_mask].dt.strftime('%H:%M')):14}"
        )

    print()
    print("Times are the OPEN time of the candle, in the file's own timestamps.")


if __name__ == "__main__":
    main()
