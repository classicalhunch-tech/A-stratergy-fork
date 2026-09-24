"""
make_staged_csv.py

Generates chronological candle-count slices of your_data_file.csv,
matching the same "first N rows" approach used for the existing
500/1000/2000 staged files.

Usage:
    python make_staged_csv.py 3000
    python make_staged_csv.py 4000
"""

import sys
import pandas as pd

def main():
    if len(sys.argv) != 2:
        print("Usage: python make_staged_csv.py <n_candles>")
        sys.exit(1)

    n = int(sys.argv[1])

    df = pd.read_csv("your_data_file.csv")
    sliced = df.iloc[:n]

    out_path = f"your_data_file_{n}.csv"
    sliced.to_csv(out_path, index=False)

    print(f"Wrote {len(sliced)} rows to {out_path}")

if __name__ == "__main__":
    main()