import pandas as pd

df = pd.read_csv("real_gold_data_mt5_90000.csv")

for n in [500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 90000]:
    slice_df = df.head(n)
    filename = f"real_gold_data_mt5_{n}.csv"
    slice_df.to_csv(filename, index=False)
    print(f"Wrote {len(slice_df)} rows to {filename}")