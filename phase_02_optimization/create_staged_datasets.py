import pandas as pd

df = pd.read_csv("your_data_file.csv")

for n in [1000, 2000]:
    subset = df.head(n)
    filename = f"your_data_file_{n}.csv"
    subset.to_csv(filename, index=False)
    print(f"Wrote {len(subset)} rows to {filename}")