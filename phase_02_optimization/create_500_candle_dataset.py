import pandas as pd

df = pd.read_csv("your_data_file.csv")

df_500 = df.head(500)

df_500.to_csv("your_data_file_500.csv", index=False)

print(f"Wrote {len(df_500)} rows to your_data_file_500.csv")