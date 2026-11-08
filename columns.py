import pandas as pd
df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv", nrows=1)
print(df.columns.tolist())