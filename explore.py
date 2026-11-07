import pandas as pd

df = pd.read_csv("data/PS_20174392719_1491204439457_log.csv", nrows=100000)

print(df.shape)
print(df.head())
print(df["isFraud"].value_counts())
print(df["type"].value_counts())