import pandas as pd

df = pd.read_csv("data/career_dataset_large.xlsx")  # update filename

print(df.shape)
print(df.columns.tolist())
print("\n--- Last column (likely career label) ---")
print(df.iloc[:, -1].value_counts())
print("Unique:", df.iloc[:, -1].nunique())
print("\n--- Nulls ---")
print(df.isnull().sum())