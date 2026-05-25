import pandas as pd
df = pd.read_csv("career_recommender.csv")

job_col = [c for c in df.columns if 'Job' in c][0]
print(df[job_col].value_counts().head(30))
print("Unique values:", df[job_col].nunique())
print("NAs:", df[job_col].isna().sum())