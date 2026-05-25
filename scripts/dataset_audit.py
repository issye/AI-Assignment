"""
dataset_audit.py
Full audit script for career_dataset_large.xlsx
Run from the Assignment root folder:
    python scripts/dataset_audit.py
"""

import pandas as pd
import numpy as np

FILE = "data/career_dataset_large.xlsx"
LABEL_COL = "Recommended Career"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_excel(FILE)

print("=" * 60)
print("1. BASIC INFO")
print("=" * 60)
print(f"Rows       : {df.shape[0]}")
print(f"Columns    : {df.shape[1]}")
print(f"Col names  : {df.columns.tolist()}")
print(f"Dtypes:\n{df.dtypes}\n")

# ── Nulls ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("2. NULL / MISSING VALUES")
print("=" * 60)
null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).round(2)
null_df = pd.DataFrame({"nulls": null_counts, "% missing": null_pct})
print(null_df, "\n")

# ── Career label distribution ─────────────────────────────────────────────────
print("=" * 60)
print(f"3. CAREER LABEL DISTRIBUTION  [{LABEL_COL}]")
print("=" * 60)
vc = df[LABEL_COL].value_counts()
print(vc)
print(f"\nUnique careers : {vc.nunique()}")
print(f"Min class size : {vc.min()}  ({vc.idxmin()})")
print(f"Max class size : {vc.max()}  ({vc.idxmax()})")
print(f"Imbalance ratio: {vc.max() / vc.min():.2f}x  (1.0 = perfect balance)\n")

# ── Feature columns ───────────────────────────────────────────────────────────
print("=" * 60)
print("4. FEATURE COLUMN DETAILS")
print("=" * 60)

for col in df.columns:
    if col == LABEL_COL:
        continue
    print(f"\n--- {col} ---")
    print(f"  dtype   : {df[col].dtype}")
    print(f"  unique  : {df[col].nunique()}")
    if df[col].dtype == object:
        print(f"  samples : {df[col].dropna().head(3).tolist()}")
        # Check if it looks like a multi-label (comma-separated) column
        sample = df[col].dropna().iloc[0]
        if isinstance(sample, str) and "," in sample:
            all_vals = df[col].dropna().str.split(",").explode().str.strip()
            print(f"  [MULTI-LABEL] {all_vals.nunique()} unique tokens")
            print(f"  Top tokens: {all_vals.value_counts().head(5).to_dict()}")
    else:
        print(f"  min/max : {df[col].min()} / {df[col].max()}")
        print(f"  mean    : {df[col].mean():.2f}")

# ── Duplicate rows ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. DUPLICATE ROWS")
print("=" * 60)
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes} ({dupes/len(df)*100:.1f}%)\n")

# ── ML readiness summary ──────────────────────────────────────────────────────
print("=" * 60)
print("6. ML READINESS SUMMARY")
print("=" * 60)

checks = {
    "Row count >= 500"          : df.shape[0] >= 500,
    "Career labels 8-20"        : 8 <= df[LABEL_COL].nunique() <= 20,
    "Imbalance ratio < 3x"      : vc.max() / vc.min() < 3,
    "Min class size >= 30"      : vc.min() >= 30,
    "No critical nulls (>50%)"  : (null_pct < 50).all(),
    "Label column has no nulls" : df[LABEL_COL].isnull().sum() == 0,
}

for check, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}]  {check}")

print("\nDone.")
