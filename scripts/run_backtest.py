"""
Standalone backtest script — runs SVD CF vs Raw Cosine CF vs Popularity.
Temporal split: train < 2011-11-01, test >= 2011-11-01.
Produces Hit Rate@K for K in {1,3,5,10}.
"""
import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# ── Load & clean ──────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel('data/online+retail/Online Retail.xlsx')
df = df[df['CustomerID'].notna()]
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df['StockCode']  = df['StockCode'].astype(str)
df['CustomerID'] = df['CustomerID'].astype(int).astype(str)
print(f"Cleaned: {len(df):,} rows | {df.CustomerID.nunique():,} customers | {df.StockCode.nunique():,} products")

# ── Temporal split ────────────────────────────────────────────────────────
CUTOFF   = pd.Timestamp('2011-11-01')
df_train = df[df['InvoiceDate'] <  CUTOFF]
df_test  = df[df['InvoiceDate'] >= CUTOFF]

train_custs = set(df_train['CustomerID'].unique())
test_custs  = set(df_test['CustomerID'].unique())
eval_custs  = list(train_custs & test_custs)

print(f"\nTrain: {df_train.InvoiceDate.min().date()} to {df_train.InvoiceDate.max().date()} ({len(df_train):,} rows)")
print(f"Test : {df_test.InvoiceDate.min().date()}  to {df_test.InvoiceDate.max().date()} ({len(df_test):,} rows)")
print(f"Customers in both splits: {len(eval_custs)}")

# ── Build train user-item matrix ──────────────────────────────────────────
print("\nBuilding user-item matrix...")
ui_train = (
    df_train.groupby(['CustomerID','StockCode'])['Quantity']
            .sum().unstack(fill_value=0).clip(upper=1)
)
print(f"User-item matrix: {ui_train.shape} | density: {ui_train.values.mean():.4f}")

# ── TF-IDF weighting ─────────────────────────────────────────────────────
# Downweights popular items bought by many users (low discriminative value)
# Amplifies rare co-purchases (high signal)
print("\nApplying TF-IDF weighting to user-item matrix...")
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
ui_tfidf = tfidf.fit_transform(ui_train)
print(f"TF-IDF matrix shape: {ui_tfidf.shape}")

# ── Train SVD on TF-IDF weighted matrix ──────────────────────────────────
print("\nTraining SVD on TF-IDF weighted matrix...")
svd_probe = TruncatedSVD(n_components=100, random_state=42)
svd_probe.fit(ui_tfidf)
cumvar       = np.cumsum(svd_probe.explained_variance_ratio_)
n_components = max(20, min(int(np.searchsorted(cumvar, 0.50)) + 1, 100))
print(f"Variance @ 100 components : {cumvar[-1]:.4f}")
print(f"Components for 50% var    : {n_components}")

svd_model    = TruncatedSVD(n_components=n_components, random_state=42)
svd_model.fit(ui_tfidf)
item_factors = svd_model.components_.T
print(f"Variance explained (final): {svd_model.explained_variance_ratio_.sum():.4f}")

# ── Similarity matrices ───────────────────────────────────────────────────
print("\nComputing similarity matrices...")
sim_svd_tfidf = pd.DataFrame(
    cosine_similarity(item_factors),
    index=ui_train.columns, columns=ui_train.columns
)
# Raw SVD (no TF-IDF) — kept for comparison
svd_raw_     = TruncatedSVD(n_components=n_components, random_state=42)
svd_raw_.fit(ui_train)
sim_svd = pd.DataFrame(
    cosine_similarity(svd_raw_.components_.T),
    index=ui_train.columns, columns=ui_train.columns
)
sim_raw = pd.DataFrame(
    cosine_similarity(ui_train.T.values),
    index=ui_train.columns, columns=ui_train.columns
)

# ── Popularity baseline ───────────────────────────────────────────────────
pop_items = (
    df_train.groupby('StockCode')['CustomerID'].nunique()
            .sort_values(ascending=False).index.tolist()
)

# ── Backtest ──────────────────────────────────────────────────────────────
print(f"\nRunning backtest on {len(eval_custs)} customers...")

def top_k_cf(sim, bought, k):
    valid = [sc for sc in bought if sc in sim.columns]
    if not valid: return []
    scores = sim.loc[:, valid].mean(axis=1)
    scores = scores.drop(index=[i for i in bought if i in scores.index], errors='ignore')
    return scores.nlargest(k).index.tolist()

def top_k_pop(bought, k):
    return [p for p in pop_items if p not in bought][:k]

KS = [1, 3, 5, 10]
hits = {m: {k: 0 for k in KS} for m in ['tfidf_svd', 'svd', 'raw', 'pop']}
total = 0

for cid in eval_custs:
    train_items = (ui_train.loc[cid][ui_train.loc[cid]==1].index.tolist()
                   if cid in ui_train.index else [])
    test_items  = df_test[df_test['CustomerID']==cid]['StockCode'].unique().tolist()
    if not train_items or not test_items:
        continue
    for k in KS:
        hit = lambda recs, truth: int(any(t in recs for t in truth))
        hits['tfidf_svd'][k] += hit(top_k_cf(sim_svd_tfidf, train_items, k), test_items)
        hits['svd'][k]       += hit(top_k_cf(sim_svd,       train_items, k), test_items)
        hits['raw'][k]       += hit(top_k_cf(sim_raw,       train_items, k), test_items)
        hits['pop'][k]       += hit(top_k_pop(train_items, k),               test_items)
    total += 1

# ── Results ───────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"BACKTEST RESULTS  (n={total} customers, cutoff={CUTOFF.date()})")
print(f"{'='*62}")
print(f"{'Model':<24} {'HR@1':>8} {'HR@3':>8} {'HR@5':>8} {'HR@10':>8}")
print(f"{'-'*62}")
labels = {
    'tfidf_svd': 'TF-IDF + SVD (ours)',
    'svd':       'SVD only',
    'raw':       'Raw Cosine CF',
    'pop':       'Popularity',
}
for m in ['tfidf_svd', 'svd', 'raw', 'pop']:
    vals = [f"{hits[m][k]/total:.4f}" for k in KS]
    print(f"{labels[m]:<24} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8}")
print(f"{'-'*62}")
for k in KS:
    our  = hits['tfidf_svd'][k] / total
    raw  = hits['raw'][k]       / total
    pop  = hits['pop'][k]       / total
    print(f"TF-IDF+SVD vs Raw @{k:<3}: {our-raw:+.4f}  |  vs Pop @{k}: {our-pop:+.4f}")
