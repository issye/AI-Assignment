"""
Extended ML backtest: ALS, BPR, NMF, KNN, TF-IDF+Cosine, BM25
Baseline: Raw Cosine CF (HR@5=0.2247), Popularity (HR@5=0.1451)
Temporal split: train < 2011-11-01, test >= 2011-11-01, n≈1,544 customers
"""
import itertools, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load & clean (with parquet cache for fast re-runs) ────────────────────
import pickle
CACHE = "data/online+retail/_cache_cleaned.pkl"
t0 = time.time()
if os.path.exists(CACHE):
    print(f"Loading from cache: {CACHE}")
    df = pickle.load(open(CACHE, "rb"))
    print(f"Loaded {len(df):,} rows ({time.time()-t0:.1f}s)")
else:
    print("Loading data from Excel (slow, ~90s)...")
    df = pd.read_excel("data/online+retail/Online Retail.xlsx")
    df = df[df["CustomerID"].notna()]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df["StockCode"]  = df["StockCode"].astype(str)
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    pickle.dump(df, open(CACHE, "wb"))
    print(f"Cleaned + cached: {len(df):,} rows | {df.CustomerID.nunique():,} customers | "
          f"{df.StockCode.nunique():,} products  ({time.time()-t0:.1f}s)")

# ── Temporal split ─────────────────────────────────────────────────────────
CUTOFF   = pd.Timestamp("2011-11-01")
df_train = df[df["InvoiceDate"] <  CUTOFF]
df_test  = df[df["InvoiceDate"] >= CUTOFF]

eval_custs = list(
    set(df_train["CustomerID"].unique()) & set(df_test["CustomerID"].unique())
)
print(f"Train: {len(df_train):,} rows | Test: {len(df_test):,} rows | "
      f"Eval customers: {len(eval_custs)}")

# ── Build train user-item matrix ───────────────────────────────────────────
print("\nBuilding user-item matrix...")
ui_train = (
    df_train.groupby(["CustomerID", "StockCode"])["Quantity"]
            .sum().unstack(fill_value=0).clip(upper=1)
)
print(f"Shape: {ui_train.shape} | density: {ui_train.values.mean():.4f}")

columns  = ui_train.columns.tolist()          # item list
col_idx  = {sc: i for i, sc in enumerate(columns)}
n_items  = len(columns)
ui_vals  = ui_train.values                    # numpy (n_users, n_items)
row_idx  = {cid: i for i, cid in enumerate(ui_train.index)}

user_items_csr = csr_matrix(ui_vals)          # users x items  (rows=users for implicit 0.7.x)
# implicit 0.7.x: fit() treats rows as users, so pass user_items_csr directly

# ── Pre-build test lookup (CustomerID → set of test StockCodes) ────────────
test_lookup = (
    df_test.groupby("CustomerID")["StockCode"]
           .apply(set).to_dict()
)

# ── Popularity baseline ────────────────────────────────────────────────────
pop_items = (
    df_train.groupby("StockCode")["CustomerID"].nunique()
            .sort_values(ascending=False).index.tolist()
)

# ── Results accumulator ────────────────────────────────────────────────────
KS      = [1, 3, 5, 10]
results = []   # (model, params, hr1, hr3, hr5, hr10)

def record(name, params, hrs, n):
    results.append((name, params, hrs[1], hrs[3], hrs[5], hrs[10]))
    print(f"  {name:<36} {params:<26}  "
          f"HR@1={hrs[1]:.4f}  HR@3={hrs[3]:.4f}  "
          f"HR@5={hrs[5]:.4f}  HR@10={hrs[10]:.4f}  (n={n})")

# ── Similarity-matrix evaluator (uses numpy for speed) ────────────────────
def evaluate_sim(sim_vals, name, params="", already_numpy=False):
    """
    sim_vals: (n_items, n_items) numpy array, indexed by col_idx.
    """
    hits  = {k: 0 for k in KS}
    total = 0
    for cid in eval_custs:
        if cid not in row_idx:
            continue
        u = row_idx[cid]
        bought_mask = ui_vals[u]          # binary vector length n_items
        bought_idxs = np.where(bought_mask == 1)[0]
        if len(bought_idxs) == 0:
            continue
        test_items = test_lookup.get(cid)
        if not test_items:
            continue
        test_idxs = {col_idx[t] for t in test_items if t in col_idx}
        if not test_idxs:
            continue

        scores = sim_vals[:, bought_idxs].mean(axis=1)
        scores[bought_idxs] = -np.inf           # exclude already bought
        top10  = np.argsort(-scores)[:10]

        for k in KS:
            if any(i in test_idxs for i in top10[:k]):
                hits[k] += 1
        total += 1

    hrs = {k: hits[k] / total for k in KS}
    record(name, params, hrs, total)
    return hrs

# ── Implicit-model evaluator ───────────────────────────────────────────────
def evaluate_implicit(model, name, params=""):
    hits  = {k: 0 for k in KS}
    total = 0
    for cid in eval_custs:
        if cid not in row_idx:
            continue
        uid = row_idx[cid]
        test_items = test_lookup.get(cid)
        if not test_items:
            continue
        test_idxs = {col_idx[t] for t in test_items if t in col_idx}
        if not test_idxs:
            continue

        ids, _ = model.recommend(uid, user_items_csr[uid], N=10, filter_already_liked_items=True)

        ids = list(ids)
        for k in KS:
            if any(i in test_idxs for i in ids[:k]):
                hits[k] += 1
        total += 1

    hrs = {k: hits[k] / total for k in KS}
    record(name, params, hrs, total)
    return hrs

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*100)
print("BASELINES")
print("="*100)

# Raw Cosine CF
print("  Computing raw cosine similarity...")
t1 = time.time()
sim_raw = cosine_similarity(ui_train.T.values)
print(f"  Done ({time.time()-t1:.1f}s)")
evaluate_sim(sim_raw, "Raw Cosine CF", "baseline")

# Popularity
hits_p = {k: 0 for k in KS}; total_p = 0
for cid in eval_custs:
    if cid not in row_idx: continue
    bought_set = set(columns[i] for i in np.where(ui_vals[row_idx[cid]] == 1)[0])
    test_items = test_lookup.get(cid)
    if not bought_set or not test_items: continue
    top10p = [p for p in pop_items if p not in bought_set][:10]
    top10s = set(top10p)
    for k in KS:
        if any(t in set(top10p[:k]) for t in test_items): hits_p[k] += 1
    total_p += 1
hrs_p = {k: hits_p[k]/total_p for k in KS}
record("Popularity", "baseline", hrs_p, total_p)

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*100)
print("TIER 2 - SKLEARN MODELS")
print("="*100)

# TF-IDF + Cosine (no SVD)
print("\n[TF-IDF + Cosine (no SVD)]")
tfidf = TfidfTransformer()
ui_tfidf = tfidf.fit_transform(ui_vals)
sim_tfidf_cos = cosine_similarity(ui_tfidf.T)
evaluate_sim(sim_tfidf_cos, "TF-IDF+Cosine (no SVD)", "")

# NMF
print("\n[NMF - Non-Negative Matrix Factorisation]")
for n_comp in [20, 50, 100]:
    t1 = time.time()
    nmf = NMF(n_components=n_comp, max_iter=300, random_state=42)
    nmf.fit(ui_vals)
    item_factors = nmf.components_.T   # (n_items, n_comp)
    sim_nmf = cosine_similarity(item_factors)
    print(f"  NMF fit n_comp={n_comp} ({time.time()-t1:.1f}s)")
    evaluate_sim(sim_nmf, "NMF", f"n_comp={n_comp}")

# KNN Item-Based CF
print("\n[KNN Item-Based CF]")
for n_neighbors in [20, 50, 100]:
    t1 = time.time()
    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    knn.fit(ui_train.T.values)
    distances, indices = knn.kneighbors(ui_train.T.values)
    # Build symmetrised similarity matrix (dense but 107 MB is fine)
    sim_knn = np.zeros((n_items, n_items), dtype=np.float32)
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        for d, j in zip(dists, idxs):
            s = 1.0 - d
            if s > sim_knn[i, j]: sim_knn[i, j] = s
            if s > sim_knn[j, i]: sim_knn[j, i] = s
    print(f"  KNN fit n_neighbors={n_neighbors} ({time.time()-t1:.1f}s)")
    evaluate_sim(sim_knn, "KNN Item-Based CF", f"n_neighbors={n_neighbors}")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*100)
print("TIER 1 - IMPLICIT MODELS (ALS, BPR, BM25)")
print("="*100)

# ALS grid: 27 combinations
print("\n[ALS - Alternating Least Squares]")
from implicit.als import AlternatingLeastSquares

best_als = {"hr5": 0.0, "config": None}
for factors, iterations, reg in itertools.product([20, 50, 100], [10, 20, 50], [0.01, 0.1, 1.0]):
    t1 = time.time()
    model = AlternatingLeastSquares(
        factors=factors, iterations=iterations, regularization=reg,
        use_gpu=False, random_state=42
    )
    model.fit(user_items_csr, show_progress=False)
    params = f"f={factors},i={iterations},r={reg}"
    print(f"  ALS fit {params} ({time.time()-t1:.1f}s)")
    hrs = evaluate_implicit(model, "ALS", params)
    if hrs[5] > best_als["hr5"]:
        best_als["hr5"] = hrs[5]
        best_als["config"] = (factors, iterations, reg)

print(f"\n  ** Best ALS: f={best_als['config'][0]}, i={best_als['config'][1]}, "
      f"r={best_als['config'][2]}, HR@5={best_als['hr5']:.4f}")

# BPR grid: 27 combinations
print("\n[BPR - Bayesian Personalised Ranking]")
from implicit.bpr import BayesianPersonalizedRanking

best_bpr = {"hr5": 0.0, "config": None}
for factors, iterations, reg in itertools.product([20, 50, 100], [10, 20, 50], [0.01, 0.1, 1.0]):
    t1 = time.time()
    model = BayesianPersonalizedRanking(
        factors=factors, iterations=iterations, regularization=reg,
        random_state=42
    )
    model.fit(user_items_csr, show_progress=False)
    params = f"f={factors},i={iterations},r={reg}"
    print(f"  BPR fit {params} ({time.time()-t1:.1f}s)")
    hrs = evaluate_implicit(model, "BPR", params)
    if hrs[5] > best_bpr["hr5"]:
        best_bpr["hr5"] = hrs[5]
        best_bpr["config"] = (factors, iterations, reg)

print(f"\n  ** Best BPR: f={best_bpr['config'][0]}, i={best_bpr['config'][1]}, "
      f"r={best_bpr['config'][2]}, HR@5={best_bpr['hr5']:.4f}")

# BM25
print("\n[BM25]")
from implicit.nearest_neighbours import BM25Recommender

for k_val in [100, 200]:
    t1 = time.time()
    try:
        model = BM25Recommender(K=k_val)
        # BM25 expects item_users (items x users) format in implicit 0.7.x
        model.fit(user_items_csr.T.tocsr(), show_progress=False)
        params = f"K={k_val}"
        print(f"  BM25 fit {params} ({time.time()-t1:.1f}s)")
        # BM25 needs user vector in float64 csr format
        hits_bm25 = {k: 0 for k in KS}
        total_bm25 = 0
        for cid in eval_custs:
            if cid not in row_idx: continue
            uid = row_idx[cid]
            test_items = test_lookup.get(cid)
            if not test_items: continue
            test_idxs = {col_idx[t] for t in test_items if t in col_idx}
            if not test_idxs: continue
            try:
                user_vec = user_items_csr[uid].astype(np.float64)
                bm25_ids, _ = model.recommend(uid, user_vec, N=10, filter_already_liked_items=True)
                bm25_ids = list(bm25_ids)
                for k in KS:
                    if any(i in test_idxs for i in bm25_ids[:k]):
                        hits_bm25[k] += 1
                total_bm25 += 1
            except Exception:
                continue
        if total_bm25 > 0:
            hrs_bm25 = {k: hits_bm25[k] / total_bm25 for k in KS}
            record("BM25", params, hrs_bm25, total_bm25)
        else:
            print(f"  BM25 K={k_val}: could not get recommendations")
    except Exception as e:
        print(f"  BM25 K={k_val}: skipped ({e})")

# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*100)
print("FULL RESULTS TABLE")
print("="*100)
print(f"{'Model':<36} {'Params':<26}  {'HR@1':>7}  {'HR@3':>7}  {'HR@5':>7}  {'HR@10':>7}")
print("-" * 100)
for row in results:
    name, params, hr1, hr3, hr5, hr10 = row
    flag = " **WINNER**" if hr5 >= 0.2247 else ""
    print(f"  {name:<34} {params:<26}  {hr1:>7.4f}  {hr3:>7.4f}  {hr5:>7.4f}  {hr10:>7.4f}{flag}")
print("-" * 100)

# Winner
best_row = max(results, key=lambda r: r[4])
print(f"\n  WINNER: {best_row[0]} ({best_row[1]})")
print(f"  HR@1={best_row[2]:.4f}  HR@3={best_row[3]:.4f}  HR@5={best_row[4]:.4f}  HR@10={best_row[5]:.4f}")
baseline_hr5 = next(r[4] for r in results if r[0] == "Raw Cosine CF")
print(f"  vs Raw Cosine CF HR@5={baseline_hr5:.4f} : {best_row[4]-baseline_hr5:+.4f}")

print(f"\nTotal runtime: {time.time()-t0:.0f}s")
