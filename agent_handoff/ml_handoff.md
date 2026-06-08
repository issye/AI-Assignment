# Agent Handoff — ML Module (Member 3)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/ml`
**Dataset:** `data/online+retail/Online Retail.xlsx`
**Deadline:** 27 June 2026, 7 PM

---

## Model Architecture (Session 3 Update)

Two models now power `predict_product()`. ALS is still used by `recommend_products()`.

| Component | Model | Features | Trained on | Works for |
|---|---|---|---|---|
| `predict_product()` — Model A | RF (context) | segment, price_range, month (3) | All customers with test labels (~1,844) | ALL users incl. cold-start |
| `predict_product()` — Model B | RF (history) | n_purchases, n_categories, recency, avg_value, fav_cat one-hot (15) | Returning customers only (~1,544) | Returning users |
| `recommend_products()` | ALS item similarity | Binary user-item matrix | Full dataset | Returning users (popularity fallback for cold-start) |

**Confidence blend formula:**
```
confidence = 1 - 1 / (1 + n_purchases)

n=0  → confidence=0.00 → 100% context model (cold-start)
n=5  → confidence=0.83 → 17% context + 83% history
n=10 → confidence=0.91 → 9% context + 91% history
```

Scores from `predict_product()` are **0–1 RF probabilities** (not cosine similarities).
`predict_product()` no longer raises `ValueError` for empty `purchase_history`.

---

## Notebook Structure (10 sections)

### Sections 1–5: Data pipeline (unchanged)
Load → EDA → Clean → User-item matrix → Feature engineering (categories, customer features, product catalogue)

### Section 6: Model Training (4 sub-sections)

**6a. ALS** — same as before. Produces `item_sim_df` (3,665×3,665) and `category_sim_df` (8×8).

**6b. Supervised training data construction**
- Temporal split: `CUTOFF = 2011-11-01`
- Recomputes customer stats from `df_train_sup` only (no leakage from test period)
- Labels: for each customer with test-period purchases, 8 binary columns (did they buy from each category?)
- Builds rows for both returning (~1,544) and cold-start (~300-400) customers
- Combines into `df_train_ml` (one row per customer)
- Encoding: `segment_map = {'New':0,'Occasional':1,'Frequent':2}`, `price_map = {'Low':0,...}`
- `all_fav_cats = PRODUCT_CATEGORIES + ['Unknown']` (9 one-hot columns for favourite_category)

**6c. Model A — Context model**
```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

model_context = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
)
model_context.fit(X_ctx, Y)   # X_ctx: (n, 3), Y: (n, 8) binary
# len(model_context.estimators_) == 8 — one RF per category
```

**6d. Model B — History model**
```python
model_history = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42)
)
model_history.fit(X_hist, Y_hist)   # X_hist: (n_returning, 15), Y_hist: (n_returning, 8)
```

### Section 7: Evaluation (2 parts)

**7a. Category HR@3** — primary evaluation for `predict_product()`
- For ~1,544 returning customers: did top-3 predicted categories include any actually-bought category?
- Compares: RF blend vs old CF-based scoring vs popularity
- Also demonstrates cold-start working (no ValueError for `new_customer` profile)

**7b. Product HR@K** — secondary evaluation for `recommend_products()`
- Same temporal split, compares ALS vs Raw Cosine vs Popularity at item level

### Sections 8–10: Save → Inference functions → Demo

---

## Saved Artefacts (8 files)

| File | Contents | Role |
|---|---|---|
| `models/als_model.pkl` | Trained ALS model | Used internally for item factors |
| `models/similarity_matrix.pkl` | 3,665×3,665 item cosine similarity | `recommend_products()` |
| `models/category_similarity.pkl` | 8×8 category similarity DataFrame | Member 1 BFS search |
| `models/model_context.pkl` | MultiOutputClassifier(RF), 3 features | `predict_product()` Model A |
| `models/model_history.pkl` | MultiOutputClassifier(RF), 15 features | `predict_product()` Model B |
| `models/product_catalogue.pkl` | StockCode, Description, category, popularity_rank, avg_price | Both functions |
| `models/customer_features.pkl` | Per-customer aggregated features | Reference |
| `models/encoder_category.pkl` | LabelEncoder for PRODUCT_CATEGORIES | Reference |

All pkl files are gitignored — regenerate by running `Kernel → Restart & Run All`.

---

## Public Interface (signatures unchanged)

```python
# Works for ALL users including cold-start (no longer raises ValueError)
predict_product(user_profile: dict, candidates: list = None) -> list[tuple[str, float]]
# Returns: [("Home Decor", 0.312), ("Kitchen & Dining", 0.287), ...] — probabilities 0-1

# Personalized product scoring within a category (ALS). Falls back to popularity for cold-start.
recommend_products(user_profile: dict, category: str, top_n: int = 3) -> list[dict]
# Returns: [{"category": str, "product": str, "score": float}, ...]
```

Key change from previous session: `predict_product()` scores are now **0–1 probabilities**
from the RF models, not cosine similarities (previously ~0.02–0.15). Member 4's `recommend()`
just sorts by score, so it's unaffected.

---

## Rules

- `predict_product()` and `recommend_products()` signatures are fixed — Member 4 calls them
- `recommend_products()` returns a flat list of dicts with keys: category, product, score
- Never return items that appear in `purchase_history`
- Do not push — user pushes manually
- To rebuild notebook: `python scripts/build_notebook.py` then execute
