# Session Handoff
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/ml`
**Date:** 2026-06-06
**Prepared by:** Member 3 (ML Engineer) via Claude Code session

---

## What This Session Accomplished

### 1. Dataset search (7 datasets evaluated, all rejected except suvroo)

| Dataset | Verdict | Fatal flaw |
|---|---|---|
| `ecommerce_customer_behavior_dataset_v2.csv` | FAIL | Synthetic — all MI ≈ 0, p=0.60 |
| `flo_customer_data.csv` | FAIL | Fashion-only, no demographics, pre-aggregated |
| `shopping_trends.csv` | FAIL | Synthetic fashion-only, US cities, 1 row/customer |
| `Dataset.csv` (UCI Retail II augmented) | FAIL | Demographics per-customer inconsistent (96% have multiple genders) |
| `content_based_recommendation_dataset.csv` | FAIL | Wrong problem type (regression), no category column |
| `e-commerce/` (3-file set) | FAIL | customer_details = shopping_trends renamed; product IDs hash mismatch |
| **`data/suvroo/` (2 files)** | **PASS** | Viable — see below |

### 2. Architecture pivot decided

**Old:** 8-class demographic category classifier on synthetic dataset → ~20-55% accuracy ceiling
**New:** Binary purchase-probability model on user-product pairs → specific named product recommendations

**New pipeline output:**
```
User Profile → Rules Engine → A* Search → ML (predict_product) → recommend_products → ["Smartphone", "Yoga Mat", ...]
```

### 3. Three key decisions made (do not revisit)
1. `predict_product()` keeps its category-ranking signature (backward compatible with A*)
2. `recommend_products()` is the new product-level function added alongside it
3. A* uses **co-purchase frequency** heuristic (not price-gap) — because suvroo category prices are nearly flat (~₹2,500 across all 6 categories)

### 4. Files committed to `feature/ml`

| Commit | File | What changed |
|---|---|---|
| `3fda300` | `src/constants.py` | Full update for suvroo (see below) |
| `691f3db` | `agent_handoff/ml_handoff.md` | Updated ML brief |
| `691f3db` | `agent_handoff/search_handoff.md` | New — A* co-purchase heuristic brief |
| `691f3db` | `agent_handoff/rules_handoff.md` | New — Rules Engine updated brief |
| `691f3db` | `agent_handoff/integration_handoff.md` | New — Integration updated brief |
| `bead91c` | `notebooks/ml_model.ipynb` | Full 52-cell rewrite (not yet run) |

---

## Current State of Every Module

### `src/constants.py` — ✅ Updated, committed

```python
PRODUCT_CATEGORIES = ["Books", "Beauty", "Electronics", "Fashion", "Sports", "Home & Garden"]

CATEGORY_AVG_PRICES = {  # from suvroo product data — nearly flat, reference only
    "Books": 2524, "Beauty": 2501, "Electronics": 2548,
    "Fashion": 2618, "Sports": 2578, "Home & Garden": 2549,
}

SPEND_THRESHOLDS = {  # from suvroo Avg_Order_Value quartiles
    "Low":      (0,     1636),
    "Mid-Low":  (1636,  2740),
    "Mid-High": (2740,  3879),
    "High":     (3879,  float("inf")),
}

CITIES      = ["Bangalore", "Chennai", "Delhi", "Kolkata", "Mumbai"]
URBAN_CITIES = ["Mumbai", "Delhi", "Bangalore"]
CUSTOMER_SEGMENTS = ["New Visitor", "Occasional Shopper", "Frequent Buyer"]

SAMPLE_PROFILES = {
    "budget_browser":    build_user_profile(22, "Female", "Chennai",    800.0, ...),
    "beauty_enthusiast": build_user_profile(30, "Female", "Mumbai",    1800.0, ...),
    "fashion_fan":       build_user_profile(27, "Male",   "Delhi",     2500.0, ...),
    "fitness_guy":       build_user_profile(35, "Male",   "Bangalore", 3200.0, ...),
    "tech_spender":      build_user_profile(42, "Male",   "Mumbai",    4500.0, ...),
}
```

`build_user_profile()` signature is **unchanged** but gains an optional `customer_segment` param.

---

### `notebooks/ml_model.ipynb` — ✅ Written (52 cells), ❌ NOT YET RUN

**Immediate next step: `Kernel → Restart & Run All`**

The notebook will:
1. Load `data/suvroo/customer_data_collection.csv` + `data/suvroo/product_recommendation_data.csv`
2. Build ~80K user-product training pairs from Purchase_History (positive) + random sampling (negative)
3. Train 4 models (DT, KNN, RF, GBM) with GridSearchCV, select best by ROC-AUC
4. Wrap winner in `CalibratedClassifierCV(method='isotonic')`
5. Save 9 pkl artefacts to `models/`
6. Define and test `predict_product()` and `recommend_products()`

**Cell 12 (co-purchase matrix)** prints the `CO_PURCHASE_COSTS` dict that Member 1 needs to hardcode into `src/search_module.py`. Share this output with Member 1 after running.

After running, update Section 9 (Results Discussion) with actual ROC-AUC numbers.

---

### `src/rules_engine.py` — ❌ Not yet implemented (Member 2)
Brief at `agent_handoff/rules_handoff.md`. Key changes from old brief:
- 6 categories (not 8), ₹ thresholds, Indian cities
- New Rule 10: `Customer_Segment == "Frequent Buyer"` → broaden eligible set
- Signature unchanged: `apply_rules(user_profile) -> list[str]`

---

### `src/search_module.py` — ❌ Needs conversion (Member 1)
Brief at `agent_handoff/search_handoff.md`. Key changes from old brief:
- 6-node graph (not 8)
- **New heuristic: co-purchase frequency** (not price-gap). Edge cost = `1 - (co_purchase_count / 10000)`
- New start node: use `user_profile.get('browsing_history', [])[0]` if available
- CO_PURCHASE_COSTS matrix values come from notebook Cell 12 output (run notebook first)
- Signature unchanged: `find_product_path(user_profile, target_category) -> list[str]`

---

### `notebooks/career_recommender.ipynb` — ❌ Not yet implemented (Member 4)
Brief at `agent_handoff/integration_handoff.md`. Key changes from old brief:
- `recommend()` gains a 4th step calling `recommend_products()` after `predict_product()`
- Output dict now includes `recommended_products` key
- Import `recommend_products` from ml_model.ipynb (or copy function definition)
- Updated SAMPLE_PROFILES (Indian cities, ₹ spend values)

---

## The Suvroo Dataset

**Location:** `data/suvroo/`

### `customer_data_collection.csv` (10,000 rows × 11 cols)
One row per customer. Key columns:
- `Customer_ID` — C1000–C9999
- `Age` — 18–60
- `Gender` — Female / Male / Other (balanced ~33% each)
- `Location` — 5 Indian cities
- `Browsing_History` — string list e.g. `['Electronics', 'Fitness']`
- `Purchase_History` — string list e.g. `['Smartphone', 'Yoga Mat']` (24 unique items, 4 per category)
- `Customer_Segment` — New Visitor / Occasional Shopper / Frequent Buyer
- `Avg_Order_Value` — ₹500–₹5,000 (PRIMARY spend signal, replaces median_spend)
- `Holiday`, `Season` — context features

### `product_recommendation_data.csv` (10,000 rows × 14 cols after cleanup)
One row per product. Key columns:
- `Product_ID` — P2000–P11999 (no join key to customer file)
- `Category` — 6 categories (Fitness→Sports, Home Decor→Home & Garden via CAT_MAP)
- `Subcategory` — 24 subcategories (4 per category)
- `Price` — ₹100–₹5,000 (nearly flat across categories, avg ~₹2,550)
- `Brand` — Brand A/B/C/D
- `Product_Rating`, `Average_Rating_of_Similar_Products` — 2.0–5.0
- `Customer_Review_Sentiment_Score` — 0.0–1.0
- `Probability_of_Recommendation` — 0.1–1.0 (use as a product quality feature)
- `Similar_Product_List` — string list of similar subcategory names

**Category mapping used in notebook and inference:**
```python
CAT_MAP = {'Fitness': 'Sports', 'Home Decor': 'Home & Garden'}

ITEM_TO_CAT = {
    'Biography': 'Books',         'Non-fiction': 'Books',
    'Fiction': 'Books',           'Comics': 'Books',
    'Moisturizer': 'Beauty',      'Lipstick': 'Beauty',
    'Foundation': 'Beauty',       'Perfume': 'Beauty',
    'Smartphone': 'Electronics',  'Headphones': 'Electronics',
    'Laptop': 'Electronics',      'Smartwatch': 'Electronics',
    'T-shirt': 'Fashion',         'Jeans': 'Fashion',
    'Jacket': 'Fashion',          'Shoes': 'Fashion',
    'Resistance Bands': 'Sports', 'Dumbbells': 'Sports',
    'Yoga Mat': 'Sports',         'Treadmill': 'Sports',
    'Wall Art': 'Home & Garden',  'Curtains': 'Home & Garden',
    'Cushions': 'Home & Garden',  'Lamp': 'Home & Garden',
}
```

---

## ML Notebook Architecture (52 cells)

| Section | Cells | Key content |
|---|---|---|
| 1 | 2 | Imports, constants, CAT_MAP, ITEM_TO_CAT, parse_list() |
| 2 | 3 | Load both CSVs, drop unnamed cols, map categories |
| 3 | 8 | EDA: demographics, spend tiers, browse/purchase patterns, co-purchase heatmap |
| 4 | 6 | Feature engineering: customer features, product features, pair construction, cross features |
| 5 | 3 | Preprocessing: fit encoders, assemble X (30 features) and y (binary) |
| 6 | 2 | 80/20 stratified train/test split |
| 7 | 8 | DT, KNN, RF, GBM GridSearchCV + model selection + calibration |
| 8 | 8 | ROC-AUC, ROC curves, calibration diagram, classification report, feature importance, CV, P@K, per-category AUC |
| 9 | 1 | Results discussion (pre-written, fill in actual numbers after run) |
| 10 | 2 | Save 9 pkl artefacts to models/ |
| 11 | 3 | predict_product() definition + tests on SAMPLE_PROFILES |
| 12 | 3 | recommend_products() definition + tests on SAMPLE_PROFILES |
| 13 | 3 | Full integration demo + re-ranking mode test + confidence bar chart |

### Feature groups (30 total)
```
USER_FEATURES (19): Age, gender_enc, city_enc, is_urban, Avg_Order_Value,
  log_avg_order, price_range_enc, segment_enc, season_enc, holiday_enc,
  n_browse_cats, n_purchase_items, is_multi_buyer,
  browsed_books, browsed_beauty, browsed_electronics,
  browsed_fashion, browsed_sports, browsed_home_and_garden

PRODUCT_FEATURES (9): category_enc, subcat_enc, Price, log_price,
  Product_Rating, Average_Rating_of_Similar_Products,
  Customer_Review_Sentiment_Score, Probability_of_Recommendation, brand_enc

CROSS_FEATURES (2): price_fit, category_browsed
```

### Saved artefacts (created on notebook run)
```
models/product_recommendation_model.pkl   ← CalibratedClassifierCV(best_model)
models/encoder_gender.pkl
models/encoder_city.pkl
models/encoder_segment.pkl
models/encoder_season.pkl
models/encoder_category.pkl
models/encoder_subcat.pkl
models/encoder_brand.pkl
models/product_catalogue.pkl              ← full product DataFrame for inference
```

---

## Public Interface Contracts (DO NOT CHANGE)

```python
# Member 2
apply_rules(user_profile: dict) -> list[str]

# Member 1
find_product_path(user_profile: dict, target_category: str) -> list[str]

# Member 3 — EXISTING (backward compatible)
predict_product(user_profile: dict, candidates: list = None) -> list[tuple[str, float]]
# Returns: [("Electronics", 0.84), ("Sports", 0.61), ...]

# Member 3 — NEW
recommend_products(user_profile: dict, category: str, top_n: int = 3) -> list[tuple[str, float]]
# Returns: [("Smartphone", 0.91), ("Laptop", 0.87), ("Headphones", 0.73)]

# Member 4
recommend(user_profile: dict) -> dict
# Returns: {"top_3_categories": [...], "recommended_products": [...],
#           "roadmap": [...], "eligible": [...]}
```

---

## Immediate Next Steps (in order)

### Step 1 — Push constants.py to main (BLOCKING for all members)
```
git push origin feature/ml
# Then merge src/constants.py to main via PR or direct merge
# All members must pull this before building anything
```

### Step 2 — Run the ML notebook
```
Kernel → Restart & Run All
Expected: zero errors, 9 pkl files in models/, ROC-AUC printed in Section 8
```

### Step 3 — Copy CO_PURCHASE_COSTS to Member 1
After running, Cell 12 prints the full `CO_PURCHASE_COSTS` dict.
Send this output to Member 1 — they hardcode it into `src/search_module.py`.

### Step 4 — Update Results Discussion (Section 9)
Fill in actual ROC-AUC, P@3, R@3 values from the run. Remove placeholder text.

### Step 5 — Commit trained models + final notebook
```
git add models/ notebooks/ml_model.ipynb
git commit -m "feat(ml): add trained models and evaluation results"
```

### Step 6 — Draft PR
```
gh pr create --draft --title "feat(ml): suvroo pivot — binary purchase-probability model"
```

---

## Known Issues / Watch Out For

1. **Notebook runtime** — the negative sampling loop (Section 4.4) iterates 10,000 customers.
   Expect ~2–5 minutes on first run. Normal.

2. **`display()` requires Jupyter** — the notebook uses `display()` for DataFrames.
   Run in Jupyter, not plain Python.

3. **`models/` directory** — old pkl files from the previous architecture may exist.
   On notebook run, they will be overwritten. The new model filename is
   `product_recommendation_model.pkl` (not `rf_product_model.pkl`).

4. **`browsing_history` in user_profile** — `predict_product()` and `recommend_products()`
   check for `user_profile.get('browsing_history', [])`. The standard `build_user_profile()`
   does not set this key. If browsing context is available at inference, pass it as:
   ```python
   profile = build_user_profile(...)
   profile['browsing_history'] = ['Electronics', 'Fashion']
   ```

5. **`data/suvroo/` not committed** — the suvroo CSV files are likely gitignored as data.
   Confirm they are present locally before running. If not: re-download from Kaggle.

6. **Old datasets still in repo** — `data/career_dataset_large.xlsx`, `data/Dataset.csv`,
   `data/content_based_recommendation_dataset.csv` and `data/e-commerce/` are untracked.
   Safe to delete. Not urgently needed.

---

## Deadline
**27 June 2026, 7 PM** — 21 days remaining as of session date.
