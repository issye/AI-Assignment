# Agent Handoff — ML Module (Member 3)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/ml`
**Dataset:** `data/suvroo/customer_data_collection.csv` + `data/suvroo/product_recommendation_data.csv`
**Deadline:** 27 June 2026, 7 PM

---

## What Changed and Why

The project previously used `ecommerce_customer_behavior_dataset_v2.csv` (synthetic dataset, all demographic MI ≈ 0). It has been replaced with the **suvroo two-file dataset** which enables product-level recommendations.

The ML module has been redesigned from:
> Category classifier (8 classes, ~20–55% accuracy ceiling, synthetic data)

To:
> **Binary purchase-probability model** trained on user-product pairs, returning specific named products

---

## Datasets

### `customer_data_collection.csv` (10,000 rows)
| Column | Type | Notes |
|---|---|---|
| `Customer_ID` | str | C1000–C9999 |
| `Age` | int | 18–60 |
| `Gender` | str | Female / Male / Other |
| `Location` | str | 5 Indian cities |
| `Browsing_History` | str (list) | e.g. `['Electronics', 'Fitness']` |
| `Purchase_History` | str (list) | e.g. `['Smartphone', 'Yoga Mat']` |
| `Customer_Segment` | str | New Visitor / Occasional Shopper / Frequent Buyer |
| `Avg_Order_Value` | float | ₹500–₹5,000 — PRIMARY spend signal |
| `Holiday` | str | Yes / No |
| `Season` | str | Spring / Summer / Autumn / Winter |
| `Unnamed: 10` | — | Drop this column |

### `product_recommendation_data.csv` (10,000 rows)
| Column | Type | Notes |
|---|---|---|
| `Product_ID` | str | P2000–P11999 |
| `Category` | str | 6 categories (see mapping below) |
| `Subcategory` | str | 24 subcategories (4 per category) |
| `Price` | int | ₹100–₹5,000 |
| `Brand` | str | Brand A/B/C/D |
| `Product_Rating` | float | 2.0–5.0 |
| `Average_Rating_of_Similar_Products` | float | 2.0–5.0 |
| `Customer_Review_Sentiment_Score` | float | 0.0–1.0 |
| `Holiday` | str | Yes / No |
| `Season` | str | Spring / Summer / Autumn / Winter |
| `Geographical_Location` | str | Drop — mismatches customer cities |
| `Similar_Product_List` | str (list) | e.g. `['Smartphone', 'Laptop']` |
| `Probability_of_Recommendation` | float | 0.1–1.0 — use as a product quality feature |
| `Unnamed: 13`, `Unnamed: 14` | — | Drop these columns |

### Category mapping (suvroo → PRODUCT_CATEGORIES)
| In dataset | In constants.py |
|---|---|
| Fitness | Sports |
| Home Decor | Home & Garden |
| Books | Books |
| Beauty | Beauty |
| Electronics | Electronics |
| Fashion | Fashion |

---

## Architecture: Binary Purchase-Probability Model

### Training data construction
No direct join key exists between the two files. Training pairs are built implicitly:

```
For each customer:
  Purchase_History = ['Smartphone', 'Yoga Mat']
  
  Positive pairs (bought=1):
    customer features + Smartphone product features → 1
    customer features + Yoga Mat product features   → 1
    (match by Subcategory column in product file)

  Negative pairs (bought=0):
    sample 3 products per positive NOT in purchase history → 0

Total: ~20,000 positive + ~60,000 negative = ~80,000 training rows
Negative:Positive ratio = 3:1
```

### Feature groups (~30 total)

**User features (19):**
```
Age, gender_enc, city_enc, is_urban,
Avg_Order_Value, log_avg_order, price_range_enc,
segment_enc, season_enc, holiday_enc,
n_browse_cats, n_purchase_items, is_multi_buyer,
browsed_books, browsed_beauty, browsed_electronics,
browsed_fashion, browsed_sports, browsed_home_and_garden
```

**Product features (9):**
```
category_enc, subcat_enc, Price, log_price,
Product_Rating, avg_rating_similar, sentiment_score,
prob_recommendation, brand_enc
```

**Cross features (2):**
```
price_fit         = |Avg_Order_Value - Price| / Avg_Order_Value
category_browsed  = 1 if product category in user Browsing_History else 0
```

### Target
`bought` — binary (1 = purchased, 0 = not purchased)

### Models trained
1. Decision Tree (GridSearchCV, baseline)
2. KNN (StandardScaler pipeline, GridSearchCV)
3. Random Forest (GridSearchCV, class_weight='balanced') ← primary
4. Gradient Boosting (GridSearchCV, sample_weight balancing)
→ Best model → `CalibratedClassifierCV(method='isotonic')`

### Primary evaluation metric
**ROC-AUC** (measures ranking quality, not just accuracy). Target: > 0.70

---

## Public Interface

### `predict_product()` — kept for A* re-ranking compatibility
```python
def predict_product(user_profile: dict, candidates: list = None) -> list[tuple[str, float]]:
    """
    Re-rank product categories by predicted purchase probability.

    Parameters
    ----------
    user_profile : dict  — from build_user_profile() in src/constants.py
    candidates   : list[str] — category names from A* shortlist
                               if None, scores all 6 PRODUCT_CATEGORIES

    Returns
    -------
    list[tuple[str, float]] — [(category, confidence), ...] sorted descending
    Example: [("Electronics", 0.84), ("Sports", 0.61), ("Fashion", 0.42)]
    """
```

### `recommend_products()` — NEW product-level function
```python
def recommend_products(user_profile: dict, category: str, top_n: int = 3) -> list[tuple[str, float]]:
    """
    Recommend specific named products within a category for a user.

    Parameters
    ----------
    user_profile : dict  — from build_user_profile() in src/constants.py
    category     : str   — one of PRODUCT_CATEGORIES
    top_n        : int   — number of products to return (default 3)

    Returns
    -------
    list[tuple[str, float]] — [(product_subcategory_name, score), ...] descending
    Example: [("Smartphone", 0.91), ("Laptop", 0.87), ("Headphones", 0.73)]
    """
```

---

## Saved Artefacts (models/ directory)

After running the notebook: `Kernel → Restart & Run All`

| File | Contents |
|---|---|
| `models/product_recommendation_model.pkl` | CalibratedClassifierCV(best_model) |
| `models/encoder_gender.pkl` | LabelEncoder for Gender |
| `models/encoder_city.pkl` | LabelEncoder for Location |
| `models/encoder_segment.pkl` | LabelEncoder for Customer_Segment |
| `models/encoder_season.pkl` | LabelEncoder for Season |
| `models/encoder_category.pkl` | LabelEncoder for Category |
| `models/encoder_subcat.pkl` | LabelEncoder for Subcategory |
| `models/encoder_brand.pkl` | LabelEncoder for Brand |
| `models/product_catalogue.pkl` | Full product DataFrame (for inference) |

---

## Notebook Structure (`notebooks/ml_model.ipynb`)

Full rewrite — ~55 cells across 13 sections:

| Section | Cells | Content |
|---|---|---|
| 1 | 2 | Setup & Imports |
| 2 | 3 | Data Loading (both files, drop unnamed cols) |
| 3 | 8 | EDA (demographics, spend, browse/purchase patterns, co-purchase heatmap) |
| 4 | 6 | Feature Engineering (customer features, product features, pair construction, cross features) |
| 5 | 3 | Preprocessing (encoders, FEATURE_COLS assembly) |
| 6 | 2 | Train/Test Split (80/20 stratified on bought) |
| 7 | 8 | Model Training (DT, KNN, RF, GBM, calibration) |
| 8 | 8 | Evaluation (ROC-AUC, F1, Precision@K, feature importance, cross-val) |
| 9 | 1 | Results Discussion |
| 10 | 2 | Save Artefacts |
| 11 | 3 | predict_product() definition + tests |
| 12 | 3 | recommend_products() definition + tests |
| 13 | 3 | Integration demo (all 5 SAMPLE_PROFILES) |

---

## Verification Checklist

After running the notebook:

```python
# 1. Imports work
from src.constants import SAMPLE_PROFILES

# 2. Models saved
import os
assert os.path.exists('models/product_recommendation_model.pkl')

# 3. predict_product smoke test
result = predict_product(SAMPLE_PROFILES['tech_spender'])
assert result[0][0] == 'Electronics'

result2 = predict_product(SAMPLE_PROFILES['tech_spender'],
                          candidates=['Sports', 'Electronics', 'Home & Garden'])
assert result2[0][0] == 'Electronics'

# 4. recommend_products smoke test
products = recommend_products(SAMPLE_PROFILES['tech_spender'], 'Electronics')
assert len(products) == 3
assert products[0][1] > products[1][1]  # sorted descending
assert products[0][0] in ['Smartphone', 'Laptop', 'Headphones', 'Smartwatch']

# 5. ROC-AUC > 0.70 on test set (check Section 8 output)
```

---

## Key Rules
- Never change `predict_product()` or `recommend_products()` signatures
- Never change `build_user_profile()` signature
- All constants come from `src/constants.py` — never hardcode
- Do not push — user pushes manually
- PRs must be draft: `gh pr create --draft`
- Windows environment — use PowerShell/CMD syntax in terminal
