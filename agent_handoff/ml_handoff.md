# Agent Handoff — ML Module (Member 3)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/ml`
**Dataset:** `data/suvroo/customer_data_collection.csv` + `data/suvroo/product_recommendation_data.csv`
**Deadline:** 27 June 2026, 7 PM

---

## Why Binary Purchase-Probability Instead of Category Classification

The previous approach trained an 8-class category classifier on a synthetic dataset where all
demographic mutual information ≈ 0. This produced a 20–55% accuracy ceiling — a data problem,
not a modelling problem.

The redesign reframes the task:
> Instead of asking *"which category does this user prefer?"*
> we ask *"would this specific user buy this specific product?"*

This unlocks richer signal: user-product interaction features (browsing match, price fit) that
a category classifier cannot access. It also enables the pipeline to return **specific named
products** rather than just categories.

The model sits as a **re-ranker** in the multi-agent pipeline:
```
User Profile → Rules Engine → A* Search → [ML re-ranks] → recommend_products() → Output
```

Rules Engine and A* narrow the candidate space first. ML only discriminates among 3–5
pre-filtered categories — a far easier task than classifying across the full product space.

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
| `Geographical_Location` | str | Drop — mismatches customer cities |
| `Similar_Product_List` | str (list) | e.g. `['Smartphone', 'Laptop']` |
| `Probability_of_Recommendation` | float | 0.1–1.0 — product quality signal |

### Category mapping (suvroo → PRODUCT_CATEGORIES)
| In dataset | In constants.py |
|---|---|
| Fitness | Sports |
| Home Decor | Home & Garden |
| Books, Beauty, Electronics, Fashion | unchanged |

---

## Training Data Construction — The Join Problem

The two CSV files share **no direct join key**. A customer's `Purchase_History` contains
subcategory names (e.g. `"Smartphone"`), not product IDs. The workaround uses `Subcategory`
as a bridge:

```
Customer bought "Smartphone"
→ find all products where Subcategory = "Smartphone" in product file
→ pair customer with those products → label = 1 (positive)

Customer did NOT buy "Jeans"
→ randomly sample products where Subcategory = "Jeans"
→ pair customer with those products → label = 0 (negative)
```

**Assumption:** if a customer bought a Smartphone, they would buy any Smartphone in the
catalogue. This is approximate — we don't know the exact brand or price point purchased —
but the model learns generalised patterns from the combination of user and product features,
not from memorising individual transactions.

```
Total pairs: ~60K positive + ~180K negative = ~240K training rows
Negative:Positive ratio = 3:1
```

---

## Feature Engineering (29 features total)

### Why 29 and not 30 — the subcat_enc removal

An earlier version included `subcat_enc` (subcategory label encoding) as a product feature.
This caused **data leakage**:

- Positive pairs are constructed by matching customer purchase items to products by `Subcategory`
- So every positive pair has `subcat_enc` = the encoded value of what the customer bought
- Every negative pair has a random `subcat_enc` value

The model learned one trivial rule: *"does subcat_enc match the customer's purchase history?"*
— which is just replaying the construction logic, not genuine purchase intent. This inflated
ROC-AUC to 0.9997 artificially. `subcat_enc` was removed to force the model to learn from
real user-product affinity signals.

### User features (19)
```
Age, gender_enc, city_enc, is_urban,
Avg_Order_Value, log_avg_order, price_range_enc,
segment_enc, season_enc, holiday_enc,
n_browse_cats, n_purchase_items, is_multi_buyer,
browsed_books, browsed_beauty, browsed_electronics,
browsed_fashion, browsed_sports, browsed_home_and_garden
```

The 6 `browsed_*` binary flags encode stated category interest directly from `Browsing_History`.

### Product features (8)
```
category_enc, Price, log_price,
Product_Rating, Average_Rating_of_Similar_Products,
Customer_Review_Sentiment_Score, Probability_of_Recommendation,
brand_enc
```

Note: `subcat_enc` deliberately excluded — see above.

### Cross features (2)
```
price_fit        = |Avg_Order_Value - Price| / Avg_Order_Value
category_browsed = 1 if product's category appears in user's Browsing_History else 0
```

Cross features connect the user and product sides. `category_browsed` is the dominant
signal after `subcat_enc` removal — it captures whether the user has stated interest in
the product's category.

---

## Why ROC-AUC as Primary Metric

A recommender's job is **ranking** — putting products the user would buy above ones they
wouldn't, regardless of a classification threshold. ROC-AUC measures exactly this: the
probability that a randomly chosen positive pair scores higher than a randomly chosen
negative pair. A random model scores 0.500; a perfect ranker scores 1.000.

Accuracy is misleading here because the dataset is imbalanced (75% negative pairs). A model
that always predicts "not bought" achieves 75% accuracy while being useless.

---

## Models Trained

| Model | Role | Notes |
|---|---|---|
| Decision Tree | Baseline | GridSearchCV, interpretable splits |
| KNN | Distance-based | StandardScaler pipeline, GridSearchCV |
| Random Forest | Primary candidate | `class_weight='balanced'`, GridSearchCV |
| Gradient Boosting | Comparison | `sample_weight` balancing, GridSearchCV |

Winner → wrapped in `CalibratedClassifierCV(method='isotonic')` so output probabilities
reflect true purchase likelihoods, not just rankings.

---

## Honest Limitations

**High ROC-AUC is partially artificial.** After removing `subcat_enc`, `category_browsed`
became the dominant feature — it correlates with the construction pattern (customers browse
categories they buy from). The model is partly learning the construction rule rather than
genuine demographic-driven purchase intent.

**Within-category ranking is quality-driven, not personalised.** `recommend_products()`
uses combined scoring because ML cannot differentiate products within the same category
(all share the same `category_enc`). Quality signals provide the within-category ranking.

**No brand preference signal.** Purchase history records subcategory names only, not brands.
The 4 generic brands (A/B/C/D) carry no learnable preference signal.

**Single snapshot.** 1 row per customer — no recency, frequency, or sequence signals.

---

## Public Interface

### `predict_product()` — category re-ranker

```python
def predict_product(user_profile: dict, candidates: list = None) -> list[tuple[str, float]]:
    """
    Re-rank product categories by predicted purchase probability.

    Parameters
    ----------
    user_profile : dict  — from build_user_profile() in src/constants.py
                           MUST include 'browsing_history' key for meaningful scores
                           e.g. profile['browsing_history'] = ['Electronics', 'Sports']
    candidates   : list[str] — category names from A* shortlist
                               if None, scores all 6 PRODUCT_CATEGORIES

    Returns
    -------
    list[tuple[str, float]] — [(category, confidence), ...] sorted descending
    Example: [("Electronics", 0.996), ("Sports", 0.994), ("Home & Garden", 0.000)]

    Note: unbrowsed categories score near 0.0 — this is expected behaviour.
    The Rules Engine and A* ensure only eligible categories are passed as candidates.
    """
```

### `recommend_products()` — product-level output

```python
def recommend_products(user_profile: dict, category: str, top_n: int = 3) -> list[tuple[str, float]]:
    """
    Recommend specific named products within a category.

    Scoring: combined signal
      final_score = 0.5 * ML_purchase_probability + 0.5 * quality_score
      quality_score = 0.4*(Rating/5) + 0.3*Sentiment + 0.3*Prob_of_Recommendation

    ML probability alone collapses to near-zero within a single category
    (category_enc is identical for all products in the same category).
    Quality signals provide within-category discrimination.

    Parameters
    ----------
    user_profile : dict  — from build_user_profile() in src/constants.py
    category     : str   — one of PRODUCT_CATEGORIES
    top_n        : int   — number of products to return (default 3)

    Returns
    -------
    list[tuple[str, float]] — [(subcategory_name, combined_score), ...] descending
    Example: [("Smartwatch", 0.332), ("Smartphone", 0.329), ("Headphones", 0.326)]

    Note: scores ~0.3 are expected — they reflect 0.5 * quality_score since
    ML probability ≈ 0 within a single category. Ranking order is what matters.
    """
```

### Critical usage note — browsing_history

`build_user_profile()` does not set `browsing_history` by default. Both inference functions
require it for meaningful scores. Always set it before calling:

```python
profile = build_user_profile(42, "Male", "Mumbai", 4500.0)
profile['browsing_history'] = ['Electronics', 'Sports']   # required
result = predict_product(profile, candidates=['Electronics', 'Sports', 'Home & Garden'])
```

Without `browsing_history`, all `browsed_*` flags = 0 and `category_browsed` = 0,
causing all categories to score near 0.000.

---

## Saved Artefacts (models/ directory — 8 files)

| File | Contents |
|---|---|
| `models/product_recommendation_model.pkl` | CalibratedClassifierCV(best_model) |
| `models/encoder_gender.pkl` | LabelEncoder for Gender |
| `models/encoder_city.pkl` | LabelEncoder for Location |
| `models/encoder_segment.pkl` | LabelEncoder for Customer_Segment |
| `models/encoder_season.pkl` | LabelEncoder for Season |
| `models/encoder_category.pkl` | LabelEncoder for Category_mapped |
| `models/encoder_brand.pkl` | LabelEncoder for Brand |
| `models/product_catalogue.pkl` | Full product DataFrame for inference |

Note: `encoder_subcat.pkl` is no longer saved — `subcat_enc` was removed from the feature set.
Encoders must be loaded from pkl files at inference — never refit them, as encoding order
must match exactly what the model was trained on.

---

## Notebook Structure (`notebooks/ml_model.ipynb`)

| Section | Content |
|---|---|
| 1 | Setup & Imports |
| 2 | Data Loading (both files, drop unnamed cols, map categories) |
| 3 | EDA (demographics, spend tiers, browse/purchase patterns, co-purchase heatmap) |
| 4 | Feature Engineering (customer features, product features, pair construction, cross features) |
| 5 | Preprocessing (encoders, FEATURE_COLS assembly — 29 features) |
| 6 | Train/Test Split (80/20 stratified on bought) |
| 7 | Model Training (DT, KNN, RF, GBM + calibration) |
| 8 | Evaluation (ROC-AUC, calibration diagram, classification report, feature importance, CV, P@K, per-category AUC) |
| 9 | Results Discussion |
| 10 | Save Artefacts (8 pkl files) |
| 11 | predict_product() definition + tests |
| 12 | recommend_products() definition + tests |
| 13 | Integration demo (all 5 SAMPLE_PROFILES) |

---

## Verification Checklist

After running the notebook (`Kernel → Restart & Run All`):

```python
import os, joblib
from src.constants import SAMPLE_PROFILES

# 1. Artefacts saved (8 files)
assert os.path.exists('../models/product_recommendation_model.pkl')
assert not os.path.exists('../models/encoder_subcat.pkl')  # deliberately removed

# 2. predict_product with browsing_history
profile = {**SAMPLE_PROFILES['tech_spender'], 'browsing_history': ['Electronics', 'Sports']}
result = predict_product(profile, candidates=['Sports', 'Electronics', 'Home & Garden'])
assert result[0][0] == 'Electronics'   # browsed Electronics → ranked first
assert result[-1][1] == 0.0            # unbrowsed category → near zero

# 3. recommend_products returns 3 products sorted descending
products = recommend_products(profile, 'Electronics')
assert len(products) == 3
assert products[0][1] >= products[1][1] >= products[2][1]
assert products[0][0] in {'Smartphone', 'Laptop', 'Headphones', 'Smartwatch'}

# 4. Scores are in expected range (~0.3 for recommend_products)
assert 0.1 < products[0][1] < 0.6   # quality-driven, not near 0 or near 1
```

---

## Key Rules
- Never change `predict_product()` or `recommend_products()` signatures
- Never change `build_user_profile()` signature
- Always pass `browsing_history` in user_profile before calling inference functions
- All constants come from `src/constants.py` — never hardcode
- Do not push — user pushes manually
- PRs must be draft: `gh pr create --draft`
- Windows environment — use PowerShell/CMD syntax in terminal
