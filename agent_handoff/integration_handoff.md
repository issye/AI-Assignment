# Agent Handoff — Integration Layer (Member 4)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/integration`
**Deadline:** 27 June 2026, 7 PM

---

## What This Module Does

Member 4 builds the final integration layer — the single `recommend()` function that takes a user profile and returns a complete recommendation output. This is the only function the outside world calls. Everything else (rules, search, ML) is internal machinery that `recommend()` orchestrates.

You are also responsible for the final notebook: `notebooks/career_recommender.ipynb`, which demonstrates the full end-to-end pipeline running on all 5 sample profiles.

---

## How the Full System Works

The system is a sequential pipeline. Each component passes its output to the next:

```
build_user_profile()
        │
        ▼
apply_rules(profile)
        │
        └── eligible: list of up to 8 categories the user is allowed to see
        │
        ├── IF purchase_history is EMPTY (new customer)
        │         │
        │         ▼
        │   find_popular_categories(eligible, price_range)
        │         │
        │         └── categories ranked by global buyer count
        │         │
        │         ▼
        │   get_popular_products(category, top_n=3) × 3 categories
        │         │
        │         └── recommendation_type = "popular"
        │
        └── IF purchase_history is NON-EMPTY (returning customer)
                  │
                  ▼
            find_reachable_categories(profile, eligible)
                  │
                  └── shortlist: categories reachable by BFS from favourite_category
                  │
                  ▼
            predict_product(profile, candidates=shortlist)
                  │
                  └── categories: [(category, score), ...] ranked by RF probability
                  │
                  ▼
            recommend_products(profile, category, top_n=3) × top 3 categories
                  │
                  └── recommendation_type = "personalised"
                  │
                  ▼
            recommend() → output dict
```

**The routing decision is yours.** You check `purchase_history` and decide which path to take. All other modules simply implement their individual functions without knowing which path is active.

---

## Understanding the Two Paths

### Personalised path (returning customers)

A returning customer has purchase history. The ML model can use that history to predict which categories they will buy from next, and the ALS similarity matrix can find specific products similar to what they already own.

Scores in this path are **float probabilities** (0–1) from the Random Forest models.

### Popular path (new customers)

A new customer has no purchase history. The ML model cannot personalise — it has nothing to go on. Instead, we fall back to showing the most globally popular categories and products within their eligible set.

Scores in this path are **integer buyer counts** (e.g. 8420 unique buyers).

**The `recommendation_type` flag tells the display layer which type of scores to expect.** This matters for formatting — floats display as probabilities, integers display as "X buyers".

---

## Output Schema

Your `recommend()` function must return exactly this structure:

```python
{
    "recommendation_type":  "personalised" | "popular",
    "top_3_categories":     [(category, score), (category, score), (category, score)],
    "recommended_products": [
        {"category": str, "product": str, "price": float, "score": float_or_int},
        ...  # 9 items total: 3 products × 3 categories
    ],
    "eligible":             [category, ...],
}
```

Key rules:
- `top_3_categories` is always exactly 3 items
- `recommended_products` is a **flat list** of up to 9 dicts — 3 per top category
- `price` is always a float (£, 2 decimal places)
- `score` type differs: **float** for personalised, **int** for popular
- No product in `recommended_products` should appear in `purchase_history`

---

## Functions to Implement

### `recommend(user_profile) -> dict`

The main orchestrator. Routes between personalised and popular paths.

```python
def recommend(user_profile: dict) -> dict:
    eligible = apply_rules(user_profile)

    # Cold-start path
    if not user_profile['purchase_history']:
        categories = find_popular_categories(eligible, user_profile['price_range'], top_n=3)
        products   = []
        for cat, _ in categories:
            products += get_popular_products(cat, top_n=3)
        return {
            "recommendation_type":  "popular",
            "top_3_categories":     categories,
            "recommended_products": products,
            "eligible":             eligible,
        }

    # Personalised path
    shortlist  = find_reachable_categories(user_profile, eligible)
    categories = predict_product(user_profile, candidates=shortlist)[:3]
    products   = []
    for cat, _ in categories:
        products += recommend_products(user_profile, category=cat, top_n=3)

    return {
        "recommendation_type":  "personalised",
        "top_3_categories":     categories,
        "recommended_products": products,
        "eligible":             eligible,
    }
```

### `get_popular_products(category, top_n=3) -> list[dict]`

Returns the most globally purchased products in a category. Used only on the cold-start path.

```python
def get_popular_products(category: str, top_n: int = 3) -> list[dict]:
    cat_items = product_catalogue[product_catalogue['category'] == category]
    top = cat_items.nlargest(top_n, 'popularity_rank')
    return [
        {
            "category": category,
            "product":  row['Description'],
            "price":    round(float(row['avg_price']), 2),
            "score":    int(row['popularity_rank']),
        }
        for _, row in top.iterrows()
    ]
```

### `display_recommendation(result) -> None`

Pretty-prints the output dict. Handles both score types cleanly.

```python
def display_recommendation(result: dict) -> None:
    tag = "Recommended for you" if result['recommendation_type'] == 'personalised' \
          else "Popular right now"

    print(f"\n{'─'*60}")
    print(f"  {tag}  ({result['recommendation_type']})")
    print(f"{'─'*60}")
    print(f"  Eligible : {result['eligible']}")
    print(f"\n  Top 3 categories:")
    for cat, score in result['top_3_categories']:
        label = f"{score:.4f}" if isinstance(score, float) else f"{score:,} buyers"
        print(f"    {cat:<28} {label}")
    print(f"\n  Recommended products:")
    for item in result['recommended_products']:
        label = f"{item['score']:.4f}" if isinstance(item['score'], float) \
                else f"{item['score']:,} buyers"
        print(f"    [{item['category'][:22]:<22}] "
              f"{item['product'][:38]:<38} "
              f"£{item['price']:>5.2f}  {label}")
```

---

## Module Imports

```python
# Member 2
from src.rules_engine import apply_rules

# Member 1
from src.search_module import find_reachable_categories, find_popular_categories

# Member 3 — load artefacts then copy predict_product() and recommend_products()
# from notebooks/ml_model.ipynb Section 9
import pickle, numpy as np, pandas as pd
item_sim_df       = pickle.load(open('models/similarity_matrix.pkl',  'rb'))
product_catalogue = pickle.load(open('models/product_catalogue.pkl',  'rb'))
model_context     = pickle.load(open('models/model_context.pkl',      'rb'))
model_history     = pickle.load(open('models/model_history.pkl',      'rb'))

# encoding constants (must match training — copy from ml_model.ipynb Section 9)
segment_map  = {'New': 0, 'Occasional': 1, 'Frequent': 2}
price_map    = {'Low': 0, 'Mid-Low': 1, 'Mid-High': 2, 'High': 3}
from src.constants import PRODUCT_CATEGORIES, SAMPLE_PROFILES, build_user_profile
all_fav_cats = PRODUCT_CATEGORIES + ['Unknown']

# Constants
from src.constants import PRODUCT_CATEGORIES, SAMPLE_PROFILES, build_user_profile
```

**Important:** Copy `predict_product()`, `recommend_products()`, `extract_context_features()`, and `extract_history_features()` directly from `notebooks/ml_model.ipynb` Section 9 into your notebook. Do not re-implement them.

---

## Expected Outputs for SAMPLE_PROFILES

| Profile | Expected type | Expected top category |
|---|---|---|
| `new_customer` | popular | Home Decor (most global buyers) |
| `gift_buyer` | personalised | Seasonal & Gifts (from history) |
| `home_decorator` | personalised | Home Decor (from history) |
| `kitchen_enthusiast` | personalised | Kitchen & Dining or Food & Confectionery |
| `craft_lover` | personalised | Stationery & Craft (from history) |

---

## Notebook Structure (`notebooks/career_recommender.ipynb`)

```
Section 1  — Imports and artefact loading (4 pkl files + encoding constants)
Section 2  — Copy predict_product() and recommend_products() from ml_model.ipynb Section 9
Section 3  — recommend() implementation
Section 4  — get_popular_products() helper
Section 5  — display_recommendation() formatter
Section 6  — Demo: run all 5 SAMPLE_PROFILES and show full output
Section 7  — Results discussion
```

---

## Verification Checklist

Before creating your PR, verify every item:

- [ ] `new_customer` → `recommendation_type = "popular"`
- [ ] All 4 returning profiles → `recommendation_type = "personalised"`
- [ ] Every output dict has exactly 4 keys: `recommendation_type`, `top_3_categories`, `recommended_products`, `eligible`
- [ ] `recommended_products` is a flat list with exactly 9 items (3 per category)
- [ ] Every product dict has keys: `category`, `product`, `price`, `score`
- [ ] No product in `recommended_products` appears in `purchase_history`
- [ ] Scores are floats for personalised, integers for popular

---

## What To Do

1. Pull `main` — get `src/constants.py`, `src/rules_engine.py`, `src/search_module.py`, `models/*.pkl`
2. Create `notebooks/career_recommender.ipynb` following the structure above
3. Copy `predict_product()`, `recommend_products()`, and helper functions from `ml_model.ipynb` Section 9
4. Implement `recommend()`, `get_popular_products()`, `display_recommendation()`
5. Run demo on all 5 SAMPLE_PROFILES — include printed output in the notebook
6. Create draft PR: `gh pr create --draft`

## Rules

- `recommend()` signature is frozen
- `recommendation_type` key is mandatory in every output
- Do not push — user pushes manually
