# Agent Handoff — Integration Layer (Member 4)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/integration`
**Deadline:** 27 June 2026, 7 PM

---

## What Changed and Why

The project has **fully pivoted** to item-based collaborative filtering on the UCI Online
Retail dataset. The integration layer's structure is largely the same, but the pipeline
now has **two distinct paths** based on whether the user has purchase history.

| What changed | Old (suvroo) | New (Online Retail) |
|---|---|---|
| Dataset | Synthetic, Indian, ₹ | Real transactions, UK, £ |
| Approach | Rule + ML binary classifier | RF classifiers + ALS CF |
| Pipeline | Single path | Two paths: personalised vs popular |
| Output key `roadmap` | Category path | **Removed** |
| Output key `recommendation_type` | N/A | **New** — "personalised" or "popular" |
| `recommended_products` format | list[tuple] | Flat list of dicts (category + product + score) |
| Products shown | Top-1 category only | Top-3 categories, 3 products each (9 total) |

### Session 3 ML update (important)

`predict_product()` **no longer raises ValueError for empty purchase_history**.
It now works for all users by blending a context model (segment, price_range, month)
with a history model (purchase features), weighted by purchase confidence.

What this means for you:
- You can call `predict_product()` for ANY user — cold-start users get context-model scores
- The two-path routing in `recommend()` can remain as-is (popular path still needed for
  `recommend_products()` which falls back to popularity for cold-start, but category
  selection via `predict_product()` works for everyone)
- Scores from `predict_product()` are now 0–1 probabilities (not cosine similarities)
  — your `recommend()` just sorts by score, so no change needed

---

## Updated Pipeline

```
build_user_profile()
        │
        ▼
apply_rules(user_profile)           → eligible: list[str]
        │
        ├─ if purchase_history is EMPTY ──────────────────────────────
        │   find_popular_categories(eligible, price_range)
        │           → categories: list[tuple[str, int]]   (buyer count scores)
        │   get_popular_products(category, top_n=3)        (per top-3 cat)
        │           → products: list[dict]
        │   recommendation_type = "popular"
        │
        └─ if purchase_history is NON-EMPTY ──────────────────────────
            find_reachable_categories(user_profile, eligible)
                    → shortlist: list[str]
            predict_product(user_profile, candidates=shortlist)
                    → categories: list[tuple[str, float]]   (RF probabilities 0-1)
            recommend_products(user_profile, category, top_n=3)  x 3 calls
                    → products: list[dict]
            recommendation_type = "personalised"
                    │
                    ▼
             recommend() output dict
```

---

## Module Imports

```python
# Member 2
from src.rules_engine import apply_rules

# Member 1
from src.search_module import find_reachable_categories, find_popular_categories

# Member 3 (load artefacts, call inference functions defined in ml_model.ipynb)
import pickle, numpy as np, pandas as pd
item_sim_df       = pickle.load(open('models/similarity_matrix.pkl',  'rb'))
product_catalogue = pickle.load(open('models/product_catalogue.pkl',  'rb'))
model_context     = pickle.load(open('models/model_context.pkl',      'rb'))
model_history     = pickle.load(open('models/model_history.pkl',      'rb'))

# constants
from src.constants import PRODUCT_CATEGORIES, SAMPLE_PROFILES, build_user_profile
```

Note: `model_context.pkl` and `model_history.pkl` are loaded and used inside
`predict_product()`. You don't call them directly — just call `predict_product()`.

---

## Functions to Implement

### `recommend(user_profile) -> dict`

```python
def recommend(user_profile: dict) -> dict:
    """
    Full recommendation pipeline — routes between personalised and popular paths.

    Parameters
    ----------
    user_profile : dict — from build_user_profile() in src/constants.py

    Returns
    -------
    dict with keys:
        recommendation_type  : "personalised" | "popular"
        top_3_categories     : list[tuple[str, float|int]]
        recommended_products : list[dict]  — flat, 9 items max
        eligible             : list[str]
    """
    eligible = apply_rules(user_profile)

    # ── Cold-start path ──────────────────────────────────────────────────
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

    # ── Personalised path ────────────────────────────────────────────────
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

```python
def get_popular_products(category: str, top_n: int = 3) -> list[dict]:
    """Return the most globally purchased products within a category."""
    cat_items = product_catalogue[product_catalogue['category'] == category]
    top = cat_items.nlargest(top_n, 'popularity_rank')
    return [{"category": category, "product": row['Description'], "score": int(row['popularity_rank'])}
            for _, row in top.iterrows()]
```

### `display_recommendation(result) -> None`

```python
def display_recommendation(result: dict) -> None:
    """Pretty-print a recommend() output dict."""
    tag = "Recommended for you" if result['recommendation_type'] == 'personalised' \
          else "Popular right now"

    print(f"\n{'─'*55}")
    print(f"  {tag}")
    print(f"{'─'*55}")
    print(f"  Eligible categories : {result['eligible']}")
    print(f"\n  Top 3 categories:")
    for cat, score in result['top_3_categories']:
        label = f"{score:.4f}" if isinstance(score, float) else f"{score:,} buyers"
        print(f"    - {cat:<28} {label}")
    print(f"\n  Recommended products:")
    for item in result['recommended_products']:
        label = f"{item['score']:.4f}" if isinstance(item['score'], float) else f"{item['score']:,} buyers"
        print(f"    [{item['category']:<24}] {item['product']:<45} {label}")
```

---

## Output Schema

```python
# Personalised (returning user) — scores are RF probabilities (0-1)
{
    "recommendation_type": "personalised",
    "top_3_categories": [
        ("Home Decor",       0.312),
        ("Kitchen & Dining", 0.287),
        ("Seasonal & Gifts", 0.241),
    ],
    "recommended_products": [
        {"category": "Home Decor",       "product": "PICTURE FRAME",    "score": 0.4670},
        {"category": "Home Decor",       "product": "WALL CLOCK",       "score": 0.3820},
        {"category": "Home Decor",       "product": "LANTERN",          "score": 0.3510},
        # ... 6 more items
    ],
    "eligible": ["Home Decor", "Kitchen & Dining", ...],
}

# Popular (cold-start / new user) — scores are integer buyer counts
{
    "recommendation_type": "popular",
    "top_3_categories": [
        ("Home Decor",       8420),
        ("Seasonal & Gifts", 5310),
        ("Kitchen & Dining", 4190),
    ],
    "recommended_products": [
        {"category": "Home Decor", "product": "WHITE METAL LANTERN", "score": 847},
        # ... 8 more items
    ],
    "eligible": ["Home Decor", "Seasonal & Gifts", "Kitchen & Dining"],
}
```

Key points:
- `score` type differs: **float** (personalised, 0–1 probability) vs **int** (popular, buyer count)
- `recommended_products` is always a flat list — iterate directly
- 9 products max (3 per top-3 category)
- No item in `recommended_products` should appear in `user_profile['purchase_history']`

---

## Expected Outputs for SAMPLE_PROFILES

| Profile | recommendation_type | Expected top category |
|---|---|---|
| `new_customer` | popular | Home Decor (most buyers) |
| `gift_buyer` | personalised | Seasonal & Gifts (from history) |
| `home_decorator` | personalised | Home Decor (from history) |
| `kitchen_enthusiast` | personalised | Kitchen & Dining or Food & Confectionery |
| `craft_lover` | personalised | Stationery & Craft (from history) |

---

## Notebook Structure

```
Section 1  — Imports and artefact loading (load 4 pkl files + define encoding constants)
Section 2  — Copy predict_product() and recommend_products() from ml_model.ipynb Section 9
Section 3  — recommend() implementation
Section 4  — get_popular_products() helper
Section 5  — display_recommendation() formatter
Section 6  — Demo: run all 5 SAMPLE_PROFILES
Section 7  — Results discussion
```

---

## What To Do

1. Pull `src/constants.py`, `src/rules_engine.py`, `src/search_module.py`, `models/*.pkl`
2. Copy `predict_product()`, `recommend_products()`, and their helper functions from
   `notebooks/ml_model.ipynb` Section 9 (or load from the pkl artefacts and redefine)
3. Implement `recommend()`, `get_popular_products()`, `display_recommendation()`
4. Run demo on all 5 `SAMPLE_PROFILES` — capture and include output in notebook
5. Verify:
   - `new_customer` → `recommendation_type = "popular"`
   - All 4 returning profiles → `recommendation_type = "personalised"`
   - All output dicts have exactly 4 keys
   - No product in `recommended_products` appears in `purchase_history`
   - Scores are floats (personalised) or integers (popular)
6. Create draft PR: `gh pr create --draft`

## Rules

- `recommend()` signature is fixed — never change it
- `recommendation_type` key is mandatory in every output dict
- Do not push — user pushes manually
