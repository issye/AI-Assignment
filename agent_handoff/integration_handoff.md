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
| Approach | Rule + ML binary classifier | Item-based CF + popularity fallback |
| Pipeline | Single path | Two paths: personalised vs popular |
| Output key `roadmap` | Category path | **Removed** |
| Output key `basket_completion` | N/A | **Removed** |
| Output key `recommendation_type` | N/A | **New** — "personalised" or "popular" |
| `recommended_products` format | list[tuple] | Flat list of dicts (category + product + score) |
| Products shown | Top-1 category only | Top-3 categories, 3 products each (9 total) |

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
        │           → categories: list[tuple[str, int]]   (count scores)
        │   get_popular_products(category, top_n=3)        (per top-3 cat)
        │           → products: list[dict]
        │   recommendation_type = "popular"
        │
        └─ if purchase_history is NON-EMPTY ──────────────────────────
            find_reachable_categories(user_profile, eligible)
                    → shortlist: list[str]
            predict_product(user_profile, candidates=shortlist)
                    → categories: list[tuple[str, float]]  (CF scores)
            recommend_products(user_profile, category, top_n=3)  × 3 calls
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

# Member 3 (defined in notebook, load from models/ and call directly)
import pickle
item_sim_df       = pickle.load(open('models/similarity_matrix.pkl',   'rb'))
product_catalogue = pickle.load(open('models/product_catalogue.pkl',   'rb'))

# constants
from src.constants import PRODUCT_CATEGORIES, SAMPLE_PROFILES, build_user_profile
```

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
    """
    Return the most globally purchased products within a category.
    Used for cold-start users — no purchase history needed.
    """
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
        print(f"    • {cat:<28} {label}")
    print(f"\n  Recommended products:")
    for item in result['recommended_products']:
        label = f"{item['score']:.4f}" if isinstance(item['score'], float) else f"{item['score']:,} buyers"
        print(f"    [{item['category']:<24}] {item['product']:<45} {label}")
```

---

## Output Schema

```python
# ── Personalised (returning user) ──────────────────────────────────────────
{
    "recommendation_type": "personalised",

    "top_3_categories": [
        ("Home Decor",       0.8214),
        ("Kitchen & Dining", 0.7103),
        ("Seasonal & Gifts", 0.5832),
    ],

    "recommended_products": [
        {"category": "Home Decor",       "product": "PICTURE FRAME",              "score": 0.8100},
        {"category": "Home Decor",       "product": "WOODEN PHOTO FRAME",         "score": 0.7420},
        {"category": "Home Decor",       "product": "WALL CLOCK",                 "score": 0.6310},
        {"category": "Kitchen & Dining", "product": "CERAMIC MUG",                "score": 0.7700},
        {"category": "Kitchen & Dining", "product": "TEAPOT",                     "score": 0.6900},
        {"category": "Kitchen & Dining", "product": "CAKE STAND",                 "score": 0.6100},
        {"category": "Seasonal & Gifts", "product": "CHRISTMAS WREATH",           "score": 0.6400},
        {"category": "Seasonal & Gifts", "product": "GIFT BAG",                   "score": 0.5800},
        {"category": "Seasonal & Gifts", "product": "ADVENT CALENDAR",            "score": 0.5100},
    ],

    "eligible": ["Home Decor", "Kitchen & Dining", "Seasonal & Gifts", ...],
}

# ── Popular (cold-start / new user) ────────────────────────────────────────
{
    "recommendation_type": "popular",

    "top_3_categories": [
        ("Home Decor",       8420),   # unique buyer count
        ("Seasonal & Gifts", 5310),
        ("Kitchen & Dining", 4190),
    ],

    "recommended_products": [
        {"category": "Home Decor",       "product": "WHITE METAL LANTERN",        "score": 847},
        {"category": "Home Decor",       "product": "CREAM CUPID COAT HANGER",    "score": 743},
        {"category": "Home Decor",       "product": "PICTURE FRAME",              "score": 698},
        # ... 6 more items
    ],

    "eligible": ["Home Decor", "Seasonal & Gifts", "Kitchen & Dining"],
}
```

Key points:
- `score` is cosine similarity float (personalised) or integer buyer count (popular)
- `recommended_products` is a flat list — iterate directly in `display_recommendation()`
- 9 products max (3 per top-3 category)
- No item in `recommended_products` should appear in `user_profile['purchase_history']`

---

## Expected Outputs for SAMPLE_PROFILES

| Profile | recommendation_type | Expected top category |
|---|---|---|
| `new_customer` | popular | Home Decor |
| `gift_buyer` | personalised | Seasonal & Gifts |
| `home_decorator` | personalised | Home Decor |
| `kitchen_enthusiast` | personalised | Kitchen & Dining |
| `craft_lover` | personalised | Stationery & Craft |

---

## Notebook Structure

```
Section 1  — Imports and artefact loading
Section 2  — recommend() implementation
Section 3  — get_popular_products() helper
Section 4  — display_recommendation() formatter
Section 5  — Demo: run all 5 SAMPLE_PROFILES
Section 6  — Results discussion (fill in actual outputs)
```

---

## What To Do

1. Pull `src/constants.py`, `src/rules_engine.py`, `src/search_module.py`, `models/*.pkl`
   from `main` (all updated for pivot)
2. Implement `recommend()`, `get_popular_products()`, `display_recommendation()`
3. Run demo on all 5 `SAMPLE_PROFILES` — capture and include output in notebook
4. Verify:
   - `new_customer` → `recommendation_type = "popular"`
   - All 4 returning profiles → `recommendation_type = "personalised"`
   - All output dicts have 4 keys
   - No product in `recommended_products` appears in `purchase_history`
   - Scores are floats (personalised) or integers (popular)
5. Write Section 6 discussion explaining the two-path design
6. Create draft PR: `gh pr create --draft`

## Rules

- `recommend()` signature is fixed — never change it
- Always check `purchase_history` before calling CF functions
  (predict_product and recommend_products will error on empty history)
- `recommendation_type` key is mandatory in every output dict
- Do not push — user pushes manually
