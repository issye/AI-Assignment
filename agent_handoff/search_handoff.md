# Agent Handoff — Search Module (Member 1)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/search`
**Deadline:** 27 June 2026, 7 PM

---

## ✅ STATUS: COMPLETE

| File | Status |
|---|---|
| `src/search_module.py` | ✅ Implemented and tested |
| `notebooks/search_demo.ipynb` | ✅ Fully executed with visualisations |
| `scripts/test_search.py` | ✅ 15/15 tests pass |
| `models/category_similarity.pkl` | ⚠️ Synthetic fixture for now — replaced when Member 3 runs `ml_model.ipynb` |
| `models/product_catalogue.pkl` | ⚠️ Synthetic fixture for now — replaced when Member 3 runs `ml_model.ipynb` |

---

## What This Module Does

The search module sits between the rules engine and the ML model. It takes the broad list of eligible categories from the rules engine and narrows it down to ones actually relevant to the specific customer — using the purchase similarity structure of the data, not just rules or ML predictions.

It also handles the cold-start path entirely: when a customer has no purchase history, the ML model cannot score anything meaningful, so the search module ranks categories by global popularity instead, and the ML model is bypassed.

---

## How It Fits Into the Pipeline

```
User Profile
     │
     ▼
Rules Engine (Member 2)
  apply_rules(profile) → eligible: up to 8 categories
     │
     ▼
Search Module (Member 1)  ◄─── THIS MODULE
     │
     ├── purchase_history NON-EMPTY?
     │     find_reachable_categories(profile, eligible)
     │     → shortlist passed to ML model as candidates
     │
     └── purchase_history EMPTY?
           find_popular_categories(eligible, price_range)
           → ranked by global buyer count
           → ML model bypassed; this IS the recommendation
     │
     ▼
ML Module (Member 3)
  predict_product(profile, candidates=shortlist)
     │
     ▼
Integration (Member 4)
  recommend(profile) → final output dict
```

Member 4 handles the routing decision. This module only implements the two functions below.

---

## The Category Similarity Graph

Member 3's ALS model produces a 3,665 × 3,665 item similarity matrix. This is aggregated to an **8×8 category similarity matrix** saved as `models/category_similarity.pkl`.

Each cell is the average cosine similarity between all products in category A and category B, computed from ALS latent factors. It reflects which categories tend to be bought together in the real data.

This matrix is the graph. Categories are nodes. An edge exists between two categories if their similarity exceeds the threshold (default 0.40).

Example values (approximate):

| | Home Decor | Kitchen & Dining | Seasonal & Gifts | Fashion & Accessories |
|---|---|---|---|---|
| **Home Decor** | 1.00 | 0.81 | 0.74 | 0.22 |
| **Kitchen & Dining** | 0.81 | 1.00 | 0.69 | 0.18 |
| **Seasonal & Gifts** | 0.74 | 0.69 | 1.00 | 0.21 |
| **Fashion & Accessories** | 0.22 | 0.18 | 0.21 | 1.00 |

Fashion & Accessories is isolated from the home/kitchen/seasonal cluster. A Home Decor customer will never reach it within 2 hops above 0.40.

---

## Function 1 — `find_reachable_categories()`

For returning users with purchase history.

```python
def find_reachable_categories(
    user_profile: dict,
    eligible:     list,
    max_hops:     int   = 2,
    threshold:    float = 0.40,
) -> list[str]:
```

**Algorithm — BFS on the category similarity graph:**
1. Start at `user_profile['favourite_category']` (their most-purchased category)
2. Run BFS: at each node, follow edges where similarity > threshold, sorted descending
3. Collect categories reachable within `max_hops` that are also in `eligible`
4. Return in BFS discovery order (nearest categories first)

**Key behaviours:**
- If `favourite_category` is None → returns `eligible` unchanged (no start node)
- Result is always a **subset of `eligible`** — never returns anything the rules engine didn't permit
- Uses `deque` for O(1) pops; early-breaks on sorted neighbours for efficiency

---

## Function 2 — `find_popular_categories()`

For new users with no purchase history (cold-start path).

```python
def find_popular_categories(
    eligible:    list,
    price_range: str,
    top_n:       int = 3,
) -> list[tuple[str, int]]:
```

**What it does:**
1. Filters `product_catalogue` to products in `eligible` categories
2. Groups by category, sums `popularity_rank` (unique buyer count per product)
3. Returns top `top_n` categories ordered by total buyer count

Scores are **integers** (unique buyer counts), not floats.

---

## Worked Example — `home_decorator` (returning user)

```
Profile:
  favourite_category : "Home Decor"
  customer_segment   : "Frequent"
  purchase_history   : 4 items

Rules engine returns: all 8 categories (Frequent buyer → full catalogue)

find_reachable_categories(profile, eligible):

  START: Home Decor

  Hop 1 (sim > 0.40):
    Kitchen & Dining    0.81  ✓ → add
    Seasonal & Gifts    0.74  ✓ → add
    Stationery & Craft  0.61  ✓ → add
    Garden & Outdoor    0.44  ✓ → add
    Fashion & Accessories 0.22 ✗ below threshold → skip

  Hop 2 (neighbours of hop-1 not yet visited):
    Toys & Games        reachable via Kitchen & Dining → add
    Food & Confectionery reachable via Seasonal & Gifts → add

  RESULT: ['Home Decor', 'Kitchen & Dining', 'Seasonal & Gifts',
           'Stationery & Craft', 'Garden & Outdoor',
           'Toys & Games', 'Food & Confectionery']
  (7 of 8 — Fashion & Accessories pruned)

ML model receives shortlist of 7 instead of 8.
```

## Worked Example — `new_customer` (cold-start)

```
Profile:
  purchase_history : []
  customer_segment : "New"
  price_range      : "Low"

Rules engine returns: ["Home Decor", "Seasonal & Gifts", "Kitchen & Dining"]

find_popular_categories(eligible, "Low", top_n=3):
  → [("Home Decor", 8420), ("Seasonal & Gifts", 5310), ("Kitchen & Dining", 4190)]

ML model bypassed. Popularity ranking IS the recommendation.
```

---

## For Member 4 — Integration

```python
from src.search_module import find_reachable_categories, find_popular_categories

# Routing decision is yours — check purchase_history
if not profile['purchase_history']:
    # Cold-start path
    popular = find_popular_categories(eligible, profile['price_range'], top_n=3)
    # popular = [('Home Decor', 8420), ('Seasonal & Gifts', 5310), ...]
    # recommendation_type = "popular", scores are integers
else:
    # Personalised path
    shortlist  = find_reachable_categories(profile, eligible)
    categories = predict_product(profile, candidates=shortlist)  # Member 3
    # recommendation_type = "personalised", scores are floats
```

**Contracts you must respect:**
1. `find_reachable_categories` always returns a subset of `eligible` — safe to pass directly to `predict_product(candidates=...)`
2. `find_popular_categories` scores are **integers** — set `recommendation_type = "popular"` in output dict
3. If `favourite_category` is None, `find_reachable_categories` returns `eligible` unchanged — ML model still runs normally
4. Both functions raise `FileNotFoundError` if `models/*.pkl` are missing — ensure Member 3 runs `ml_model.ipynb` before integration testing

---

## Artefacts

| File | Used by | Contents |
|---|---|---|
| `models/category_similarity.pkl` | `find_reachable_categories` | 8×8 pandas DataFrame, category→category cosine similarity |
| `models/product_catalogue.pkl` | `find_popular_categories` | DataFrame: StockCode, Description, category, avg_price, popularity_rank |

Both generated by `notebooks/ml_model.ipynb` (Member 3). Currently replaced by synthetic fixtures with matching structure for testing.

---

## Constants

```python
from src.constants import PRODUCT_CATEGORIES
```

Never hardcode category names. Always import from constants.
