# Agent Brief — Member 1: A* Search
# CIC6314 Smart Product Recommendation System

## Your job
Rebuild `notebooks/career_recommender.ipynb` (your A* section) for the Smart Product Recommendation domain. The domain switched from Career Recommendation. The algorithm core (A* priority queue logic) is unchanged — only the graph data and three helper functions change.

---

## Context: Why the domain switched
The original career dataset had randomly assigned labels (8.3% accuracy — random chance). The team switched to an e-commerce dataset (`ecommerce_customer_behavior_dataset_v2.csv`) where `median_spend` strongly predicts product category (MI = 0.20). The reachability-first architecture was chosen because demographic features (Age, Gender) have weak direct signal — spend is the primary discriminator.

---

## Pipeline (your position)
```
Rules Engine (Member 2)  →  eligible product categories (list[str])
        │
        ▼
A* Search (YOU)          →  paths + costs per eligible category
        │
        ▼
ML Model (Member 3)      →  re-ranks candidates by predicted preference
        │
        ▼
Integration (Member 4)   →  Top-3 + purchase journey roadmap
```

A* runs **once per eligible category** from the rules engine — not once toward a single user-specified target. The user never inputs a target career/product. A* outputs a dict of paths and costs which the ML model uses as its candidate shortlist.

---

## Your function signature (unchanged interface pattern)

```python
find_product_path(user_profile: dict, target_category: str) -> list[str]
# Returns: ordered category steps from start to target, inclusive
# Returns: [] if no path exists
```

Member 4 calls this once per eligible category:
```python
eligible = apply_rules(user_profile)  # e.g. ["Beauty", "Electronics", "Sports"]
paths = {cat: find_product_path(user_profile, cat) for cat in eligible}
```

---

## New PRODUCT_GRAPH — replace CAREER_GRAPH entirely

```python
PRODUCT_GRAPH = {
    "Books": {
        "avg_price": 56,
        "neighbors": ["Food", "Beauty", "Toys"],
    },
    "Food": {
        "avg_price": 71,
        "neighbors": ["Books", "Beauty", "Toys"],
    },
    "Beauty": {
        "avg_price": 112,
        "neighbors": ["Food", "Fashion", "Toys"],
    },
    "Toys": {
        "avg_price": 169,
        "neighbors": ["Beauty", "Fashion", "Sports"],
    },
    "Fashion": {
        "avg_price": 276,
        "neighbors": ["Beauty", "Sports", "Home & Garden"],
    },
    "Sports": {
        "avg_price": 493,
        "neighbors": ["Fashion", "Home & Garden", "Electronics"],
    },
    "Home & Garden": {
        "avg_price": 691,
        "neighbors": ["Sports", "Electronics"],
    },
    "Electronics": {
        "avg_price": 1767,
        "neighbors": ["Home & Garden", "Sports"],
    },
}
```

Edges represent natural purchase journey progressions — generally from low-price to high-price categories with lateral moves between adjacent price tiers.

**Average prices are from the real dataset** (Unit_Price mean per category) — do not change them. They are also in `src/constants.py` as `CATEGORY_AVG_PRICES`.

---

## New helper functions — replace all three

```python
def heuristic(current_category, target_category, user_median_spend):
    """
    h(n): estimated cost from current to target.
    = absolute price gap between user's spend and target's avg price.
    Admissible — never overestimates true remaining cost.
    """
    target_price = PRODUCT_GRAPH[target_category]["avg_price"]
    return abs(target_price - user_median_spend)


def edge_cost(to_category, user_median_spend):
    """
    Step cost to enter a category node.
    = absolute price gap between user's spend and next category's avg price.
    """
    return abs(PRODUCT_GRAPH[to_category]["avg_price"] - user_median_spend)


def get_start_category(user_profile):
    """
    Find the product category closest to user's current spend level.
    This is the A* starting node.
    """
    user_spend = user_profile["median_spend"]
    return min(
        PRODUCT_GRAPH.keys(),
        key=lambda c: abs(PRODUCT_GRAPH[c]["avg_price"] - user_spend)
    )
```

---

## Admissibility proof (update your markdown cell)

The heuristic h(n) = abs(user_spend - target_avg_price) is admissible because:
- To reach the target category, the user must traverse at least one edge whose cost includes the price gap to the target
- h(n) measures only the direct price gap — it never accounts for intermediate steps, so it never overestimates
- Therefore h(n) ≤ true remaining cost for every node n ✓

---

## User profile schema (from src/constants.py)

```python
user_profile = {
    "age":            int,       # 18–75
    "gender":         str,       # "Female" | "Male" | "Other"
    "city":           str,       # one of 10 Turkish cities
    "median_spend":   float,     # user's median transaction spend — PRIMARY SIGNAL
    "device_type":    str,       # "Mobile" | "Desktop" | "Tablet"
    "payment_method": str,       # one of 5 payment types
    "age_group":      str,       # derived: "18-25" | "26-35" | "36-45" | "46-55" | "56+"
    "price_range":    str,       # derived: "Low" | "Mid-Low" | "Mid-High" | "High"
}
```

**The only field you need:** `user_profile["median_spend"]`

---

## Sample profiles to test against

| Profile | median_spend | Expected start category | Expected path to Electronics |
|---|---|---|---|
| budget_reader | 55.0 | Books | Books → Food → Beauty → Toys → Fashion → Sports → Home & Garden → Electronics |
| beauty_shopper | 180.0 | Beauty | Beauty → Fashion → Sports → Home & Garden → Electronics |
| fashion_enthusiast | 380.0 | Fashion | Fashion → Sports → Home & Garden → Electronics |
| sports_buyer | 650.0 | Sports | Sports → Home & Garden → Electronics |
| tech_spender | 1500.0 | Electronics | [Electronics] (already there) |

---

## Imports

```python
import sys, os
sys.path.append(os.path.abspath('..'))

from src.constants import (
    PRODUCT_CATEGORIES,
    CATEGORY_AVG_PRICES,
    SAMPLE_PROFILES,
    build_user_profile,
)
```

---

## What stays the same
- A* core algorithm (heapq priority queue, visited set, path tracking, f=g+h)
- Function structure of `find_product_path()`
- All markdown explanation cells — just update domain-specific wording
- Graph visualisation code — update node labels and colours only

## What to rename
- `CAREER_GRAPH` → `PRODUCT_GRAPH`
- `get_start_career()` → `get_start_category()`
- `find_career_path()` → `find_product_path()`
- `display_path_detailed()` → update labels (career → category, skills → price gap)

## Do NOT touch
- `src/constants.py` — read only
- `notebooks/ml_model.ipynb` — Member 3's file
- Any file outside `notebooks/career_recommender.ipynb` and your section
