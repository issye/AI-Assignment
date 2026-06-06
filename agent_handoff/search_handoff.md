# Agent Handoff — A* Search Module (Member 1)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/search`
**Deadline:** 27 June 2026, 7 PM

---

## What Changed and Why

The project has pivoted to the **suvroo dataset** (Indian market, ₹ currency). Two things affect your module:

1. **Categories reduced from 8 to 6** — Food and Toys are no longer in the system. Your graph has 6 nodes instead of 8.

2. **Price-gap heuristic is no longer viable** — suvroo category avg prices are nearly flat (all ~₹2,500). A price-based heuristic would produce near-arbitrary paths. Instead, use **co-purchase frequency** as edge costs: how often customers bought from both category A and B in the same purchase session. High co-purchase = low cost = categories are "close."

Everything else stays the same:
- A* algorithm logic unchanged
- `find_product_path()` signature unchanged
- Output format unchanged (list of category strings)

---

## Updated Constants (pull from `src/constants.py`)

```python
from src.constants import (
    PRODUCT_CATEGORIES,    # 6 categories
    CATEGORY_AVG_PRICES,   # reference only — NOT used for edge costs anymore
    SPEND_THRESHOLDS,      # ₹ quartile-based tiers
    CITIES, URBAN_CITIES,  # Indian cities
    SAMPLE_PROFILES,
    build_user_profile,
    get_price_range,
)
```

### PRODUCT_CATEGORIES (6 nodes in your graph)
```python
["Books", "Beauty", "Electronics", "Fashion", "Sports", "Home & Garden"]
```

### SPEND_THRESHOLDS (updated ₹ values)
```
Low:      Avg_Order_Value < ₹1,636
Mid-Low:  ₹1,636 – ₹2,740
Mid-High: ₹2,740 – ₹3,879
High:     ≥ ₹3,879
```

---

## New Heuristic: Co-Purchase Frequency

### How to compute the co-purchase matrix

From `data/suvroo/customer_data_collection.csv`, parse each customer's `Purchase_History` and map items to categories:

```python
ITEM_TO_CAT = {
    'Biography': 'Books', 'Non-fiction': 'Books', 'Fiction': 'Books', 'Comics': 'Books',
    'Moisturizer': 'Beauty', 'Lipstick': 'Beauty', 'Foundation': 'Beauty', 'Perfume': 'Beauty',
    'Smartphone': 'Electronics', 'Headphones': 'Electronics', 'Laptop': 'Electronics', 'Smartwatch': 'Electronics',
    'T-shirt': 'Fashion', 'Jeans': 'Fashion', 'Jacket': 'Fashion', 'Shoes': 'Fashion',
    'Resistance Bands': 'Sports', 'Dumbbells': 'Sports', 'Yoga Mat': 'Sports', 'Treadmill': 'Sports',
    'Wall Art': 'Home & Garden', 'Curtains': 'Home & Garden', 'Cushions': 'Home & Garden', 'Lamp': 'Home & Garden',
}
```

For each customer, find all category pairs in their purchase history. Count how many customers share each pair:

```python
import pandas as pd
import re
from itertools import combinations
from collections import defaultdict

df = pd.read_csv('data/suvroo/customer_data_collection.csv')

co_counts = defaultdict(int)

for _, row in df.iterrows():
    items = re.findall(r"'([^']+)'", str(row['Purchase_History']))
    cats = list(set(ITEM_TO_CAT.get(i) for i in items if ITEM_TO_CAT.get(i)))
    for a, b in combinations(sorted(cats), 2):
        co_counts[(a, b)] += 1

total_customers = len(df)  # 10,000

# Edge cost: high co-purchase = low cost (categories are "close")
# Low co-purchase = high cost (categories are "far")
CO_PURCHASE_COSTS = {
    (a, b): round(1 - (count / total_customers), 4)
    for (a, b), count in co_counts.items()
}
```

**Hardcode the resulting matrix** — run once, copy the values into your module as a constant. This avoids loading the CSV at inference time.

### Edge cost formula
```
cost(A → B) = 1 - (co_purchase_count(A, B) / 10000)
```
- Categories bought together by 7,000 customers → cost = 0.30 (very close)
- Categories bought together by 800 customers → cost = 0.92 (far)

### Heuristic function
```python
def heuristic(current_category: str, target_category: str) -> float:
    """
    Admissible heuristic: co-purchase cost between current and target.
    Never overestimates true path cost through the graph.
    """
    key = tuple(sorted([current_category, target_category]))
    return CO_PURCHASE_COSTS.get(key, 1.0)  # default 1.0 if pair not found
```

**Admissibility proof:** The heuristic returns the direct co-purchase cost between two categories. The true shortest path cost through the graph must be ≥ direct edge cost (triangle inequality holds because all edge costs are non-negative). Therefore h(n) never overestimates — the heuristic is admissible.

---

## New Start Node Logic

Old: start = category closest to user's median_spend by price.
New: start = most recently browsed category from user profile.

```python
def get_start_category(user_profile: dict) -> str:
    """
    Determine A* start node from user's browsing history.
    Falls back to spend-tier default if no browsing data available.
    """
    # user_profile may optionally contain browsing_history
    browsing = user_profile.get('browsing_history', [])
    if browsing:
        return browsing[0]  # first browsed category as entry point

    # Fallback: map spend tier to a default starting category
    tier = user_profile['price_range']
    tier_defaults = {
        'Low':      'Books',
        'Mid-Low':  'Fashion',
        'Mid-High': 'Sports',
        'High':     'Electronics',
    }
    return tier_defaults.get(tier, 'Books')
```

---

## Updated PRODUCT_GRAPH (6 nodes)

```python
# Edges connect all 6 categories as a fully connected graph.
# Edge costs come from CO_PURCHASE_COSTS (hardcoded from suvroo data).
# Node 'avg_price' kept for reference — not used for routing.

PRODUCT_GRAPH = {
    "Books":         {"avg_price": 2524, "neighbors": ["Beauty", "Electronics", "Fashion", "Sports", "Home & Garden"]},
    "Beauty":        {"avg_price": 2501, "neighbors": ["Books", "Electronics", "Fashion", "Sports", "Home & Garden"]},
    "Electronics":   {"avg_price": 2548, "neighbors": ["Books", "Beauty", "Fashion", "Sports", "Home & Garden"]},
    "Fashion":       {"avg_price": 2618, "neighbors": ["Books", "Beauty", "Electronics", "Sports", "Home & Garden"]},
    "Sports":        {"avg_price": 2578, "neighbors": ["Books", "Beauty", "Electronics", "Fashion", "Home & Garden"]},
    "Home & Garden": {"avg_price": 2549, "neighbors": ["Books", "Beauty", "Electronics", "Fashion", "Sports"]},
}
```

Since all 6 categories are connected to all others, A* will find direct paths for closely related categories (low co-purchase cost) and longer indirect paths for distant ones.

---

## Unchanged: Function Signature

```python
def find_product_path(user_profile: dict, target_category: str) -> list[str]:
    """
    Find the A* path from the user's entry category to the target category.

    Parameters
    ----------
    user_profile    : dict — from build_user_profile() in src/constants.py
    target_category : str  — one of PRODUCT_CATEGORIES

    Returns
    -------
    list[str] — ordered category steps from start to target (inclusive)
               Returns [] if no path found.

    Example
    -------
    find_product_path(SAMPLE_PROFILES['tech_spender'], 'Electronics')
    → ['Fashion', 'Sports', 'Electronics']  # path via co-purchase affinity
    """
```

---

## Expected Paths for SAMPLE_PROFILES

These are illustrative — actual paths depend on your computed co-purchase matrix:

| Profile | Spend | Likely start | Target | Expected path concept |
|---|---|---|---|---|
| budget_browser | ₹800 | Books | Electronics | Long path via affinity |
| beauty_enthusiast | ₹1,800 | Beauty | Electronics | Via Fashion or Sports |
| fashion_fan | ₹2,500 | Fashion | Electronics | Short — Fashion→Electronics likely high co-purchase |
| fitness_guy | ₹3,200 | Sports | Electronics | Short — Sports→Electronics likely high co-purchase |
| tech_spender | ₹4,500 | Electronics | Electronics | [Electronics] — already there |

---

## What To Do

1. Pull `src/constants.py` from `main` (Member 3 has already updated it)
2. Run the co-purchase matrix calculation once from suvroo data
3. Hardcode `CO_PURCHASE_COSTS` dict into your module
4. Replace price-based `heuristic()` and `edge_cost()` with co-purchase versions
5. Replace `get_start_category()` with browsing-history version
6. Update graph from 8 nodes to 6 nodes
7. Keep `find_product_path()` signature identical — Member 4 calls this
8. Update markdown cells in your notebook section to explain the new heuristic

## Rules
- Never change `find_product_path()` signature
- All category names come from `src/constants.py PRODUCT_CATEGORIES`
- Do not push — user pushes manually
- PRs must be draft: `gh pr create --draft`
