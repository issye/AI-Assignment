# Agent Handoff — Search Module (Member 1)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/search`
**Deadline:** 27 June 2026, 7 PM

---

## What Changed and Why

The project has **fully pivoted** from A* category pathfinding to a **BFS-based category
reachability search + popularity-based cold-start fallback**.

| What changed | Old (suvroo) | New (Online Retail) |
|---|---|---|
| Algorithm | A* pathfinding | BFS on category similarity graph |
| Edge costs | Co-purchase frequency (synthetic) | Cosine similarity from real CF matrix |
| Graph nodes | 6 categories | 8 categories |
| Start node | browsing_history[0] | favourite_category |
| Output | Category path list | Reachable category shortlist |
| Cold-start | Not handled | `find_popular_categories()` (new function) |

---

## Your Module's New Role

The search module sits **between the rules engine and the CF model**:

```
apply_rules()               → eligible: broad set (up to 8 categories)
        ↓
find_reachable_categories() → shortlist: categories reachable from user's buying pattern
        ↓
predict_product()           → CF scores the shortlist
```

For new users with no purchase history, CF cannot function. You own that path entirely:

```
apply_rules()               → eligible: ["Home Decor", "Seasonal & Gifts", "Kitchen & Dining"]
        ↓
find_popular_categories()   → ordered by global transaction count
        ↓
(CF is bypassed — Member 4 handles routing)
```

---

## Two Functions to Implement

### Function 1: `find_reachable_categories(user_profile, eligible, max_hops=2) -> list[str]`

**For returning users (purchase_history is non-empty).**

BFS on the 8×8 category similarity graph starting from `favourite_category`.
Returns the subset of `eligible` that is reachable within `max_hops` hops above a
similarity threshold. Ordered by graph proximity (closest categories first).

```python
def find_reachable_categories(
    user_profile: dict,
    eligible:     list,
    max_hops:     int = 2,
    threshold:    float = 0.40,
) -> list[str]:
    """
    BFS on category similarity graph.

    Parameters
    ----------
    user_profile : dict  — from build_user_profile() in src/constants.py
    eligible     : list  — output of apply_rules(), subset of PRODUCT_CATEGORIES
    max_hops     : int   — maximum BFS depth (default 2)
    threshold    : float — minimum similarity to traverse an edge (default 0.40)

    Returns
    -------
    list[str] — subset of eligible, ordered by BFS discovery (closest first).
                If favourite_category is None, returns eligible as-is.
    """
    import pickle

    start = user_profile.get('favourite_category')
    if not start:
        return eligible   # no start node: return eligible unchanged

    cat_sim = pickle.load(open('models/category_similarity.pkl', 'rb'))

    visited = {start}
    queue   = [(start, 0)]
    result  = [start] if start in eligible else []

    while queue:
        node, hops = queue.pop(0)
        if hops >= max_hops:
            continue
        neighbours = cat_sim[node].drop(node).sort_values(ascending=False)
        for neighbour, sim in neighbours.items():
            if sim < threshold:
                break   # sorted descending — no point continuing
            if neighbour not in visited:
                visited.add(neighbour)
                if neighbour in eligible:
                    result.append(neighbour)
                queue.append((neighbour, hops + 1))

    return result
```

**Worked example — home_decorator:**
```
eligible      = all 8 categories (Frequent buyer, Rule 9)
favourite     = "Home Decor"
Hop 1: Kitchen & Dining (~0.81), Seasonal & Gifts (~0.74),
        Stationery & Craft (~0.61), Garden & Outdoor (~0.44)
Hop 2: Toys & Games (via Kitchen & Dining),
        Food & Confectionery (via Seasonal & Gifts)
Fashion & Accessories has low similarity to Home Decor → pruned

→ returns 7 categories (Fashion & Accessories excluded)
```

**Worked example — craft_lover:**
```
eligible      = ["Stationery & Craft", + 1 adjacent] (Rule 7, single-cat buyer)
favourite     = "Stationery & Craft"
All eligible categories reachable in hop 1 → no pruning
→ returns all eligible (already tight from rules)
```

---

### Function 2: `find_popular_categories(eligible, price_range, top_n=3) -> list[tuple[str, int]]`

**For new users (purchase_history is empty). Cold-start fallback.**

Returns the most globally purchased categories within the eligible set,
ordered by total unique buyers (descending).

```python
def find_popular_categories(
    eligible:    list,
    price_range: str,
    top_n:       int = 3,
) -> list[tuple[str, int]]:
    """
    Popularity-based category ranking for cold-start users.

    Parameters
    ----------
    eligible    : list — output of apply_rules()
    price_range : str  — "Low"|"Mid-Low"|"Mid-High"|"High"
    top_n       : int  — number of categories to return

    Returns
    -------
    list[tuple[str, int]] — [(category, unique_buyer_count), ...]
                             ordered by unique_buyer_count descending
    """
    import pickle

    catalogue = pickle.load(open('models/product_catalogue.pkl', 'rb'))
    counts = (
        catalogue[catalogue['category'].isin(eligible)]
        .groupby('category')['popularity_rank']
        .sum()
        .sort_values(ascending=False)
    )
    return [(cat, int(count)) for cat, count in counts.head(top_n).items()]
```

**Worked example — new_customer:**
```
eligible    = ["Home Decor", "Seasonal & Gifts", "Kitchen & Dining"]
→ [("Home Decor", 8420), ("Seasonal & Gifts", 5310), ("Kitchen & Dining", 4190)]
  (exact counts from training data — Home Decor dominant at ~56% of products)
```

---

## Artefacts to Load

| File | Used by | Contains |
|---|---|---|
| `models/category_similarity.pkl` | `find_reachable_categories` | 8×8 pandas DataFrame, category→category cosine similarity |
| `models/product_catalogue.pkl` | `find_popular_categories` | DataFrame: StockCode, Description, category, avg_price, popularity_rank |

---

## Constants to Import

```python
from src.constants import PRODUCT_CATEGORIES
```

---

## Public Interface

```python
# For returning users — called when purchase_history is non-empty
find_reachable_categories(
    user_profile: dict,
    eligible:     list,
    max_hops:     int   = 2,
    threshold:    float = 0.40,
) -> list[str]

# For new users — called when purchase_history is empty
find_popular_categories(
    eligible:    list,
    price_range: str,
    top_n:       int = 3,
) -> list[tuple[str, int]]
```

---

## Session 3 Note (ML update)

`predict_product()` now works for cold-start users too — it returns RF probability scores
for ALL users including those with empty `purchase_history`. However, your
`find_popular_categories()` is still needed because:
1. The integration notebook (`recommend()`) still routes cold-start users through the popular
   path for `recommendation_type = "popular"` (which uses integer buyer counts, not probabilities)
2. `recommend_products()` falls back to `popularity_rank` for cold-start product selection,
   which depends on the data your function surfaces

No changes needed to your function signatures or implementation.

---

## What To Do

1. Pull `src/constants.py` and `models/*.pkl` from `main` (artefacts saved by Member 3)
2. Create `src/search_module.py` and implement both functions
3. Test `find_reachable_categories()` on `home_decorator` and `craft_lover` SAMPLE_PROFILES
4. Test `find_popular_categories()` on `new_customer` SAMPLE_PROFILE
5. Verify: `find_reachable_categories` always returns a **subset** of `eligible`

## Rules

- Both function signatures are fixed — Member 4 calls them directly
- `find_reachable_categories` must never return categories outside `eligible`
- `find_popular_categories` scores must be integers (not floats)
- Never hardcode category names — use `PRODUCT_CATEGORIES` from `src/constants.py`
- Do not push — user pushes manually
- PRs must be draft: `gh pr create --draft`
