# Agent Handoff — Search Module (Member 1)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/search`
**Deadline:** 27 June 2026, 7 PM

---

## What This Module Does

The search module sits between the rules engine and the ML model. Its job is to take a broad list of eligible categories and narrow it down to ones that are actually relevant to a specific customer — using the purchase similarity structure of the data, not just rules or ML predictions.

It also handles the cold-start path entirely: when a customer has no purchase history, the ML model cannot score anything meaningful, so the search module ranks categories by global popularity instead.

---

## Why This Module Exists

The rules engine can return up to all 8 categories for a frequent buyer. Passing all 8 to the ML model is valid, but it means the model spends effort scoring categories that have nothing to do with the customer.

For example, a customer who has only ever bought Home Decor items is unlikely to suddenly want Fashion & Accessories. The search module prunes those irrelevant categories *before* the ML model runs, using a similarity graph built from real co-purchase patterns in the data.

This is not a rule ("if Home Decor, remove Fashion") — it is a graph traversal that uses the actual similarity structure learned from 397,884 transactions.

---

## How This Module Fits Into the Full Pipeline

```
User Profile
     │
     ▼
Rules Engine (Member 2)
  apply_rules(profile) → eligible: up to 8 categories
     │
     ▼
Search Module (Member 1)  ◄─── YOU ARE HERE
     │
     ├── purchase_history is NON-EMPTY?
     │     find_reachable_categories(profile, eligible)
     │     → shortlist: categories reachable from customer's buying pattern
     │     → passed to predict_product() as candidates
     │
     └── purchase_history is EMPTY?
           find_popular_categories(eligible, price_range)
           → ranked by global buyer count
           → ML model is bypassed entirely; this IS the recommendation
     │
     ▼
ML Module (Member 3)
  predict_product(profile, candidates=shortlist)
     │
     ▼
Integration (Member 4)
  recommend(profile) → final output dict
```

Member 4 handles the routing decision (which path to take). You only implement the two functions.

---

## The Category Similarity Graph

The ALS model (Member 3) produces a 3,665 × 3,665 item similarity matrix. After training, this is aggregated to an 8 × 8 category similarity matrix — saved as `models/category_similarity.pkl`.

Each cell in this matrix is the average cosine similarity between all products in category A and all products in category B, computed from ALS latent factors. It reflects which categories tend to be bought together across the full dataset.

This matrix is your graph. Categories are nodes. An edge exists between two categories if their similarity exceeds the threshold (default 0.40).

Example similarity values (approximate, based on the dataset):

| | Home Decor | Kitchen & Dining | Seasonal & Gifts | Fashion & Accessories |
|---|---|---|---|---|
| **Home Decor** | 1.00 | 0.81 | 0.74 | 0.22 |
| **Kitchen & Dining** | 0.81 | 1.00 | 0.69 | 0.18 |
| **Seasonal & Gifts** | 0.74 | 0.69 | 1.00 | 0.21 |
| **Fashion & Accessories** | 0.22 | 0.18 | 0.21 | 1.00 |

Fashion & Accessories sits far from the home/kitchen/seasonal cluster. A customer whose favourite is Home Decor will not reach it within 2 hops above the 0.40 threshold.

---

## Two Functions to Implement

### Function 1 — `find_reachable_categories()`

For returning users with purchase history.

```python
def find_reachable_categories(
    user_profile: dict,
    eligible:     list,
    max_hops:     int   = 2,
    threshold:    float = 0.40,
) -> list[str]:
```

**What it does:**
1. Starts at `user_profile['favourite_category']` (their most-purchased category)
2. Runs BFS on the category similarity graph
3. At each step, follows edges where similarity > threshold
4. Collects categories that are reachable within `max_hops` AND are in `eligible`
5. Returns them in order of discovery (closest categories first)

**If `favourite_category` is None:** return `eligible` unchanged — no graph to start from.

**The result must be a subset of `eligible`** — never return a category not in eligible.

```python
def find_reachable_categories(user_profile, eligible, max_hops=2, threshold=0.40):
    import pickle
    start = user_profile.get('favourite_category')
    if not start:
        return eligible

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
                break
            if neighbour not in visited:
                visited.add(neighbour)
                if neighbour in eligible:
                    result.append(neighbour)
                queue.append((neighbour, hops + 1))

    return result
```

---

### Function 2 — `find_popular_categories()`

For new users with no purchase history (cold-start path).

```python
def find_popular_categories(
    eligible:    list,
    price_range: str,
    top_n:       int = 3,
) -> list[tuple[str, int]]:
```

**What it does:**
1. Filters `product_catalogue` to only products in `eligible` categories
2. Groups by category, sums `popularity_rank` (unique buyer count per product)
3. Returns the top `top_n` categories ordered by total buyer count

Scores must be **integers** (unique buyer counts), not floats.

```python
def find_popular_categories(eligible, price_range, top_n=3):
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

---

## Worked Example — home_decorator

```
Profile:
  favourite_category : "Home Decor"
  customer_segment   : "Frequent"
  purchase_history   : 4 items

Step 1 — Rules engine returns:
  eligible = all 8 categories  (Rule 9: Frequent buyer gets full catalogue)

Step 2 — find_reachable_categories(profile, eligible)

  START: Home Decor

  Hop 1 (direct neighbours above 0.40):
    Kitchen & Dining    sim=0.81  ✓ in eligible → add
    Seasonal & Gifts    sim=0.74  ✓ in eligible → add
    Stationery & Craft  sim=0.61  ✓ in eligible → add
    Garden & Outdoor    sim=0.44  ✓ in eligible → add
    Fashion & Accessories sim=0.22  ✗ below threshold → skip

  Hop 2 (neighbours of hop 1 not yet visited):
    Toys & Games        reachable via Kitchen & Dining → add
    Food & Confectionery reachable via Seasonal & Gifts → add

  Fashion & Accessories never reached within 2 hops → pruned

  RESULT: [Home Decor, Kitchen & Dining, Seasonal & Gifts,
           Stationery & Craft, Garden & Outdoor,
           Toys & Games, Food & Confectionery]
  (7 of 8 — Fashion & Accessories removed)

Step 3 — predict_product(profile, candidates=shortlist)
  ML model scores only these 7 categories
```

## Worked Example — new_customer

```
Profile:
  purchase_history   : []
  customer_segment   : "New"
  price_range        : "Low"

Step 1 — Rules engine returns:
  eligible = ["Home Decor", "Seasonal & Gifts", "Kitchen & Dining"]
  (Rule 10: new customers get only 3 most popular)

Step 2 — find_popular_categories(eligible, "Low", top_n=3)
  Count unique buyers per category across eligible products:
  → [("Home Decor", 8420), ("Seasonal & Gifts", 5310), ("Kitchen & Dining", 4190)]

Step 3 — Integration routes to cold-start path
  ML model bypassed entirely
  Popularity becomes the recommendation directly
```

---

## Artefacts to Load

| File | Used by | Contents |
|---|---|---|
| `models/category_similarity.pkl` | `find_reachable_categories` | 8×8 pandas DataFrame, category→category cosine similarity |
| `models/product_catalogue.pkl` | `find_popular_categories` | DataFrame with StockCode, Description, category, avg_price, popularity_rank |

Both files are generated by running `notebooks/ml_model.ipynb`. They will not exist until Member 3 runs their notebook.

---

## Constants to Import

```python
from src.constants import PRODUCT_CATEGORIES
```

Never hardcode category names. Always use `PRODUCT_CATEGORIES` from constants.

---

## What To Do

1. Pull `main` to get `src/constants.py` and wait for `models/*.pkl` from Member 3
2. Create `src/search_module.py` and implement both functions
3. Test `find_reachable_categories()` using `home_decorator` SAMPLE_PROFILE — should return 7 categories excluding Fashion & Accessories
4. Test `find_popular_categories()` using `new_customer` SAMPLE_PROFILE — should return 3 categories as integers
5. Verify that `find_reachable_categories` never returns a category outside `eligible`
6. Create draft PR: `gh pr create --draft`

## Rules

- Both function signatures are frozen — Member 4 calls them directly
- `find_reachable_categories` must only return categories that are in `eligible`
- `find_popular_categories` scores must be integers, not floats
- Never hardcode category names
- Do not push — user pushes manually
