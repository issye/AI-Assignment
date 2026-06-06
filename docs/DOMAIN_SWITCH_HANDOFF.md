# Domain Switch Handoff — Career → Smart Product Recommendation

**Prepared by:** Member 3 (ML)  
**Branch:** `feature/ml`  
**Date:** 2026-06-06  
**Status:** Action required from all members before integration

---

## What Changed and Why

The team has agreed to switch the application domain from **Career Recommendation** to **Smart Product Recommendation**. This decision was made after discovering that the original career dataset had randomly assigned labels with no learnable signal (all models performed at 8.3% — random chance for 12 classes).

The new dataset (`ecommerce_customer_behavior_dataset_v2.csv`) is a real e-commerce transaction log with 17,049 orders from 5,000 unique customers. After aggregation into user profiles, the spend signal strongly predicts product category preference (MI = 0.20).

---

## New Architecture: Reachability-First

```
User Profile
     │
     ▼
Rules Engine  → filters eligible product categories (Logic)
     │
     ▼
A* Search     → ranks eligible categories by price reachability
     │
     ▼
ML Model      → re-ranks A*'s shortlist by learned preference probability
     │
     ▼
Integration   → Top-3 recommendations + purchase journey roadmap
```

**Why this order?** The dataset's demographic features (Age, Gender) have weak direct signal for category prediction — they work indirectly through spending behaviour. By running A* first to rank by price reachability, ML only needs to re-rank 3–5 candidates rather than classify across all 8 — a much more tractable task with limited features.

---

## New Dataset

**File:** `data/ecommerce_customer_behavior_dataset_v2.csv`  
**Raw shape:** 17,049 rows × 18 columns  
**After aggregation:** 5,000 user profiles × 13 features  

**Key columns:**

| Column | Type | Signal strength |
|---|---|---|
| `Total_Amount` / `median_spend` | Numeric | ★★★ Strong (MI=0.20) |
| `Age` | Numeric | ★ Weak direct, indirect via spend |
| `Gender` | Categorical | ★ Weak direct, indirect via spend |
| `City` | Categorical | ★ Weak |
| `Product_Category` | Categorical — **target** | — |

**Average price per category (from raw data — use in A* graph):**

| Category | Avg Price (RM) |
|---|---|
| Electronics | ~1,800 |
| Home & Garden | ~900 |
| Sports | ~500 |
| Fashion | ~400 |
| Toys | ~350 |
| Beauty | ~200 |
| Food | ~100 |
| Books | ~80 |

---

## Updated `src/constants.py`

`constants.py` has been fully replaced. **All members must update their imports.**

### New constants
```python
from src.constants import (
    PRODUCT_CATEGORIES,   # 8 product categories — target labels
    AGE_GROUPS,           # 5 age bands
    GENDERS,              # Female, Male, Other
    CITIES,               # 10 Turkish cities
    PRICE_RANGES,         # Low, Mid-Low, Mid-High, High
    DEVICE_TYPES,         # Mobile, Desktop, Tablet
    PAYMENT_METHODS,      # 5 payment types
    CATEGORY_AVG_PRICES,  # dict: category → avg price (for A* graph)
    SAMPLE_PROFILES,      # 5 test profiles
    build_user_profile,   # profile constructor
)
```

### New user profile schema
```python
user_profile = build_user_profile(
    age           = 28,
    gender        = "Female",
    city          = "Istanbul",
    median_spend  = 350.0,      # user's typical spend per transaction
    device_type   = "Mobile",
    payment_method= "Credit Card",
)
```

### New function signatures (unchanged interface pattern)
```python
# Member 2 — rules engine
apply_rules(user_profile)
# returns: list[str] — eligible categories from PRODUCT_CATEGORIES

# Member 1 — A* search
find_product_path(user_profile, target_category)
# returns: list[str] — ordered category steps, or [] if no path

# Member 3 — ML model
predict_product(user_profile, candidates=None)
# returns: list[tuple[str, float]] — [(category, confidence), ...] descending
# if candidates provided, only scores those categories
```

---

## What Each Member Needs to Do

---

### Member 1 — A* Search (`feature/search`)

**Effort: Medium — rebuild graph and helper functions, algorithm core unchanged**

#### 1. Replace `CAREER_GRAPH` with `PRODUCT_GRAPH`

```python
PRODUCT_GRAPH = {
    "Books": {
        "avg_price": 80,
        "neighbors": ["Food", "Beauty", "Toys"],
    },
    "Food": {
        "avg_price": 100,
        "neighbors": ["Books", "Beauty", "Toys"],
    },
    "Beauty": {
        "avg_price": 200,
        "neighbors": ["Fashion", "Food", "Books"],
    },
    "Toys": {
        "avg_price": 350,
        "neighbors": ["Beauty", "Fashion", "Sports"],
    },
    "Fashion": {
        "avg_price": 400,
        "neighbors": ["Beauty", "Sports", "Home & Garden"],
    },
    "Sports": {
        "avg_price": 500,
        "neighbors": ["Fashion", "Home & Garden", "Electronics"],
    },
    "Home & Garden": {
        "avg_price": 900,
        "neighbors": ["Sports", "Electronics"],
    },
    "Electronics": {
        "avg_price": 1800,
        "neighbors": ["Home & Garden", "Sports"],
    },
}
```

Edges represent natural purchase journey progression — generally from low-price to high-price categories, with lateral moves between similar categories.

#### 2. Replace helper functions

```python
def heuristic(current_category, target_category, user_median_spend):
    """h(n): price gap between user's spend and target category's avg price."""
    target_price = PRODUCT_GRAPH[target_category]["avg_price"]
    return abs(target_price - user_median_spend)

def edge_cost(to_category, user_median_spend):
    """Step cost: price gap to enter next category."""
    return abs(PRODUCT_GRAPH[to_category]["avg_price"] - user_median_spend)

def get_start_category(user_profile):
    """Find category closest to user's current spend level."""
    user_spend = user_profile["median_spend"]
    return min(
        PRODUCT_GRAPH.keys(),
        key=lambda c: abs(PRODUCT_GRAPH[c]["avg_price"] - user_spend)
    )
```

#### 3. Rename function and update signature

```python
def find_product_path(user_profile, target_category):
    """
    A* search — finds optimal category progression path.
    Returns list[str] — ordered steps from start to target.
    Returns [] if no path exists.
    """
```

#### 4. How it runs in the waterfall

A* now runs **once per eligible category** from the rules engine (not once toward a single target). Member 4 calls it like this:

```python
eligible = apply_rules(user_profile)           # e.g. ["Beauty", "Electronics", "Sports"]
paths = {cat: find_product_path(user_profile, cat) for cat in eligible}
# paths = {"Beauty": ["Books","Beauty"], "Electronics": ["Books","Fashion","Electronics"], ...}
# sorted by total cost → passed to predict_product as candidates
```

#### 5. What stays the same
- A* core algorithm (priority queue, visited set, path tracking)
- Admissibility proof (price gap never overestimates — same logic)
- Markdown explanation cells (just update domain-specific wording)

---

### Member 2 — Rules Engine (`feature/rules`)

**Effort: Medium — rebuild rules for new domain, inference mechanism unchanged**

#### Key finding from data analysis
Age and Gender have near-zero direct signal (MI < 0.01) for category prediction. They work **indirectly through spending behaviour**. Rules based on these are still valid commonsense logic but should be secondary to spend-based rules.

#### Primary rules (spend-based — strongest signal)
```python
# Low spenders → affordable categories
IF median_spend < 150  → eligible: Books, Food, Beauty

# Mid spenders → mid-range categories  
IF 150 <= median_spend < 500 → eligible: Beauty, Fashion, Toys, Sports

# High spenders → premium categories
IF median_spend >= 500 → eligible: Electronics, Home & Garden, Sports
```

#### Secondary rules (demographic — commonsense logic)
```python
# Age-based
IF age < 25    → add eligible: Toys, Fashion
IF age >= 45   → add eligible: Home & Garden, Books
IF age >= 60   → remove eligible: Electronics (unless high spender)

# Gender-based
IF gender = Female → add eligible: Beauty, Fashion
IF gender = Male   → add eligible: Sports, Electronics

# Location-based
IF city IN [Istanbul, Ankara, Izmir]  → add eligible: Electronics, Fashion  (urban)
IF city NOT IN [Istanbul, Ankara, Izmir] → add eligible: Food, Home & Garden (regional)

# Behavioural
IF is_returning = True AND avg_rating >= 4 → eligible for all categories
IF device_type = Mobile → add eligible: Fashion, Beauty
```

#### Target: 8–10 well-defined rules for excellent rubric score

---

### Member 4 — Integration (`feature/integration`)

**Effort: Low-Medium — rewire pipeline for new domain and waterfall order**

#### Updated pipeline call
```python
from src.rules_engine import apply_rules
from src.search_module import find_product_path
from src.ml_model import predict_product
from src.constants import PRODUCT_CATEGORIES

def recommend(user_profile):
    # Step 1: Rules — filter eligible categories
    eligible = apply_rules(user_profile)
    if not eligible:
        eligible = PRODUCT_CATEGORIES  # fallback: all categories

    # Step 2: A* — rank by reachability, compute paths
    paths = {cat: find_product_path(user_profile, cat) for cat in eligible}
    sorted_by_cost = sorted(eligible, key=lambda c: sum_path_cost(paths[c], user_profile))

    # Step 3: ML — re-rank by predicted preference
    predictions = predict_product(user_profile, candidates=sorted_by_cost)

    # Step 4: Output
    top3 = predictions[:3]
    top1_category = top3[0][0]
    roadmap = paths[top1_category]

    return {
        "top_3": top3,
        "roadmap": roadmap,
    }
```

#### Handle missing target_category
Since users don't specify a target (it's a recommender system), `find_product_path()` is called with ML's output — not user input. The integration layer passes ML's top-1 as the target for the final roadmap display.

---

## Submission Checklist Updates

- [ ] `src/constants.py` updated with product domain constants
- [ ] `notebooks/ml_model.ipynb` rebuilt for product domain
- [ ] Member 1 rebuilds A* graph and helper functions
- [ ] Member 2 rebuilds rules for spend + demographic logic
- [ ] Member 4 rewires integration pipeline
- [ ] All SAMPLE_PROFILES updated in constants.py
- [ ] Notebook runs clean: Kernel → Restart & Run All, zero errors
- [ ] predict_product(profile, candidates=[...]) tested and working
- [ ] Deadline: 27 June 2026, 7 PM
