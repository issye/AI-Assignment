# Agent Handoff — Integration & System Design (Member 4)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/integration`
**Deadline:** 27 June 2026, 7 PM

---

## What Changed and Why

The system now recommends **specific named products** (e.g. "Smartphone", "Yoga Mat"), not just categories. Your integration layer gains one extra step — calling `recommend_products()` after `predict_product()` — and your output schema has one new key.

Everything else is the same: the pipeline order (Rules → A* → ML → Output) is unchanged, and all three function signatures you call are backward compatible.

---

## Updated Pipeline

```
User Profile
    │  build_user_profile(age, gender, city, median_spend, ...)
    ▼
apply_rules(user_profile)
    │  → eligible: list[str]   e.g. ["Beauty", "Fashion", "Sports", "Electronics"]
    ▼
find_product_path(user_profile, target_category)  [called per eligible category]
    │  → paths: dict[str, list[str]]
    │    e.g. {"Electronics": ["Fashion", "Sports", "Electronics"],
    │          "Sports": ["Fashion", "Sports"], ...}
    ▼
predict_product(user_profile, candidates=list(paths.keys()))
    │  → categories: list[tuple[str, float]]
    │    e.g. [("Electronics", 0.84), ("Sports", 0.61), ("Fashion", 0.42)]
    ▼
recommend_products(user_profile, category=top_category, top_n=3)   ← NEW STEP
    │  → products: list[tuple[str, float]]
    │    e.g. [("Smartphone", 0.91), ("Laptop", 0.87), ("Headphones", 0.73)]
    ▼
Output dict
```

---

## Imports

```python
import sys, os
sys.path.insert(0, os.path.abspath('..'))

# Load saved ML models (Member 3 saves these on notebook run)
import joblib
model          = joblib.load('../models/product_recommendation_model.pkl')
prod_catalogue = joblib.load('../models/product_catalogue.pkl')

# Import ML functions from Member 3's notebook
# (run ml_model.ipynb first so the functions are defined)
from src.constants import (
    PRODUCT_CATEGORIES, SAMPLE_PROFILES, build_user_profile,
    CITIES, URBAN_CITIES, CUSTOMER_SEGMENTS,
)

# Import other modules (once Members 1 and 2 commit their code)
from src.rules_engine  import apply_rules
from src.search_module import find_product_path

# predict_product and recommend_products are defined in ml_model.ipynb
# Either run that notebook first, or copy the function definitions here
```

---

## `recommend()` — Full Pipeline Function

```python
def recommend(user_profile: dict) -> dict:
    """
    Run the full recommendation pipeline for a user.

    Parameters
    ----------
    user_profile : dict — from build_user_profile() in src/constants.py

    Returns
    -------
    dict with keys:
        top_3_categories    : list[tuple[str, float]]  — [(category, confidence), ...]
        recommended_products: list[tuple[str, float]]  — [(product_name, score), ...]
        roadmap             : list[str]                — category path to top-1 category
        eligible            : list[str]                — categories that passed rules
    """
    # Step 1: Rules Engine — filter eligible categories
    eligible = apply_rules(user_profile)
    if not eligible:
        eligible = PRODUCT_CATEGORIES  # fallback: all categories

    # Step 2: A* Search — find reachability path for each eligible category
    paths = {}
    for cat in eligible:
        path = find_product_path(user_profile, cat)
        paths[cat] = path if path else [cat]  # fallback: direct single-step

    # Step 3: ML Model — re-rank eligible categories by purchase probability
    categories = predict_product(user_profile, candidates=list(paths.keys()))
    if not categories:
        categories = [(cat, 0.0) for cat in eligible]

    # Step 4: Product-level recommendations for top category  ← NEW
    top_category = categories[0][0]
    products = recommend_products(user_profile, category=top_category, top_n=3)

    return {
        "top_3_categories":    categories[:3],
        "recommended_products": products,
        "roadmap":             paths[top_category],
        "eligible":            eligible,
    }
```

---

## `display_recommendation()` — Output Formatter

```python
def display_recommendation(profile_name: str, user_profile: dict, result: dict) -> None:
    """Pretty-print a single recommendation result."""
    print("=" * 60)
    print(f"  RECOMMENDATION FOR: {profile_name.upper()}")
    print("=" * 60)
    print(f"  Age: {user_profile['age']}  |  Gender: {user_profile['gender']}")
    print(f"  City: {user_profile['city']}  |  Spend: ₹{user_profile['median_spend']:,.0f}  ({user_profile['price_range']})")
    print(f"  Segment: {user_profile.get('customer_segment', 'N/A')}")
    print()
    print(f"  Eligible categories : {result['eligible']}")
    print()
    print("  Top 3 Categories:")
    for i, (cat, conf) in enumerate(result['top_3_categories'], 1):
        print(f"    {i}. {cat:<15} ({conf:.0%} confidence)")
    print()
    print(f"  Recommended Products (from {result['top_3_categories'][0][0]}):")
    for i, (product, score) in enumerate(result['recommended_products'], 1):
        print(f"    {i}. {product:<20} (score: {score:.2f})")
    print()
    print(f"  Discovery Roadmap → {result['top_3_categories'][0][0]}:")
    print(f"    {' → '.join(result['roadmap'])}")
    print()
```

---

## Demo — All Sample Profiles

```python
for profile_name, user_profile in SAMPLE_PROFILES.items():
    result = recommend(user_profile)
    display_recommendation(profile_name, user_profile, result)
```

**Expected output (illustrative):**

```
============================================================
  RECOMMENDATION FOR: TECH_SPENDER
============================================================
  Age: 42  |  Gender: Male
  City: Mumbai  |  Spend: ₹4,500  (High)
  Segment: Frequent Buyer

  Eligible categories : ['Sports', 'Home & Garden', 'Electronics', 'Fashion', 'Beauty']

  Top 3 Categories:
    1. Electronics      (84% confidence)
    2. Sports           (61% confidence)
    3. Home & Garden    (42% confidence)

  Recommended Products (from Electronics):
    1. Smartphone            (score: 0.91)
    2. Laptop                (score: 0.87)
    3. Headphones            (score: 0.73)

  Discovery Roadmap → Electronics:
    Fashion → Sports → Electronics
```

---

## Notebook Structure (Section 4 of `career_recommender.ipynb`)

```
## Section 4: Integration & System Design

### 4.1 Pipeline Overview
  (markdown diagram — Rules → A* → ML → Products)

### 4.2 Imports & Model Loading
  (load saved pkl files, import functions)

### 4.3 recommend() — Full Pipeline Function
  (code cell with recommend() definition)

### 4.4 display_recommendation() — Output Formatter
  (code cell with display function)

### 4.5 Demo — All Sample Profiles
  (loop over SAMPLE_PROFILES, display each result)

### 4.6 Pipeline Discussion
  (markdown: how the three AI components work together,
   why reachability-first + ML re-ranking gives better results
   than any single component alone)
```

---

## Updated SAMPLE_PROFILES Reference

| Key | Age | Gender | City | Spend | Tier | Segment | Expected top product |
|---|---|---|---|---|---|---|---|
| budget_browser | 22 | Female | Chennai | ₹800 | Low | New Visitor | Fiction / Lipstick |
| beauty_enthusiast | 30 | Female | Mumbai | ₹1,800 | Mid-Low | Frequent Buyer | Moisturizer / Lipstick |
| fashion_fan | 27 | Male | Delhi | ₹2,500 | Mid-Low | Occasional | Jeans / T-shirt |
| fitness_guy | 35 | Male | Bangalore | ₹3,200 | Mid-High | Frequent Buyer | Dumbbells / Yoga Mat |
| tech_spender | 42 | Male | Mumbai | ₹4,500 | High | Frequent Buyer | Smartphone / Laptop |

---

## Edge Cases to Handle

```python
# 1. apply_rules returns empty list
eligible = apply_rules(user_profile)
if not eligible:
    eligible = PRODUCT_CATEGORIES  # use all categories as fallback

# 2. find_product_path returns empty list
path = find_product_path(user_profile, cat)
paths[cat] = path if path else [cat]  # single-step fallback

# 3. predict_product returns fewer than 3 results
categories = predict_product(user_profile, candidates=list(paths.keys()))
# pad if needed:
top_3 = categories[:3]
while len(top_3) < 3:
    top_3.append(("Unknown", 0.0))

# 4. recommend_products returns fewer than 3 products
products = recommend_products(user_profile, category=top_category, top_n=3)
# display however many are returned — don't crash
```

---

## What To Do

1. Pull `src/constants.py` from `main` (already updated — Indian cities, 6 categories)
2. Confirm `models/` directory exists and contains pkl files (Member 3 creates these)
3. Implement `recommend()` with the new 4-step flow above
4. Implement `display_recommendation()` showing product names
5. Run demo loop for all 5 SAMPLE_PROFILES
6. Write Section 4.6 discussion explaining how the three components collaborate
7. Create poster summarising the full system

## Rules
- Never change signatures of `apply_rules()`, `find_product_path()`, `predict_product()`, `recommend_products()`
- Never modify `src/constants.py` — read-only for you
- Do not push — user pushes manually
- PRs must be draft: `gh pr create --draft`
