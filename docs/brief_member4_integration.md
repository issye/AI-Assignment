# Agent Brief — Member 4: Integration & System Design
# CIC6314 Smart Product Recommendation System

## Your job
Wire the three AI components into a complete end-to-end pipeline in `notebooks/career_recommender.ipynb` (integration section). Produce a clean demo using all 5 sample profiles. The domain switched from Career Recommendation to Smart Product Recommendation.

---

## Pipeline you are building
```
User Profile
     │
     ▼
apply_rules(user_profile)           ← Member 2
     │ list[str] eligible categories
     ▼
find_product_path(user_profile,     ← Member 1
                  target_category)
     │ dict: {category: list[str] path}
     ▼
predict_product(user_profile,       ← Member 3
                candidates)
     │ list[tuple[str, float]]
     ▼
Final Output:
  Top-3 recommended categories + confidence scores
  Step-by-step purchase journey for top-1
```

---

## Function signatures — exactly as implemented

### Member 2 — Rules Engine
```python
from src.rules_engine import apply_rules

apply_rules(user_profile: dict) -> list[str]
# Returns: eligible product categories from PRODUCT_CATEGORIES
# Returns: [] if no rules match — handle with fallback (see below)

# Example output:
# ["Books", "Beauty", "Fashion", "Food"]
```

### Member 1 — A* Search
```python
from src.search_module import find_product_path

find_product_path(user_profile: dict, target_category: str) -> list[str]
# Returns: ordered category steps from start to target, inclusive
# Returns: [] if no path exists

# Example output:
# ["Fashion", "Sports", "Home & Garden", "Electronics"]
```

### Member 3 — ML Model
```python
# predict_product is defined in notebooks/ml_model.ipynb
# For integration, load the saved model and encoders from models/
# OR import the function if running as a single notebook

predict_product(user_profile: dict, candidates: list = None) -> list[tuple[str, float]]
# Returns: [(category, confidence), ...] sorted by confidence descending
# If candidates provided: only scores those categories (re-ranking mode)
# If candidates=None: scores all 8 PRODUCT_CATEGORIES

# Example output (re-ranking mode with candidates=["Electronics","Sports","Home & Garden"]):
# [("Electronics", 0.71), ("Sports", 0.18), ("Home & Garden", 0.11)]
```

---

## Imports

```python
import sys, os
sys.path.append(os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from src.constants import (
    PRODUCT_CATEGORIES,
    CATEGORY_AVG_PRICES,
    SAMPLE_PROFILES,
    build_user_profile,
)
from src.rules_engine  import apply_rules
from src.search_module import find_product_path

# ML model — load saved artefacts
import sys
sys.path.insert(0, '../notebooks')
# OR redefine predict_product here using saved models:
rf_model    = joblib.load('../models/rf_product_model.pkl')
enc_gender  = joblib.load('../models/encoder_gender.pkl')
enc_city    = joblib.load('../models/encoder_city.pkl')
enc_payment = joblib.load('../models/encoder_payment.pkl')
enc_device  = joblib.load('../models/encoder_device.pkl')
```

---

## Full integration function

```python
def recommend(user_profile: dict) -> dict:
    """
    Full pipeline: Rules → A* → ML → Output.

    Returns
    -------
    dict with keys:
        'top_3'   : list[tuple[str, float]] — top 3 categories with confidence
        'roadmap' : list[str]              — purchase journey to top-1 category
        'eligible': list[str]              — categories that passed rules filter
    """

    # ── Step 1: Rules Engine — filter eligible categories ─────────────────
    eligible = apply_rules(user_profile)
    if not eligible:
        eligible = PRODUCT_CATEGORIES  # fallback: all categories

    # ── Step 2: A* Search — compute path to each eligible category ────────
    paths     = {}
    path_costs = {}

    for cat in eligible:
        path = find_product_path(user_profile, cat)
        paths[cat] = path
        # path cost = sum of price gaps at each step
        cost = 0
        for step in path:
            cost += abs(CATEGORY_AVG_PRICES[step] - user_profile['median_spend'])
        path_costs[cat] = cost

    # Sort eligible by A* path cost (cheapest = most reachable first)
    candidates = sorted(eligible, key=lambda c: path_costs[c])

    # ── Step 3: ML Model — re-rank candidates by predicted preference ──────
    predictions = predict_product(user_profile, candidates=candidates)

    # ── Step 4: Output ────────────────────────────────────────────────────
    top_3   = predictions[:3]
    top1    = top_3[0][0]
    roadmap = paths.get(top1, [top1])

    return {
        'top_3':    top_3,
        'roadmap':  roadmap,
        'eligible': eligible,
    }
```

---

## Display function

```python
def display_recommendation(profile_name: str, user_profile: dict, result: dict):
    print('=' * 60)
    print(f'  Profile     : {profile_name}')
    print(f'  Age         : {user_profile["age"]} ({user_profile["age_group"]})')
    print(f'  Gender      : {user_profile["gender"]}')
    print(f'  City        : {user_profile["city"]}')
    print(f'  Median spend: RM {user_profile["median_spend"]:.0f} ({user_profile["price_range"]})')
    print('=' * 60)

    print(f'\n  Eligible categories (rules filter): {result["eligible"]}')
    print(f'\n  Top-3 Recommendations:')
    for i, (cat, conf) in enumerate(result['top_3'], 1):
        bar = '█' * int(conf * 20)
        print(f'    {i}. {cat:<18} {conf*100:.1f}%  {bar}')

    print(f'\n  Purchase journey to "{result["top_3"][0][0]}":')
    print(f'    {" → ".join(result["roadmap"])}')
    print()
```

---

## Demo — run all sample profiles

```python
for profile_name, profile in SAMPLE_PROFILES.items():
    result = recommend(profile)
    display_recommendation(profile_name, profile, result)
```

---

## Expected outputs per sample profile

| Profile | Expected top-1 | Expected roadmap end |
|---|---|---|
| budget_reader | Books or Food | Short path (already near Books) |
| beauty_shopper | Beauty | Short path (already near Beauty) |
| fashion_enthusiast | Fashion or Sports | Fashion → Sports or similar |
| sports_buyer | Sports or Home & Garden | Sports → Home & Garden |
| tech_spender | Electronics | Home & Garden → Electronics |

---

## Edge cases to handle

| Situation | How to handle |
|---|---|
| `apply_rules()` returns `[]` | Fall back to all `PRODUCT_CATEGORIES` |
| `find_product_path()` returns `[]` | Use `[target_category]` as single-step path |
| `predict_product()` returns fewer than 3 items | Show however many exist |
| User's `median_spend` is 0 | Still valid — use as-is |

---

## Notebook structure for your section

```
## Section 4: Integration & System Design

### 4.1 Overview — Pipeline Architecture
  (markdown: explain Logic → A* → ML → Output with diagram)

### 4.2 Imports & Model Loading

### 4.3 recommend() — Full Pipeline Function

### 4.4 display_recommendation() — Output Formatter

### 4.5 Demo — All Sample Profiles
  (run all 5, show formatted output)

### 4.6 Pipeline Visualisation
  (optional: bar chart of top-3 for each profile)

### 4.7 Discussion
  (how the three components contribute, what each adds)
```

---

## Do NOT touch
- `src/constants.py` — read only
- `notebooks/ml_model.ipynb` — Member 3's file
- `src/rules_engine.py` — Member 2's file
- `src/search_module.py` — Member 1's file
