# Agent Handoff — Rules Engine (Member 2)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/rules`
**Deadline:** 27 June 2026, 7 PM

---

## What Changed and Why

The project has pivoted to the **suvroo dataset** (Indian market, ₹ currency). Your module has the **lowest impact of all four members** — the rule logic stays the same, only the values change:

| What changed | Old value | New value |
|---|---|---|
| Category labels | 8 categories (incl. Food, Toys) | 6 categories (Food/Toys removed) |
| Spend thresholds | RM 262 / 502 / 989 | ₹ 1,636 / 2,740 / 3,879 |
| Cities | 10 Turkish cities | 5 Indian cities |
| Urban cities | Istanbul, Ankara, Izmir | Mumbai, Delhi, Bangalore |
| New feature | — | `customer_segment` (New Visitor / Occasional / Frequent Buyer) |

**Function signature unchanged:** `apply_rules(user_profile: dict) -> list[str]`

---

## Updated Constants (pull from `src/constants.py`)

```python
from src.constants import (
    PRODUCT_CATEGORIES,   # 6 categories — your rules MUST return values from this list
    SPEND_THRESHOLDS,     # updated ₹ quartile boundaries
    PRICE_RANGES,         # ["Low", "Mid-Low", "Mid-High", "High"]
    CITIES,               # ["Bangalore", "Chennai", "Delhi", "Kolkata", "Mumbai"]
    URBAN_CITIES,         # ["Mumbai", "Delhi", "Bangalore"]
    GENDERS,
    AGE_GROUPS,
    DEVICE_TYPES,
    CUSTOMER_SEGMENTS,    # NEW: ["New Visitor", "Occasional Shopper", "Frequent Buyer"]
    get_price_range,
    get_age_group,
)
```

### PRODUCT_CATEGORIES (valid return values)
```python
["Books", "Beauty", "Electronics", "Fashion", "Sports", "Home & Garden"]
```
Any category your rules return **must** be exactly one of these strings.

### SPEND_THRESHOLDS (₹ based)
```
Low:      Avg_Order_Value < ₹1,636        → entry-level spend
Mid-Low:  ₹1,636 – ₹2,740               → moderate spend
Mid-High: ₹2,740 – ₹3,879               → above-average spend
High:     ≥ ₹3,879                       → premium spend
```

### User Profile Schema (what `apply_rules()` receives)
```python
user_profile = {
    "age":              int,      # 18–60
    "gender":           str,      # "Female" | "Male" | "Other"
    "city":             str,      # one of CITIES (Indian)
    "median_spend":     float,    # Avg_Order_Value in ₹
    "device_type":      str,      # "Mobile" | "Desktop" | "Tablet"
    "payment_method":   str,      # one of PAYMENT_METHODS
    "customer_segment": str,      # NEW: one of CUSTOMER_SEGMENTS
    "age_group":        str,      # derived: "18-25" | "26-35" | ...
    "price_range":      str,      # derived: "Low" | "Mid-Low" | ...
}
```

---

## Rules to Implement (10 minimum for full marks)

### Primary rules — spend-based (4 rules, strongest signal)

```python
# RULE 1: Low spenders — entry-level categories
if user_profile['price_range'] == 'Low':
    eligible.update({'Books', 'Beauty'})

# RULE 2: Mid-Low spenders — mid-range categories
if user_profile['price_range'] == 'Mid-Low':
    eligible.update({'Beauty', 'Fashion', 'Sports'})

# RULE 3: Mid-High spenders — upper categories
if user_profile['price_range'] == 'Mid-High':
    eligible.update({'Fashion', 'Sports', 'Home & Garden', 'Electronics'})

# RULE 4: High spenders — premium categories
if user_profile['price_range'] == 'High':
    eligible.update({'Sports', 'Home & Garden', 'Electronics'})
```

### Secondary rules — demographic (6 rules, additive/subtractive)

```python
# RULE 5: Female users — Beauty and Fashion affinity
if user_profile['gender'] == 'Female':
    eligible.update({'Beauty', 'Fashion'})

# RULE 6: Male users — Sports and Electronics affinity
if user_profile['gender'] == 'Male':
    eligible.update({'Sports', 'Electronics'})

# RULE 7: Young adults (18-25) — Fashion and Sports, less Home & Garden
if user_profile['age_group'] == '18-25':
    eligible.update({'Fashion', 'Sports'})
    eligible.discard('Home & Garden')

# RULE 8: Older adults (46-55, 56+) — Home & Garden and Books affinity
if user_profile['age_group'] in ('46-55', '56+'):
    eligible.update({'Home & Garden', 'Books'})
    eligible.discard('Sports')

# RULE 9: Urban users (Mumbai, Delhi, Bangalore) — Electronics and Fashion
if user_profile['city'] in URBAN_CITIES:
    eligible.update({'Electronics', 'Fashion'})

# RULE 10: Frequent Buyers — broaden eligible set (cross-category buyers)
if user_profile['customer_segment'] == 'Frequent Buyer':
    eligible.update({'Beauty', 'Sports', 'Electronics'})
```

### Fallback
```python
# If no rules matched (shouldn't happen, but be safe)
if not eligible:
    eligible = set(PRODUCT_CATEGORIES)
```

---

## Implementation Pattern

```python
from src.constants import (
    PRODUCT_CATEGORIES, URBAN_CITIES, CUSTOMER_SEGMENTS,
    get_price_range, get_age_group,
)

def apply_rules(user_profile: dict) -> list[str]:
    """
    Apply logical inference rules to determine eligible product categories.

    Parameters
    ----------
    user_profile : dict — from build_user_profile() in src/constants.py

    Returns
    -------
    list[str] — subset of PRODUCT_CATEGORIES the user is eligible for
                Returns all categories if no rules match (fallback).
    """
    eligible = set()

    # --- Primary rules: spend tier ---
    price_range = user_profile['price_range']
    if price_range == 'Low':
        eligible.update({'Books', 'Beauty'})
    elif price_range == 'Mid-Low':
        eligible.update({'Beauty', 'Fashion', 'Sports'})
    elif price_range == 'Mid-High':
        eligible.update({'Fashion', 'Sports', 'Home & Garden', 'Electronics'})
    elif price_range == 'High':
        eligible.update({'Sports', 'Home & Garden', 'Electronics'})

    # --- Secondary rules: demographics ---
    if user_profile['gender'] == 'Female':
        eligible.update({'Beauty', 'Fashion'})
    if user_profile['gender'] == 'Male':
        eligible.update({'Sports', 'Electronics'})

    if user_profile['age_group'] == '18-25':
        eligible.update({'Fashion', 'Sports'})
        eligible.discard('Home & Garden')
    if user_profile['age_group'] in ('46-55', '56+'):
        eligible.update({'Home & Garden', 'Books'})
        eligible.discard('Sports')

    if user_profile['city'] in URBAN_CITIES:
        eligible.update({'Electronics', 'Fashion'})

    if user_profile.get('customer_segment') == 'Frequent Buyer':
        eligible.update({'Beauty', 'Sports', 'Electronics'})

    # --- Fallback ---
    if not eligible:
        eligible = set(PRODUCT_CATEGORIES)

    # Return only valid categories, preserving PRODUCT_CATEGORIES order
    return [cat for cat in PRODUCT_CATEGORIES if cat in eligible]
```

---

## Expected Outputs for SAMPLE_PROFILES

| Profile | Spend | Tier | Expected eligible categories |
|---|---|---|---|
| budget_browser | ₹800 | Low | Books, Beauty, Fashion (Female+Mobile) |
| beauty_enthusiast | ₹1,800 | Mid-Low | Beauty, Fashion, Sports, Electronics (Female+Urban+Frequent) |
| fashion_fan | ₹2,500 | Mid-Low | Beauty, Fashion, Sports, Electronics (Male+Urban) |
| fitness_guy | ₹3,200 | Mid-High | Fashion, Sports, Home & Garden, Electronics (Male+Urban+Frequent) |
| tech_spender | ₹4,500 | High | Sports, Home & Garden, Electronics (Male+Urban+Frequent) |

---

## What To Do

1. Pull `src/constants.py` from `main` (already updated)
2. Implement `apply_rules()` in `src/rules_engine.py`
3. Add `customer_segment` rule (Rule 10) — this is new vs the old brief
4. Test against all 5 SAMPLE_PROFILES and compare to expected outputs above
5. Document each rule with a comment explaining the real-world rationale
6. Keep `apply_rules()` signature identical — Member 4 calls this directly

## Rules
- Return values MUST be from `PRODUCT_CATEGORIES` — no other strings
- Never change `apply_rules()` signature
- All constants come from `src/constants.py` — never hardcode category names
- Do not push — user pushes manually
- PRs must be draft: `gh pr create --draft`
