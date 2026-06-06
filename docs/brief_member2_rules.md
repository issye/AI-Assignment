# Agent Brief — Member 2: Rules Engine
# CIC6314 Smart Product Recommendation System

## Your job
Implement `src/rules_engine.py` with one public function: `apply_rules(user_profile)`. The domain switched from Career Recommendation to Smart Product Recommendation. Write 8–10 logical rules that filter which product categories a user is eligible for, based on their profile.

---

## Context: Why the domain switched
The original career dataset had randomly assigned labels (8.3% accuracy — random chance). The team switched to an e-commerce dataset where `median_spend` strongly predicts product category (MI = 0.20). Demographic features (Age, Gender) have weak direct signal but are valid for commonsense rule logic.

---

## Pipeline (your position)
```
Rules Engine (YOU)       →  eligible product categories (list[str])
        │
        ▼
A* Search (Member 1)     →  ranks eligible categories by price reachability
        │
        ▼
ML Model (Member 3)      →  re-ranks candidates by predicted preference
        │
        ▼
Integration (Member 4)   →  Top-3 + purchase journey roadmap
```

You run **first**. Your output is the candidate space for everything that follows. If you return an empty list, the integration layer falls back to all categories.

---

## Your function signature

```python
def apply_rules(user_profile: dict) -> list:
    """
    Apply logical inference rules to filter eligible product categories.

    Parameters
    ----------
    user_profile : dict — from build_user_profile() in src/constants.py

    Returns
    -------
    list[str] — eligible categories from PRODUCT_CATEGORIES
                Empty list if no rules match (integration handles fallback)
    """
```

---

## User profile schema (your input)

```python
user_profile = {
    "age":            int,       # 18–75
    "gender":         str,       # "Female" | "Male" | "Other"
    "city":           str,       # one of CITIES (10 Turkish cities)
    "median_spend":   float,     # PRIMARY SIGNAL — user's median transaction spend
    "device_type":    str,       # "Mobile" | "Desktop" | "Tablet"
    "payment_method": str,       # "Credit Card" | "Debit Card" | "Digital Wallet" | "Bank Transfer" | "Cash on Delivery"
    "age_group":      str,       # "18-25" | "26-35" | "36-45" | "46-55" | "56+"
    "price_range":    str,       # "Low" | "Mid-Low" | "Mid-High" | "High"
}
```

---

## Constants to import

```python
from src.constants import (
    PRODUCT_CATEGORIES,
    CATEGORY_AVG_PRICES,
    SPEND_THRESHOLDS,
    URBAN_CITIES,
    GENDERS, CITIES, DEVICE_TYPES, PAYMENT_METHODS,
    SAMPLE_PROFILES,
    build_user_profile,
    get_price_range,
)
```

---

## Product categories and their avg prices (reference)

| Category | Avg Price | Typical buyer |
|---|---|---|
| Books | RM 56 | Low spenders, all ages |
| Food | RM 71 | Low spenders |
| Beauty | RM 112 | Low-mid spenders, Female skew |
| Toys | RM 169 | Parents (age 26-45), low-mid spenders |
| Fashion | RM 276 | Mid spenders, younger, Mobile users |
| Sports | RM 493 | Mid-high spenders, Male skew |
| Home & Garden | RM 691 | Mid-high spenders, older |
| Electronics | RM 1767 | High spenders, Desktop users |

---

## Spend thresholds (from dataset — use these exact values)

```python
SPEND_THRESHOLDS = {
    "Low":      (0,    262),
    "Mid-Low":  (262,  502),
    "Mid-High": (502,  989),
    "High":     (989,  float("inf")),
}
```

---

## Required rules — implement at least 8 of these

Design your rules as additive — start with an eligible set based on spend (primary), then expand or restrict based on demographics (secondary).

### Primary rules — spend-based (strongest signal, MI=0.20)
```
RULE 1: IF price_range == "Low"
        THEN eligible: Books, Food, Beauty

RULE 2: IF price_range == "Mid-Low"
        THEN eligible: Beauty, Toys, Fashion, Sports

RULE 3: IF price_range == "Mid-High"
        THEN eligible: Fashion, Sports, Home & Garden

RULE 4: IF price_range == "High"
        THEN eligible: Sports, Home & Garden, Electronics
```

### Secondary rules — demographic (additive adjustments)
```
RULE 5: IF gender == "Female"
        THEN add eligible: Beauty, Fashion
        (Female users skew toward Beauty and Fashion regardless of spend tier)

RULE 6: IF gender == "Male"
        THEN add eligible: Sports, Electronics
        (Male users skew toward Sports and Electronics)

RULE 7: IF age_group in ["18-25"]
        THEN add eligible: Fashion, Toys
        THEN remove eligible: Home & Garden
        (Young users less likely to buy home goods)

RULE 8: IF age_group in ["46-55", "56+"]
        THEN add eligible: Home & Garden, Books
        THEN remove eligible: Toys
        (Older users skew toward home and books)

RULE 9: IF city in URBAN_CITIES  (Istanbul, Ankara, Izmir)
        THEN add eligible: Electronics, Fashion
        (Urban users have higher Electronics/Fashion access)

RULE 10: IF device_type == "Mobile"
         THEN add eligible: Fashion, Beauty
         (Mobile shoppers skew toward fashion/beauty browsing)
```

---

## Implementation pattern

```python
def apply_rules(user_profile: dict) -> list:
    eligible = set()
    spend    = user_profile["median_spend"]
    age_grp  = user_profile["age_group"]
    gender   = user_profile["gender"]
    city     = user_profile["city"]
    device   = user_profile["device_type"]
    tier     = user_profile["price_range"]

    # ── Primary: spend-based rules ────────────────────────────────────────
    if tier == "Low":
        eligible.update(["Books", "Food", "Beauty"])
    elif tier == "Mid-Low":
        eligible.update(["Beauty", "Toys", "Fashion", "Sports"])
    elif tier == "Mid-High":
        eligible.update(["Fashion", "Sports", "Home & Garden"])
    elif tier == "High":
        eligible.update(["Sports", "Home & Garden", "Electronics"])

    # ── Secondary: demographic adjustments ───────────────────────────────
    if gender == "Female":
        eligible.update(["Beauty", "Fashion"])
    if gender == "Male":
        eligible.update(["Sports", "Electronics"])
    if age_grp == "18-25":
        eligible.update(["Fashion", "Toys"])
        eligible.discard("Home & Garden")
    if age_grp in ["46-55", "56+"]:
        eligible.update(["Home & Garden", "Books"])
        eligible.discard("Toys")
    if city in URBAN_CITIES:
        eligible.update(["Electronics", "Fashion"])
    if device == "Mobile":
        eligible.update(["Fashion", "Beauty"])

    # ── Validate: only return known categories ────────────────────────────
    return [c for c in PRODUCT_CATEGORIES if c in eligible]
```

---

## Sample profiles — expected outputs

| Profile | spend | tier | Expected eligible categories |
|---|---|---|---|
| budget_reader | 55.0 | Low | Books, Food, Beauty, Fashion (Female+Mobile) |
| beauty_shopper | 180.0 | Low | Beauty, Food, Books, Fashion, Toys (Female+Mobile+Urban) |
| fashion_enthusiast | 380.0 | Mid-Low | Beauty, Toys, Fashion, Sports, Electronics (Male+Urban) |
| sports_buyer | 650.0 | Mid-High | Fashion, Sports, Home & Garden, Electronics (Male+Urban+Desktop) |
| tech_spender | 1500.0 | High | Sports, Home & Garden, Electronics (Male+Urban+Desktop) |

---

## Rubric requirement
The rubric requires **8–10 well-defined rules with correct inference mechanism**. Each rule must be:
- Clearly documented with a comment explaining the reasoning
- Grounded in either the dataset signal or commonsense domain logic
- Traceable — you should be able to explain each rule to the lecturer

---

## Do NOT touch
- `src/constants.py` — read only
- `notebooks/ml_model.ipynb` — Member 3's file
- `notebooks/career_recommender.ipynb` beyond your section
