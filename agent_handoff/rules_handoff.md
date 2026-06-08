# Agent Handoff — Rules Engine (Member 2)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/rules`
**Deadline:** 27 June 2026, 7 PM

---

## What This Module Does

The rules engine is the **first component** every user profile passes through. Its job is to decide which product categories a customer is even eligible to receive recommendations for, before any machine learning runs.

Think of it as a gatekeeper: it narrows the field of 8 possible categories down to a sensible subset based on what we know about the customer's spending behaviour and purchase history.

---

## Why Rules Matter — The Cold-Start Problem

The ML model (Member 3) learns from purchase history. If a customer has never bought anything, there is no history — the ML model has nothing to work from.

The rules engine solves this. It uses the signals we *do* have — how much the customer spends, how often they buy, how long since their last order — to produce a reasonable eligible set even when there is zero purchase history.

**New customers:** the rules engine's output IS the entire recommendation signal. The ML model's category scoring and the search module's BFS are both bypassed. The integration layer (Member 4) routes new users directly to a popularity-based recommendation using only your eligible list.

**Returning customers:** your eligible list acts as a broad gate. The search module then narrows it further using category similarity, and the ML model scores what remains. Your rules set the ceiling; the ML model does the personalisation.

---

## How This Module Fits Into the Full Pipeline

```
User Profile
     │
     ▼
Rules Engine (Member 2)  ◄─── YOU ARE HERE
  apply_rules(profile) → eligible: list of up to 8 categories
     │
     │  (new user with no history?)
     ├──────────────────────────────────────────────────────────►
     │                                                          │
     │  (returning user with history)                          ▼
     ▼                                               find_popular_categories()
Search Module (Member 1)                             → popular categories shown
  find_reachable_categories(profile, eligible)       → no ML scoring needed
     │
     ▼
ML Module (Member 3)
  predict_product(profile, candidates=shortlist)
     │
     ▼
Integration (Member 4)
  recommend(profile) → final output dict
```

The routing decision is made by Member 4. You only need to return the eligible list — you do not need to know which path the user takes.

---

## Function Signature

```python
apply_rules(user_profile: dict) -> list[str]
```

- Input: a user profile dict from `build_user_profile()` in `src/constants.py`
- Output: a list of category strings, each from `PRODUCT_CATEGORIES`
- Order must match `PRODUCT_CATEGORIES` list order (use the return pattern shown below)
- Must never return an empty list — use the fallback at the end

---

## User Profile Schema

No demographics — the dataset has no age, gender, or location fields. All signals are behavioural.

```python
user_profile = {
    "customer_id":          str,     # e.g. "17850"
    "purchase_history":     list,    # StockCode strings — [] for new users
    "avg_order_value":      float,   # mean basket value in £ across all invoices
    "total_invoices":       int,     # number of distinct orders placed
    "recency_days":         int,     # days since last order (ref date: 2011-12-09)
    "price_range":          str,     # "Low" | "Mid-Low" | "Mid-High" | "High"
    "customer_segment":     str,     # "New" | "Occasional" | "Frequent"
    "favourite_category":   str,     # most-purchased category — None if no history
    "purchased_categories": list,    # all distinct categories bought — [] if no history
}
```

`price_range` and `customer_segment` are already computed from the raw values — you do not need to derive them yourself. They come pre-calculated in the profile dict.

---

## The 10 Rules

Rules 1–4 gate by spend tier. Rules 5–10 refine by behaviour. Rules 7 and 10 are restrictive (they override or shrink the eligible set). All others are additive.

```python
def apply_rules(user_profile: dict) -> list[str]:
    eligible = set()

    # ── SPEND TIER RULES (1–4) ────────────────────────────────────────────────
    price_range = user_profile['price_range']

    # Rule 1: Low spenders → affordable, high-volume categories
    if price_range == 'Low':
        eligible.update({'Home Decor', 'Stationery & Craft', 'Seasonal & Gifts'})

    # Rule 2: Mid-Low spenders → mid-range categories
    if price_range == 'Mid-Low':
        eligible.update({'Home Decor', 'Kitchen & Dining',
                         'Seasonal & Gifts', 'Fashion & Accessories'})

    # Rule 3: Mid-High spenders → higher-value categories
    if price_range == 'Mid-High':
        eligible.update({'Kitchen & Dining', 'Home Decor',
                         'Toys & Games', 'Garden & Outdoor'})

    # Rule 4: High spenders → full catalogue
    if price_range == 'High':
        eligible.update(set(PRODUCT_CATEGORIES))

    # ── BEHAVIOURAL RULES (5–10) ──────────────────────────────────────────────

    # Rule 5: Favourite category is always eligible
    # Rationale: a customer's primary interest should never be excluded
    if user_profile.get('favourite_category'):
        eligible.add(user_profile['favourite_category'])

    # Rule 6: Broad buyer (3+ categories purchased) → full catalogue
    # Rationale: customers who already explore widely should see everything
    if len(user_profile.get('purchased_categories', [])) >= 3:
        eligible.update(set(PRODUCT_CATEGORIES))

    # Rule 7: Focused buyer (exactly 1 category ever purchased) → favourite + 1 neighbour
    # Rationale: don't overwhelm someone who buys only one type of product;
    # gently suggest the single most similar adjacent category using the ALS similarity matrix
    if len(user_profile.get('purchased_categories', [])) == 1:
        fav = user_profile['favourite_category']
        eligible = {fav}
        try:
            import pickle
            cat_sim = pickle.load(open('models/category_similarity.pkl', 'rb'))
            neighbours = cat_sim[fav].drop(fav).sort_values(ascending=False)
            eligible.add(neighbours.index[0])
        except Exception:
            eligible.update({'Home Decor', 'Seasonal & Gifts'})  # fallback if pkl missing

    # Rule 8: Dormant customer (90+ days inactive) → re-engagement categories
    # Rationale: seasonal and food items drive impulse re-engagement after a long absence
    if user_profile.get('recency_days', 0) > 90:
        eligible.update({'Seasonal & Gifts', 'Food & Confectionery'})

    # Rule 9: Frequent buyer → full catalogue
    # Rationale: high-frequency buyers explore broadly; no restrictions needed
    if user_profile.get('customer_segment') == 'Frequent':
        eligible.update(set(PRODUCT_CATEGORIES))

    # Rule 10: New customer (no history) → most popular 3 categories only
    # Rationale: without any purchase signal, only surface high-confidence categories.
    # This OVERRIDES all previous rules — it replaces the eligible set, not adds to it.
    if user_profile.get('customer_segment') == 'New':
        eligible = {'Home Decor', 'Seasonal & Gifts', 'Kitchen & Dining'}

    # ── FALLBACK ──────────────────────────────────────────────────────────────
    # Should never trigger — but guarantees a non-empty return
    if not eligible:
        eligible = set(PRODUCT_CATEGORIES)

    # Return in consistent PRODUCT_CATEGORIES order (never arbitrary set order)
    return [cat for cat in PRODUCT_CATEGORIES if cat in eligible]
```

---

## Expected Outputs for SAMPLE_PROFILES

These are the 5 test profiles defined in `src/constants.py`. Run your function against all 5 and verify these outputs before pushing.

| Profile | Segment | Spend | Rules firing | Expected eligible |
|---|---|---|---|---|
| `new_customer` | New | Low | Rule 10 overrides all | Home Decor, Seasonal & Gifts, Kitchen & Dining |
| `gift_buyer` | Occasional | Low | Rules 1, 5, 8 | Home Decor, Stationery & Craft, Seasonal & Gifts, Food & Confectionery |
| `craft_lover` | Occasional | Low | Rule 7 (single-cat buyer) | Stationery & Craft + 1 most-similar neighbour |
| `home_decorator` | Frequent | Low | Rule 9 | All 8 categories |
| `kitchen_enthusiast` | Frequent | Mid-Low | Rule 9 | All 8 categories |

---

## Constants to Import

```python
from src.constants import (
    PRODUCT_CATEGORIES,   # the 8 valid category strings — only return from this list
    SPEND_THRESHOLDS,     # £ quartile boundaries (reference only — price_range is pre-computed)
    PRICE_RANGES,         # ["Low", "Mid-Low", "Mid-High", "High"]
    CUSTOMER_SEGMENTS,    # ["New", "Occasional", "Frequent"]
)
```

Never hardcode category names, price values, or segment labels. Everything comes from constants.

---

## What To Do

1. Pull `main` to get the latest `src/constants.py`
2. Create `src/rules_engine.py` with the `apply_rules()` function above
3. Test against all 5 `SAMPLE_PROFILES` and verify expected outputs in the table above
4. Add a comment to every rule explaining the real-world rationale (already included above)
5. Create draft PR: `gh pr create --draft`

## Rules

- Return values must all be from `PRODUCT_CATEGORIES` — never invent new strings
- Return order must match `PRODUCT_CATEGORIES` list order
- Never change `apply_rules()` signature — Member 4 calls it directly
- Do not push — user pushes manually
