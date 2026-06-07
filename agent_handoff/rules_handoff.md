# Agent Handoff — Rules Engine (Member 2)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/rules`
**Deadline:** 27 June 2026, 7 PM

---

## What Changed and Why

The project has **fully pivoted** from the suvroo dataset to the **UCI Online Retail dataset**
and from a demographic rule system to a **behavioural rule system**.

| What changed | Old (suvroo) | New (Online Retail) |
|---|---|---|
| Dataset | Synthetic, Indian market, ₹ | Real transactions, UK, £ |
| Recommendation approach | Rule + ML binary classifier | Item-based Collaborative Filtering (CF) |
| Rule basis | Demographics (age, gender, city) | Behaviour (spend, purchase history, recency) |
| Categories | 6 (Books, Beauty, Electronics, Fashion, Sports, Home & Garden) | 8 (see below) |
| Spend thresholds | ₹ quartiles | £ quartiles from real data |
| Demographics | Age groups, gender, city, urban flag | **Removed entirely — not in dataset** |

---

## Your Module's New Role

The rules engine now serves as the **cold-start engine**:

- **New users (no purchase history):** Rules output IS the entire recommendation signal.
  CF cannot score anything without purchase history — your rules drive everything.
- **Returning users (have purchase history):** Rules output is the eligible set that
  CF scores against. Your rules act as a broad gate, CF does the personalisation.

Member 4 handles routing between both paths — you don't need to know which path is taken.
Your function always returns an eligible category list.

---

## Function Signature — UNCHANGED

```python
apply_rules(user_profile: dict) -> list[str]
```

Returns a subset of `PRODUCT_CATEGORIES`. Order must match `PRODUCT_CATEGORIES` list order.

---

## New User Profile Schema

No age, gender, city, or device_type. All fields are behavioural:

```python
user_profile = {
    "customer_id":          str,     # e.g. "17850"
    "purchase_history":     list,    # list of StockCode strings — [] for new users
    "avg_order_value":      float,   # mean basket value in £
    "total_invoices":       int,     # number of distinct orders
    "recency_days":         int,     # days since last order (ref: 2011-12-09)
    "price_range":          str,     # derived: "Low"|"Mid-Low"|"Mid-High"|"High"
    "customer_segment":     str,     # derived: "New"|"Occasional"|"Frequent"
    "favourite_category":   str,     # mode category across purchases — None if no history
    "purchased_categories": list,    # all distinct categories ever bought — [] if no history
}
```

---

## Constants to Import

```python
from src.constants import (
    PRODUCT_CATEGORIES,     # 8 category strings — your rules MUST return values from this list
    SPEND_THRESHOLDS,       # £ quartile boundaries
    PRICE_RANGES,           # ["Low", "Mid-Low", "Mid-High", "High"]
    CUSTOMER_SEGMENTS,      # ["New", "Occasional", "Frequent"]
    get_price_range,
    get_customer_segment,
)
import pickle, os
# For Rule 7:
# category_sim = pickle.load(open('models/category_similarity.pkl','rb'))
```

---

## New PRODUCT_CATEGORIES (8 categories)

```python
PRODUCT_CATEGORIES = [
    "Home Decor",           # default — dominant category in dataset
    "Kitchen & Dining",
    "Seasonal & Gifts",
    "Toys & Games",
    "Stationery & Craft",
    "Fashion & Accessories",
    "Garden & Outdoor",
    "Food & Confectionery",
]
```

---

## New SPEND_THRESHOLDS (£, from real data quartiles)

```
Low:      avg_order_value < £178.62      (bottom 25% of customers)
Mid-Low:  £178.62 – £293.90
Mid-High: £293.90 – £430.11
High:     >= £430.11                     (top 25%)
```

Note: Values appear high because the dataset includes wholesale buyers placing bulk orders.
The quartile split correctly reflects the actual customer distribution.

---

## 10 Rules to Implement

All rules are additive (add to eligible set). Rules 7 and 10 are restrictive (override or limit).

```python
def apply_rules(user_profile: dict) -> list[str]:
    eligible = set()

    # ── Primary rules: spend tier (Rules 1–4) ─────────────────────────────
    price_range = user_profile['price_range']

    # RULE 1: Low spenders — affordable, broad-appeal categories
    if price_range == 'Low':
        eligible.update({'Home Decor', 'Stationery & Craft', 'Seasonal & Gifts'})

    # RULE 2: Mid-Low spenders — mid-range categories
    if price_range == 'Mid-Low':
        eligible.update({'Home Decor', 'Kitchen & Dining',
                         'Seasonal & Gifts', 'Fashion & Accessories'})

    # RULE 3: Mid-High spenders — higher-value categories
    if price_range == 'Mid-High':
        eligible.update({'Kitchen & Dining', 'Home Decor',
                         'Toys & Games', 'Garden & Outdoor'})

    # RULE 4: High spenders — full catalogue access
    if price_range == 'High':
        eligible.update(set(PRODUCT_CATEGORIES))

    # ── Behavioural rules (Rules 5–10) ────────────────────────────────────

    # RULE 5: Favourite category always eligible (user's primary interest)
    if user_profile.get('favourite_category'):
        eligible.add(user_profile['favourite_category'])

    # RULE 6: Multi-category buyer (3+ categories) — broaden to full catalogue
    if len(user_profile.get('purchased_categories', [])) >= 3:
        eligible.update(set(PRODUCT_CATEGORIES))

    # RULE 7: Single-category buyer — focus on favourite + one most-similar neighbour
    # Rationale: don't overwhelm a focused buyer; gently suggest one adjacent category
    if len(user_profile.get('purchased_categories', [])) == 1:
        fav = user_profile['favourite_category']
        eligible = {fav}   # reset to just favourite
        try:
            import pickle
            cat_sim = pickle.load(open('models/category_similarity.pkl', 'rb'))
            neighbours = cat_sim[fav].drop(fav).sort_values(ascending=False)
            adjacent = neighbours.index[0]
            eligible.add(adjacent)
        except Exception:
            eligible.update({'Home Decor', 'Seasonal & Gifts'})   # fallback

    # RULE 8: Dormant customer (90+ days since last order) — re-engagement categories
    # Rationale: seasonal and food items drive impulse re-engagement
    if user_profile.get('recency_days', 0) > 90:
        eligible.update({'Seasonal & Gifts', 'Food & Confectionery'})

    # RULE 9: Frequent buyer — full catalogue access
    # Rationale: high-frequency buyers explore broadly
    if user_profile.get('customer_segment') == 'Frequent':
        eligible.update(set(PRODUCT_CATEGORIES))

    # RULE 10: New customer (no history) — restrict to most popular categories only
    # Rationale: without purchase signal, only surface high-confidence categories
    if user_profile.get('customer_segment') == 'New':
        eligible = {'Home Decor', 'Seasonal & Gifts', 'Kitchen & Dining'}

    # ── Fallback ──────────────────────────────────────────────────────────
    if not eligible:
        eligible = set(PRODUCT_CATEGORIES)

    # Return in PRODUCT_CATEGORIES order (never arbitrary set order)
    return [cat for cat in PRODUCT_CATEGORIES if cat in eligible]
```

---

## Expected Outputs for SAMPLE_PROFILES

| Profile | Segment | Price range | Key rules firing | Expected eligible |
|---|---|---|---|---|
| `new_customer` | New | Low | Rule 10 overrides all | Home Decor, Seasonal & Gifts, Kitchen & Dining |
| `gift_buyer` | Occasional | Low | Rules 1, 5 | Home Decor, Stationery & Craft, Seasonal & Gifts |
| `craft_lover` | Occasional | Low | Rule 7 (single-cat buyer) | Stationery & Craft + 1 adjacent |
| `home_decorator` | Frequent | Low | Rule 9 | All 8 categories |
| `kitchen_enthusiast` | Frequent | Mid-Low | Rule 9 | All 8 categories |

---

## What To Do

1. Pull `src/constants.py` from `main` (already updated for the pivot)
2. Create `src/rules_engine.py` and implement `apply_rules()` using the code above
3. Test against all 5 `SAMPLE_PROFILES` from `constants.py` and verify expected outputs
4. Document each rule with a comment explaining the real-world rationale
5. Do NOT change the `apply_rules()` function signature

## Rules

- Return values MUST be from `PRODUCT_CATEGORIES` — no other strings
- Return order must match `PRODUCT_CATEGORIES` list order
- Never change `apply_rules()` signature — Member 4 calls it directly
- All constants come from `src/constants.py` — never hardcode values
- Do not push — user pushes manually
- PRs must be draft: `gh pr create --draft`
