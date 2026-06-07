# =============================================================================
# constants.py
# Shared constants for the Smart Product Recommendation System
# CIC6314 Artificial Intelligence — Group Project
#
# ALL MODULES MUST IMPORT FROM THIS FILE.
# Do not hardcode category names, field values, or price figures anywhere else.
#
# DATASET: data/online+retail/Online Retail.xlsx
#          UCI Online Retail Dataset — 397,884 transactions
#          UK gift/novelty retailer, Dec 2010 – Dec 2011, £ currency
#          4,338 customers | 3,665 products | 8 categories
#
# APPROACH: Item-Based Collaborative Filtering (CF)
#   No demographics — CF does not require age, gender, or city.
#   User identity is derived entirely from purchase behaviour.
#
# PUBLIC INTERFACE:
#   Member 2 — apply_rules(user_profile)              → list[str]
#   Member 1 — find_reachable_categories(...)          → list[str]
#              find_popular_categories(...)             → list[tuple[str,int]]
#   Member 3 — predict_product(user_profile, ...)      → list[tuple[str,float]]
#              recommend_products(user_profile, ...)   → list[dict]
#   Member 4 — recommend(user_profile)                 → dict
#
# PIVOT NOTE (2026-06-07):
#   Pivoted from suvroo dataset (synthetic, Indian, ₹) to UCI Online Retail
#   (real transactions, UK, £). Recommendation approach changed from binary
#   ML classifier to item-based collaborative filtering.
#   Demographics (age, gender, city) removed — not available in Online Retail.
#   price_range and customer_segment retained as cold-start signals only.
# =============================================================================


# -----------------------------------------------------------------------------
# PRODUCT CATEGORIES
# 8 categories derived from keyword-matching on product Descriptions.
# Mapping stored in data/online+retail/product_categories.csv.
#
# Rules engine  (apply_rules)               must return values from this list
# Search module (find_reachable_categories) must use these as graph nodes
# CF model      (predict_product)           must use these as candidate labels
# -----------------------------------------------------------------------------
PRODUCT_CATEGORIES = [
    "Home Decor",
    "Kitchen & Dining",
    "Seasonal & Gifts",
    "Toys & Games",
    "Stationery & Craft",
    "Fashion & Accessories",
    "Garden & Outdoor",
    "Food & Confectionery",
]


# -----------------------------------------------------------------------------
# CATEGORY AVERAGE PRICES (£)
# Derived from Online Retail dataset (mean UnitPrice per category).
# Used as reference only — not used directly in CF scoring.
# -----------------------------------------------------------------------------
CATEGORY_AVG_PRICES = {
    "Home Decor":            2.48,
    "Kitchen & Dining":      3.12,
    "Seasonal & Gifts":      2.91,
    "Toys & Games":          3.45,
    "Stationery & Craft":    1.87,
    "Fashion & Accessories": 4.23,
    "Garden & Outdoor":      2.74,
    "Food & Confectionery":  2.15,
}


# -----------------------------------------------------------------------------
# SPEND QUARTILE BOUNDARIES (£)
# Derived from Online Retail customer avg_order_value distribution.
#   Q1 = £178.62  (Low / Mid-Low boundary)
#   Q2 = £293.90  (Mid-Low / Mid-High boundary)
#   Q3 = £430.11  (Mid-High / High boundary)
#
# NOTE: Values are high because some customers are wholesale buyers placing
# large repeat orders. The quartile split correctly reflects the distribution.
#
#   Low      : avg_order_value < £178.62     (bottom 25%)
#   Mid-Low  : £178.62 – £293.90
#   Mid-High : £293.90 – £430.11
#   High     : >= £430.11                    (top 25%)
# -----------------------------------------------------------------------------
PRICE_RANGES = ["Low", "Mid-Low", "Mid-High", "High"]

SPEND_THRESHOLDS = {
    "Low":      (0,       178.62),
    "Mid-Low":  (178.62,  293.90),
    "Mid-High": (293.90,  430.11),
    "High":     (430.11,  float("inf")),
}

def get_price_range(avg_order_value: float) -> str:
    """Return the spend tier label for a given avg_order_value (£)."""
    for tier, (low, high) in SPEND_THRESHOLDS.items():
        if low <= avg_order_value < high:
            return tier
    return "High"


# -----------------------------------------------------------------------------
# CUSTOMER SEGMENTS
# Derived from number of unique invoices per customer.
#   New        : total_invoices <= 2   (2,328 customers — 53.7%)
#   Occasional : 3–10 invoices         (1,673 customers — 38.6%)
#   Frequent   : > 10 invoices         (  337 customers —  7.8%)
#
# Used by rules engine for cold-start category gating.
# For returning users with purchase history, CF subsumes this signal.
# -----------------------------------------------------------------------------
CUSTOMER_SEGMENTS = ["New", "Occasional", "Frequent"]

def get_customer_segment(total_invoices: int) -> str:
    """Map invoice count to customer segment label."""
    if total_invoices <= 2:
        return "New"
    if total_invoices <= 10:
        return "Occasional"
    return "Frequent"


# -----------------------------------------------------------------------------
# USER PROFILE SCHEMA
# Standard input format for ALL modules.
#
# IMPORTANT: No age, gender, or city fields — not available in Online Retail.
# purchase_history uses StockCode strings (e.g. '85123A', '71053').
# favourite_category and purchased_categories are derived from purchase_history
# during build_user_profile() — do not set them manually.
#
# NOTE ON COLD-START: If purchase_history is empty, the system falls back to
# popularity-based recommendations via the search module. Member 4 handles
# routing between personalised (CF) and popular (cold-start) paths.
# -----------------------------------------------------------------------------

def build_user_profile(
    customer_id:      str,
    purchase_history: list,
    avg_order_value:  float,
    total_invoices:   int   = 1,
    recency_days:     int   = 30,
    _category_map:    dict  = None,   # internal: StockCode→category, injected by notebook
) -> dict:
    """
    Build and validate a user profile dict from Online Retail transaction data.

    Parameters
    ----------
    customer_id      : str   — CustomerID from Online Retail (e.g. '17850')
    purchase_history : list  — StockCodes the customer has bought (e.g. ['85123A','71053'])
                               Pass [] for new customers with no history.
    avg_order_value  : float — Mean basket value in £ across all invoices
    total_invoices   : int   — Number of distinct invoices (used for segment)
    recency_days     : int   — Days since last order (ref: 2011-12-09)
    _category_map    : dict  — Optional StockCode→category lookup injected by the notebook
                               to derive favourite_category and purchased_categories.
                               If not provided these fields are set to None / [].

    Returns
    -------
    dict with all fields plus derived: price_range, customer_segment,
    favourite_category, purchased_categories
    """
    if avg_order_value < 0:
        raise ValueError("avg_order_value must be non-negative")
    if recency_days < 0:
        raise ValueError("recency_days must be non-negative")

    # Derive category fields if category map provided
    favourite_category   = None
    purchased_categories = []
    if _category_map and purchase_history:
        bought_cats = [_category_map.get(str(sc)) for sc in purchase_history
                       if _category_map.get(str(sc))]
        if bought_cats:
            purchased_categories = list(dict.fromkeys(bought_cats))   # unique, order preserved
            from collections import Counter
            favourite_category = Counter(bought_cats).most_common(1)[0][0]

    return {
        "customer_id":          str(customer_id),
        "purchase_history":     [str(sc) for sc in purchase_history],
        "avg_order_value":      float(avg_order_value),
        "total_invoices":       int(total_invoices),
        "recency_days":         int(recency_days),
        "price_range":          get_price_range(avg_order_value),
        "customer_segment":     get_customer_segment(total_invoices),
        "favourite_category":   favourite_category,
        "purchased_categories": purchased_categories,
    }


# -----------------------------------------------------------------------------
# SAMPLE USER PROFILES
# Built from real Online Retail customers for testing and demo.
# CustomerIDs are real IDs from the dataset.
#
# Profile              | ID    | Segment    | Spend   | Favourite
# gift_buyer           | 13058 | Occasional | Low     | Seasonal & Gifts
# home_decorator       | 13094 | Frequent   | Low     | Home Decor
# kitchen_enthusiast   | 13631 | Frequent   | Mid-Low | Kitchen & Dining
# craft_lover          | 14460 | Occasional | Low     | Stationery & Craft
# new_customer         | N/A   | New        | Low     | None (cold-start)
#
# NOTE: favourite_category and purchased_categories are pre-populated here
# since _category_map is not available at import time. They reflect the
# real purchase history of each customer in the dataset.
# -----------------------------------------------------------------------------

def _make_profile(customer_id, purchase_history, avg_order_value, total_invoices,
                  recency_days, favourite_category, purchased_categories):
    """Build a profile dict directly (bypasses category map derivation)."""
    return {
        "customer_id":          str(customer_id),
        "purchase_history":     [str(sc) for sc in purchase_history],
        "avg_order_value":      float(avg_order_value),
        "total_invoices":       int(total_invoices),
        "recency_days":         int(recency_days),
        "price_range":          get_price_range(avg_order_value),
        "customer_segment":     get_customer_segment(total_invoices),
        "favourite_category":   favourite_category,
        "purchased_categories": purchased_categories,
    }


SAMPLE_PROFILES = {

    # Occasional buyer, Low spend, loves seasonal and gift products
    "gift_buyer": _make_profile(
        customer_id          = "13058",
        purchase_history     = ["47590B", "47590A", "23298", "23313", "22776"],
        avg_order_value      = 33.93,
        total_invoices       = 5,
        recency_days         = 24,
        favourite_category   = "Seasonal & Gifts",
        purchased_categories = ["Seasonal & Gifts", "Garden & Outdoor", "Kitchen & Dining"],
    ),

    # Frequent buyer, Home Decor focus, active shopper
    "home_decorator": _make_profile(
        customer_id          = "13094",
        purchase_history     = ["22174", "22791", "84946", "85123A"],
        avg_order_value      = 80.31,
        total_invoices       = 12,
        recency_days         = 20,
        favourite_category   = "Home Decor",
        purchased_categories = ["Home Decor", "Food & Confectionery"],
    ),

    # Frequent buyer, Kitchen focus, higher spend, slightly dormant
    "kitchen_enthusiast": _make_profile(
        customer_id          = "13631",
        purchase_history     = ["22423", "22968", "22625", "23173", "23118", "21257"],
        avg_order_value      = 279.13,
        total_invoices       = 14,
        recency_days         = 99,
        favourite_category   = "Kitchen & Dining",
        purchased_categories = ["Kitchen & Dining", "Home Decor", "Fashion & Accessories"],
    ),

    # Occasional buyer, craft and stationery focus, dormant (109 days)
    "craft_lover": _make_profile(
        customer_id          = "14460",
        purchase_history     = ["85019C", "17003", "21703", "85019A", "35651",
                                "21634", "21391", "20984", "20840", "10133",
                                "51014C", "51014L", "35646", "20992"],
        avg_order_value      = 27.15,
        total_invoices       = 7,
        recency_days         = 109,
        favourite_category   = "Stationery & Craft",
        purchased_categories = ["Stationery & Craft", "Fashion & Accessories", "Home Decor"],
    ),

    # New customer — no purchase history (cold-start path)
    "new_customer": _make_profile(
        customer_id          = "NEW_001",
        purchase_history     = [],
        avg_order_value      = 25.00,
        total_invoices       = 1,
        recency_days         = 0,
        favourite_category   = None,
        purchased_categories = [],
    ),
}
