# =============================================================================
# constants.py
# Shared constants for the Smart Product Recommendation System
# CIC6314 Artificial Intelligence — Group Project
#
# ALL MODULES MUST IMPORT FROM THIS FILE.
# Do not hardcode category names, field values, or price figures anywhere else.
# If you need to add/change something, update here and notify the team.
#
# DATASET: ecommerce_customer_behavior_dataset_v2.csv
# 17,049 transactions | 5,000 unique customers | aggregated to user profiles
# All values below are derived from this dataset — do not change them
# without updating the dataset and notifying the team.
#
# DOMAIN SWITCH NOTE (2026-06-06):
# Project switched from Career Recommendation → Smart Product Recommendation.
# See docs/DOMAIN_SWITCH_HANDOFF.md for full details of what changed and why.
# =============================================================================


# -----------------------------------------------------------------------------
# PRODUCT CATEGORIES
# 8 product labels from the dataset — exact strings used as ML class labels.
#   - Rules engine  (apply_rules)       must return values from this list
#   - A* graph      (find_product_path) must use these as node names
#   - ML model      (predict_product)   must use these as class labels
# Ordered roughly by average unit price (ascending) — mirrors A* graph structure.
# -----------------------------------------------------------------------------
PRODUCT_CATEGORIES = [
    "Books",
    "Food",
    "Beauty",
    "Toys",
    "Fashion",
    "Sports",
    "Home & Garden",
    "Electronics",
]


# -----------------------------------------------------------------------------
# CATEGORY AVERAGE PRICES
# Derived from dataset Unit_Price column (mean per category).
# Used by the A* graph as node weights for the price-reachability heuristic.
# Member 1 imports this to build PRODUCT_GRAPH node attributes.
# -----------------------------------------------------------------------------
CATEGORY_AVG_PRICES = {
    "Books":          56,
    "Food":           71,
    "Beauty":        112,
    "Toys":          169,
    "Fashion":       276,
    "Sports":        493,
    "Home & Garden":  691,
    "Electronics":  1767,
}


# -----------------------------------------------------------------------------
# SPEND QUARTILE BOUNDARIES
# Derived from per-customer median Total_Amount.
# Used by rules engine to bucket users into spend tiers.
#   Low      : median_spend < 262
#   Mid-Low  : 262 <= median_spend < 502
#   Mid-High : 502 <= median_spend < 989
#   High     : median_spend >= 989
# -----------------------------------------------------------------------------
PRICE_RANGES = ["Low", "Mid-Low", "Mid-High", "High"]

SPEND_THRESHOLDS = {
    "Low":      (0,    262),
    "Mid-Low":  (262,  502),
    "Mid-High": (502,  989),
    "High":     (989,  float("inf")),
}

def get_price_range(median_spend: float) -> str:
    """Return the spend tier label for a given median_spend value."""
    for tier, (low, high) in SPEND_THRESHOLDS.items():
        if low <= median_spend < high:
            return tier
    return "High"


# -----------------------------------------------------------------------------
# AGE GROUPS
# Used by rules engine for demographic filtering.
# Age is an indirect predictor of category (works through spending behaviour),
# but is valid for commonsense rule logic.
# -----------------------------------------------------------------------------
AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "56+"]

def get_age_group(age: int) -> str:
    """Map a numeric age to its age group label."""
    if age <= 25:  return "18-25"
    if age <= 35:  return "26-35"
    if age <= 45:  return "36-45"
    if age <= 55:  return "46-55"
    return "56+"


# -----------------------------------------------------------------------------
# GENDERS
# Exact strings from the dataset "Gender" column.
# -----------------------------------------------------------------------------
GENDERS = ["Female", "Male", "Other"]


# -----------------------------------------------------------------------------
# CITIES
# 10 Turkish cities from the dataset "City" column.
# Urban cities (Istanbul, Ankara, Izmir) vs regional — used by rules engine.
# -----------------------------------------------------------------------------
CITIES = [
    "Adana", "Ankara", "Antalya", "Bursa", "Eskisehir",
    "Gaziantep", "Istanbul", "Izmir", "Kayseri", "Konya",
]

URBAN_CITIES = ["Istanbul", "Ankara", "Izmir"]


# -----------------------------------------------------------------------------
# DEVICE TYPES
# Exact strings from the dataset "Device_Type" column.
# -----------------------------------------------------------------------------
DEVICE_TYPES = ["Desktop", "Mobile", "Tablet"]


# -----------------------------------------------------------------------------
# PAYMENT METHODS
# Exact strings from the dataset "Payment_Method" column.
# -----------------------------------------------------------------------------
PAYMENT_METHODS = [
    "Bank Transfer", "Cash on Delivery", "Credit Card",
    "Debit Card", "Digital Wallet",
]


# -----------------------------------------------------------------------------
# USER PROFILE SCHEMA
# Standard input format for ALL three modules.
#
#   apply_rules(user_profile)              → list[str]  eligible categories
#   find_product_path(user_profile,        → list[str]  path of categories
#                     target_category)
#   predict_product(user_profile,          → list[tuple[str, float]]
#                   candidates=None)          [(category, confidence), ...]
#                                             sorted by confidence descending
#
# NOTE ON MEDIAN_SPEND: pass the user's typical spend per transaction as a
# float (e.g. 350.0). Use the raw transaction amount — not normalised.
# -----------------------------------------------------------------------------

def build_user_profile(
    age:            int,
    gender:         str,
    city:           str,
    median_spend:   float,
    device_type:    str = "Mobile",
    payment_method: str = "Credit Card",
) -> dict:
    """
    Build and validate a user profile dict.
    Raises ValueError if any field contains an unrecognised value.

    Parameters
    ----------
    age            : int, 18-75
    gender         : one of GENDERS
    city           : one of CITIES
    median_spend   : float, user's median transaction spend
    device_type    : one of DEVICE_TYPES
    payment_method : one of PAYMENT_METHODS

    Returns
    -------
    dict with all fields validated, plus derived fields: age_group, price_range
    """
    if not (18 <= age <= 75):
        raise ValueError("age must be between 18 and 75")
    if gender not in GENDERS:
        raise ValueError(f"gender must be one of {GENDERS}")
    if city not in CITIES:
        raise ValueError(f"city must be one of {CITIES}")
    if median_spend < 0:
        raise ValueError("median_spend must be non-negative")
    if device_type not in DEVICE_TYPES:
        raise ValueError(f"device_type must be one of {DEVICE_TYPES}")
    if payment_method not in PAYMENT_METHODS:
        raise ValueError(f"payment_method must be one of {PAYMENT_METHODS}")

    return {
        "age":            int(age),
        "gender":         gender,
        "city":           city,
        "median_spend":   float(median_spend),
        "device_type":    device_type,
        "payment_method": payment_method,
        "age_group":      get_age_group(age),
        "price_range":    get_price_range(median_spend),
    }


# -----------------------------------------------------------------------------
# SAMPLE USER PROFILES
# Use these to test your module during development.
# Member 4 will use these for the final integration demo.
# Values reflect realistic customer segments from the dataset.
# -----------------------------------------------------------------------------

SAMPLE_PROFILES = {

    "budget_reader": build_user_profile(
        age           = 22,
        gender        = "Female",
        city          = "Konya",
        median_spend  = 55.0,       # Low spend → Books / Food
        device_type   = "Mobile",
        payment_method= "Digital Wallet",
    ),

    "beauty_shopper": build_user_profile(
        age           = 30,
        gender        = "Female",
        city          = "Istanbul",
        median_spend  = 180.0,      # Low-Mid spend → Beauty / Toys
        device_type   = "Mobile",
        payment_method= "Credit Card",
    ),

    "fashion_enthusiast": build_user_profile(
        age           = 27,
        gender        = "Male",
        city          = "Ankara",
        median_spend  = 380.0,      # Mid-Low spend → Fashion / Sports
        device_type   = "Mobile",
        payment_method= "Debit Card",
    ),

    "sports_buyer": build_user_profile(
        age           = 35,
        gender        = "Male",
        city          = "Izmir",
        median_spend  = 650.0,      # Mid-High spend → Sports / Home
        device_type   = "Desktop",
        payment_method= "Credit Card",
    ),

    "tech_spender": build_user_profile(
        age           = 42,
        gender        = "Male",
        city          = "Istanbul",
        median_spend  = 1500.0,     # High spend → Electronics
        device_type   = "Desktop",
        payment_method= "Credit Card",
    ),
}
