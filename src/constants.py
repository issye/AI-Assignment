# =============================================================================
# constants.py
# Shared constants for the Smart Product Recommendation System
# CIC6314 Artificial Intelligence — Group Project
#
# ALL MODULES MUST IMPORT FROM THIS FILE.
# Do not hardcode category names, field values, or price figures anywhere else.
# If you need to add/change something, update here and notify the team.
#
# DATASET: data/suvroo/customer_data_collection.csv
#          data/suvroo/product_recommendation_data.csv
# 10,000 customers | 10,000 products | Indian market (₹ currency)
# All values below are derived from this dataset.
#
# PIVOT NOTE (2026-06-06):
# System now recommends specific PRODUCTS (not just categories).
# Two public ML functions:
#   predict_product(user_profile, candidates)  → ranked categories
#   recommend_products(user_profile, category) → specific product names
# See agent_handoff/ml_handoff.md for full architecture details.
# =============================================================================


# -----------------------------------------------------------------------------
# PRODUCT CATEGORIES
# 6 product labels — exact strings used as ML class labels.
#   - Rules engine  (apply_rules)        must return values from this list
#   - A* graph      (find_product_path)  must use these as node names
#   - ML model      (predict_product)    must use these as class labels
#
# NOTE: reduced from 8 to 6 categories (Food and Toys removed).
#       "Fitness" in suvroo data maps to "Sports" here.
#       "Home Decor" in suvroo data maps to "Home & Garden" here.
# -----------------------------------------------------------------------------
PRODUCT_CATEGORIES = [
    "Books",
    "Beauty",
    "Electronics",
    "Fashion",
    "Sports",
    "Home & Garden",
]


# -----------------------------------------------------------------------------
# CATEGORY AVERAGE PRICES
# Derived from suvroo product_recommendation_data.csv (mean Price per category).
# NOTE: prices are nearly flat (~₹2,500 across all categories) because suvroo
# products were price-randomised. A* therefore uses co-purchase frequency for
# edge costs — NOT price gaps. These values are kept for reference only and
# for any fallback price-fit calculations in the ML layer.
# -----------------------------------------------------------------------------
CATEGORY_AVG_PRICES = {
    "Books":         2524,
    "Beauty":        2501,
    "Electronics":   2548,
    "Fashion":       2618,
    "Sports":        2578,
    "Home & Garden": 2549,
}


# -----------------------------------------------------------------------------
# SPEND QUARTILE BOUNDARIES
# Derived from suvroo customer_data_collection.csv Avg_Order_Value column.
# Q1=₹1,636 | Median=₹2,740 | Q3=₹3,879 | Range: ₹500–₹5,000
#
#   Low      : Avg_Order_Value < 1636
#   Mid-Low  : 1636 <= Avg_Order_Value < 2740
#   Mid-High : 2740 <= Avg_Order_Value < 3879
#   High     : Avg_Order_Value >= 3879
# -----------------------------------------------------------------------------
PRICE_RANGES = ["Low", "Mid-Low", "Mid-High", "High"]

SPEND_THRESHOLDS = {
    "Low":      (0,     1636),
    "Mid-Low":  (1636,  2740),
    "Mid-High": (2740,  3879),
    "High":     (3879,  float("inf")),
}

def get_price_range(median_spend: float) -> str:
    """Return the spend tier label for a given Avg_Order_Value (₹)."""
    for tier, (low, high) in SPEND_THRESHOLDS.items():
        if low <= median_spend < high:
            return tier
    return "High"


# -----------------------------------------------------------------------------
# AGE GROUPS
# Used by rules engine for demographic filtering.
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
# Exact strings from the suvroo Gender column.
# -----------------------------------------------------------------------------
GENDERS = ["Female", "Male", "Other"]


# -----------------------------------------------------------------------------
# CITIES
# 5 Indian cities from the suvroo Location column.
# Urban cities (Mumbai, Delhi, Bangalore) vs regional — used by rules engine.
# NOTE: changed from Turkish cities to Indian cities to match suvroo dataset.
# -----------------------------------------------------------------------------
CITIES = ["Bangalore", "Chennai", "Delhi", "Kolkata", "Mumbai"]

URBAN_CITIES = ["Mumbai", "Delhi", "Bangalore"]


# -----------------------------------------------------------------------------
# CUSTOMER SEGMENTS
# Exact strings from the suvroo Customer_Segment column.
# Used by rules engine and ML model as a behavioral feature.
#   New Visitor      — first-time or rare buyer, limited history
#   Occasional Shopper — buys periodically across categories
#   Frequent Buyer   — high purchase frequency, broadest category range
# -----------------------------------------------------------------------------
CUSTOMER_SEGMENTS = ["New Visitor", "Occasional Shopper", "Frequent Buyer"]


# -----------------------------------------------------------------------------
# DEVICE TYPES & PAYMENT METHODS
# Kept for build_user_profile() compatibility.
# ML model does not use these (not present in suvroo dataset).
# Rules engine may use device_type for secondary filtering.
# -----------------------------------------------------------------------------
DEVICE_TYPES = ["Desktop", "Mobile", "Tablet"]

PAYMENT_METHODS = [
    "Bank Transfer", "Cash on Delivery", "Credit Card",
    "Debit Card", "Digital Wallet",
]


# -----------------------------------------------------------------------------
# USER PROFILE SCHEMA
# Standard input format for ALL three modules.
#
#   apply_rules(user_profile)               → list[str]  eligible categories
#   find_product_path(user_profile,         → list[str]  path of categories
#                     target_category)
#   predict_product(user_profile,           → list[tuple[str, float]]
#                   candidates=None)           [(category, confidence), ...]
#   recommend_products(user_profile,        → list[tuple[str, float]]
#                      category, top_n=3)      [(product_name, score), ...]
#
# NOTE ON MEDIAN_SPEND: pass the user's Avg_Order_Value in ₹ (e.g. 2500.0).
#                       Use raw amount — not normalised.
# NOTE ON CUSTOMER_SEGMENT: optional — defaults to "Occasional Shopper".
#                            Pass when known for better rule/ML accuracy.
# -----------------------------------------------------------------------------

def build_user_profile(
    age:              int,
    gender:           str,
    city:             str,
    median_spend:     float,
    device_type:      str = "Mobile",
    payment_method:   str = "Credit Card",
    customer_segment: str = "Occasional Shopper",
) -> dict:
    """
    Build and validate a user profile dict.
    Raises ValueError if any field contains an unrecognised value.

    Parameters
    ----------
    age              : int, 18-60 (suvroo age range)
    gender           : one of GENDERS
    city             : one of CITIES (Indian cities)
    median_spend     : float, user's Avg_Order_Value in ₹ (500–5000 range)
    device_type      : one of DEVICE_TYPES (optional, not used by ML)
    payment_method   : one of PAYMENT_METHODS (optional, not used by ML)
    customer_segment : one of CUSTOMER_SEGMENTS (default: Occasional Shopper)

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
    if customer_segment not in CUSTOMER_SEGMENTS:
        raise ValueError(f"customer_segment must be one of {CUSTOMER_SEGMENTS}")

    return {
        "age":              int(age),
        "gender":           gender,
        "city":             city,
        "median_spend":     float(median_spend),
        "device_type":      device_type,
        "payment_method":   payment_method,
        "customer_segment": customer_segment,
        "age_group":        get_age_group(age),
        "price_range":      get_price_range(median_spend),
    }


# -----------------------------------------------------------------------------
# SAMPLE USER PROFILES
# Use these to test your module during development.
# Member 4 will use these for the final integration demo.
# Updated to reflect Indian cities and ₹ spend range from suvroo dataset.
#
# Profile          | Age | Gender | City      | Spend  | Tier     | Segment
# budget_browser   |  22 | Female | Chennai   |  ₹800  | Low      | New Visitor
# beauty_enthusiast|  30 | Female | Mumbai    | ₹1800  | Mid-Low  | Frequent Buyer
# fashion_fan      |  27 | Male   | Delhi     | ₹2500  | Mid-Low  | Occasional
# fitness_guy      |  35 | Male   | Bangalore | ₹3200  | Mid-High | Frequent Buyer
# tech_spender     |  42 | Male   | Mumbai    | ₹4500  | High     | Frequent Buyer
# -----------------------------------------------------------------------------

SAMPLE_PROFILES = {

    "budget_browser": build_user_profile(
        age              = 22,
        gender           = "Female",
        city             = "Chennai",
        median_spend     = 800.0,       # Low tier → Books / Beauty
        device_type      = "Mobile",
        payment_method   = "Credit Card",
        customer_segment = "New Visitor",
    ),

    "beauty_enthusiast": build_user_profile(
        age              = 30,
        gender           = "Female",
        city             = "Mumbai",
        median_spend     = 1800.0,      # Mid-Low → Beauty / Fashion
        device_type      = "Mobile",
        payment_method   = "Credit Card",
        customer_segment = "Frequent Buyer",
    ),

    "fashion_fan": build_user_profile(
        age              = 27,
        gender           = "Male",
        city             = "Delhi",
        median_spend     = 2500.0,      # Mid-Low → Fashion / Sports
        device_type      = "Mobile",
        payment_method   = "Debit Card",
        customer_segment = "Occasional Shopper",
    ),

    "fitness_guy": build_user_profile(
        age              = 35,
        gender           = "Male",
        city             = "Bangalore",
        median_spend     = 3200.0,      # Mid-High → Sports / Electronics
        device_type      = "Desktop",
        payment_method   = "Credit Card",
        customer_segment = "Frequent Buyer",
    ),

    "tech_spender": build_user_profile(
        age              = 42,
        gender           = "Male",
        city             = "Mumbai",
        median_spend     = 4500.0,      # High tier → Electronics
        device_type      = "Desktop",
        payment_method   = "Credit Card",
        customer_segment = "Frequent Buyer",
    ),
}
