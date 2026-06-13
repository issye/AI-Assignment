"""
Rules Engine Module for the Smart Product Recommendation System (CIC6314).
Handles initial category gating and resolves the customer cold-start problem.
"""

import pickle
import logging
from src.constants import (
    PRODUCT_CATEGORIES,
    PRICE_RANGES,
    CUSTOMER_SEGMENTS
)

# Set up clean execution logging
logger = logging.getLogger(__name__)

def apply_rules(user_profile: dict) -> list[str]:
    """
    Evaluates a user profile against 10 behavioral and spend-based heuristics to 
    determine a filtered list of eligible product recommendation categories.
    
    Args:
        user_profile (dict): Structured behavioral profile containing financial
                             and transactional metrics.
                             
    Returns:
        list[str]: Filtered categories sorted in strict compliance with 
                   the standard PRODUCT_CATEGORIES order array.
    """
    eligible = set()
    price_range = user_profile.get('price_range')

    # SPEND TIER RULES (1–4) 
    # Limits risk exposure by presenting budget-appropriate inventories.
    
    # Rule 1: Low spenders → affordable, high-volume categories
    if price_range == 'Low':
        eligible.update({'Home Decor', 'Stationery & Craft', 'Seasonal & Gifts'})

    # Rule 2: Mid-Low spenders → mid-range consumer categories
    elif price_range == 'Mid-Low':
        eligible.update({'Home Decor', 'Kitchen & Dining', 
                         'Seasonal & Gifts', 'Fashion & Accessories'})

    # Rule 3: Mid-High spenders → higher-value discretionary categories
    elif price_range == 'Mid-High':
        eligible.update({'Kitchen & Dining', 'Home Decor', 
                         'Toys & Games', 'Garden & Outdoor'})

    # Rule 4: High spenders → unlock complete multi-tier inventory portfolio
    elif price_range == 'High':
        eligible.update(set(PRODUCT_CATEGORIES))

    # BEHAVIOURAL RULES (5–10)
    # Dynamically scales exploratory or defensive filters based on user loyalty.

    # Rule 5: Primary core retention loop preservation
    # Rationale: A customer's verified historical primary interest should never be omitted.
    fav_cat = user_profile.get('favourite_category')
    if fav_cat:
        eligible.add(fav_cat)

    # Rule 6: Broad diversified customer exploration profile
    # Rationale: Cross-category shoppers who explore widely benefit from an open inventory landscape.
    purchased_cats = user_profile.get('purchased_categories', [])
    if len(purchased_cats) >= 3:
        eligible.update(set(PRODUCT_CATEGORIES))

    # Rule 7: Defensive focused customer specialization containment
    # Rationale: Avoid choice paralysis for narrow shoppers; introduce exactly one 
    # highly co-purchased neighboring category computed via Alternating Least Squares (ALS).
    if len(purchased_cats) == 1:
        fav = user_profile.get('favourite_category')
        if fav:
            eligible = {fav}
            try:
                with open('models/category_similarity.pkl', 'rb') as f:
                    cat_sim = pickle.load(f)
                # Eliminate self-similarity and extract top-ranked neighbor vector index
                neighbours = cat_sim[fav].drop(fav).sort_values(ascending=False)
                eligible.add(neighbours.index[0])
            except Exception as e:
                logger.warning(f"Category similarity model failure ({e}). Deploying static fail-safes.")
                eligible.update({'Home Decor', 'Seasonal & Gifts'})  # Academic fallback spec

    # Rule 8: Inactivity churn countermeasure & re-engagement
    # Rationale: High-margin impulse items like seasonal and food gifts drive re-activation hooks.
    if user_profile.get('recency_days', 0) > 90:
        eligible.update({'Seasonal & Gifts', 'Food & Confectionery'})

    # Rule 9: Power shopper maximum discovery expansion
    # Rationale: High-frequency buyers maintain complex search habits; suppress filters completely.
    if user_profile.get('customer_segment') == 'Frequent':
        eligible.update(set(PRODUCT_CATEGORIES))

    # Rule 10: Strict Cold-Start containment constraint
    # Rationale: Lacking transactional data, minimize error boundaries by funneling 
    # user entry pathways exclusively into top baseline categories. Overrides all previous rules.
    if user_profile.get('customer_segment') == 'New':
        eligible = {'Home Decor', 'Seasonal & Gifts', 'Kitchen & Dining'}

    # STRUCTURAL PIPELINE PROTECTIONS 
    # Safeguard downstream pipeline architectures against empty candidate sets.
    if not eligible:
        eligible = set(PRODUCT_CATEGORIES)

    # Output projection preserves deterministic canonical list order arrays 
    return [cat for cat in PRODUCT_CATEGORIES if cat in eligible]