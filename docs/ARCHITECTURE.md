# System Architecture — CIC6314 Smart Product Recommendation System

> **Dataset:** UCI Online Retail (real UK transactions, Dec 2010 – Dec 2011)
> **Approach:** ALS Collaborative Filtering + Random Forest Blend + Rules + BFS Search

---

## Full Pipeline

```
User Profile (build_user_profile)
        │
        ▼
┌─────────────────────────────────┐
│  Rules Engine  (Member 2)       │
│  apply_rules(profile)           │
│  → eligible: list[str]          │
│                                 │
│  10 behavioural rules based on  │
│  spend tier, purchase history,  │
│  recency, and segment           │
└────────────┬────────────────────┘
             │
             ├── purchase_history EMPTY (new customer)
             │         │
             │         ▼
             │   ┌─────────────────────────────────┐
             │   │  Search Module  (Member 1)       │
             │   │  find_popular_categories()       │
             │   │  → [(category, buyer_count), ...]│
             │   └─────────────┬───────────────────┘
             │                 │
             │                 ▼
             │   ┌─────────────────────────────────┐
             │   │  Integration  (Member 4)         │
             │   │  get_popular_products()          │
             │   │  recommendation_type = "popular" │
             │   └─────────────────────────────────┘
             │
             └── purchase_history NON-EMPTY (returning customer)
                       │
                       ▼
               ┌─────────────────────────────────┐
               │  Search Module  (Member 1)       │
               │  find_reachable_categories()     │
               │  BFS on 8×8 category graph       │
               │  → shortlist: list[str]          │
               └──────────────┬──────────────────┘
                              │
                              ▼
               ┌─────────────────────────────────┐
               │  ML Module  (Member 3)           │
               │  predict_product(profile,        │
               │    candidates=shortlist)         │
               │                                 │
               │  RF Blend: Model A (context)    │
               │          + Model B (history)    │
               │  → [(category, score), ...]     │
               └──────────────┬──────────────────┘
                              │  top 3 categories
                              ▼
               ┌─────────────────────────────────┐
               │  ML Module  (Member 3)           │
               │  recommend_products(profile,     │
               │    category, top_n=3) × 3        │
               │                                 │
               │  ALS item similarity lookup      │
               │  → [{"category","product",       │
               │       "price","score"}, ...]     │
               └──────────────┬──────────────────┘
                              │
                              ▼
               ┌─────────────────────────────────┐
               │  Integration  (Member 4)         │
               │  recommend(profile) → dict       │
               └─────────────────────────────────┘
```

---

## Output Schema

```python
# Personalised (returning user) — scores are RF probabilities (0–1)
{
    "recommendation_type": "personalised",
    "top_3_categories": [
        ("Home Decor",       0.61),
        ("Kitchen & Dining", 0.44),
        ("Food & Confectionery", 0.38),
    ],
    "recommended_products": [
        {"category": "Home Decor", "product": "WHITE METAL LANTERN",
         "price": 3.95, "score": 0.4231},
        # ... 8 more items (3 per category)
    ],
    "eligible": ["Home Decor", "Kitchen & Dining", ...],
}

# Popular (cold-start / new user) — scores are integer buyer counts
{
    "recommendation_type": "popular",
    "top_3_categories": [
        ("Home Decor",       8420),
        ("Seasonal & Gifts", 5310),
        ("Kitchen & Dining", 4190),
    ],
    "recommended_products": [
        {"category": "Home Decor", "product": "WHITE METAL LANTERN",
         "price": 3.95, "score": 847},
        # ... 8 more items
    ],
    "eligible": ["Home Decor", "Seasonal & Gifts", "Kitchen & Dining"],
}
```

---

## ML Module — Training Architecture

### ALS (Alternating Least Squares)

Trained on the full 4,338 × 3,665 binary user-item matrix.

```
User-Item matrix  →  ALS factorisation  →  Item factors (3665 × 50)
(4338 × 3665)       50 latent factors      Cosine similarity
                    50 iterations               ↓
                    regularization=1.0     Item similarity matrix (3665 × 3665)
                                               saved: similarity_matrix.pkl

                                          Category similarity matrix (8 × 8)
                                               saved: category_similarity.pkl
```

Why ALS instead of raw cosine: ALS compresses 4,338 customer dimensions to 50 latent factors, making similarity computation 86× cheaper and generalising better on sparse data (1.54% density).

### Random Forest Blend

```
Temporal split: features from < 2011-11-01, labels from >= 2011-11-01
Customer split: 80% train, 20% test (held-out evaluation)

Training data (df_train_ml):
  one row per customer
  features = pre-November stats
  labels   = which categories bought in November-December

Model A (Context)        Model B (History)
  3 features               15 features
  segment_enc              n_purchases
  price_enc                n_categories
  month                    recency_days
                           avg_order_value
  MultiOutputClassifier    segment_enc
  RandomForest(100 trees)  price_enc
  Works for ALL users      favourite_category (9 one-hot columns)
  incl. cold-start
                           MultiOutputClassifier
                           RandomForest(100 trees)
                           Only for returning users

At inference:
  confidence = 1 - 1/(1 + n_purchases)
  score = (1-conf) × Model_A_score + conf × Model_B_score
```

---

## Data Flow — Training vs Inference

```
TRAINING (run once, offline)

Online Retail.xlsx
    │
    ▼
notebooks/ml_model.ipynb
    ├── Section 4:  user-item matrix (4338×3665)
    ├── Section 5:  category mapping, customer features, product catalogue
    ├── Section 6a: ALS → similarity_matrix.pkl, category_similarity.pkl
    ├── Section 6b: temporal split, supervised training rows
    ├── Train/Test: 80/20 customer split
    ├── Section 6c: Model A → model_context.pkl
    ├── Section 6d: Model B → model_history.pkl
    ├── Section 6e: Evaluation on 20% held-out
    ├── Section 7:  HR@K evaluation on 20% held-out
    └── Section 8:  Save 8 pkl artefacts to models/


INFERENCE (per user request)

build_user_profile()
    │
    ├── loads: model_context.pkl, model_history.pkl  (predict_product)
    ├── loads: similarity_matrix.pkl                 (recommend_products)
    ├── loads: product_catalogue.pkl                 (both functions)
    └── loads: category_similarity.pkl               (search module BFS)
```

---

## Saved Artefacts

| File | Size | Generated by | Used by |
|---|---|---|---|
| `models/als_model.pkl` | ~1.6 MB | ml_model.ipynb | Internal reference |
| `models/similarity_matrix.pkl` | ~54 MB | ml_model.ipynb | `recommend_products()` |
| `models/category_similarity.pkl` | <1 MB | ml_model.ipynb | `find_reachable_categories()` |
| `models/model_context.pkl` | <1 MB | ml_model.ipynb | `predict_product()` Model A |
| `models/model_history.pkl` | <1 MB | ml_model.ipynb | `predict_product()` Model B |
| `models/product_catalogue.pkl` | ~1 MB | ml_model.ipynb | Both ML functions + Member 4 |
| `models/customer_features.pkl` | ~1 MB | ml_model.ipynb | Reference |
| `models/encoder_category.pkl` | <1 MB | ml_model.ipynb | Reference |

All pkl files are gitignored. Regenerate by running `Kernel → Restart & Run All` on `notebooks/ml_model.ipynb`.

---

## Module Status

| Module | Owner | Branch | Status |
|---|---|---|---|
| `src/constants.py` | Member 3 | `feature/ml` | ✅ Complete |
| `notebooks/ml_model.ipynb` | Member 3 | `feature/ml` | ✅ Written — needs `Restart & Run All` |
| `src/rules_engine.py` | Member 2 | `feature/rules` | ❌ Not yet implemented |
| `src/search_module.py` | Member 1 | `feature/search` | ❌ Not yet implemented |
| `notebooks/career_recommender.ipynb` | Member 4 | `feature/integration` | ❌ Not yet implemented |

---

## Product Categories (8)

Derived from keyword matching on product descriptions in Online Retail.xlsx.

| Category | Sample keywords | Typical price |
|---|---|---|
| Home Decor | LANTERN, FRAME, CANDLE, VASE, MIRROR, SIGN | £2.48 avg |
| Kitchen & Dining | MUG, CUP, PLATE, BOWL, TEAPOT, JUG | £3.12 avg |
| Seasonal & Gifts | CHRISTMAS, XMAS, BIRTHDAY, GIFT, WRAP | £2.91 avg |
| Toys & Games | TOY, GAME, PUZZLE, DOLL, BEAR, PLAY | £3.45 avg |
| Stationery & Craft | PEN, CARD, NOTEBOOK, CRAFT, PAPER | £1.87 avg |
| Fashion & Accessories | BAG, SCARF, JEWEL, NECKLACE, PURSE | £4.23 avg |
| Garden & Outdoor | GARDEN, PLANT, OUTDOOR, WATERING, POT | £2.74 avg |
| Food & Confectionery | FOOD, CHOCOLATE, SWEET, BISCUIT, TEA | £2.15 avg |

Default category (unmatched descriptions): Home Decor

---

## Customer Segments

| Segment | Criteria | Count | Share |
|---|---|---|---|
| New | ≤ 2 invoices | 2,328 | 53.7% |
| Occasional | 3–10 invoices | 1,673 | 38.6% |
| Frequent | > 10 invoices | 337 | 7.8% |

## Spend Tiers (£, from real data quartiles)

| Tier | Range | Quartile |
|---|---|---|
| Low | < £178.62 | Q1 |
| Mid-Low | £178.62 – £293.90 | Q1–Q2 |
| Mid-High | £293.90 – £430.11 | Q2–Q3 |
| High | ≥ £430.11 | Q3+ |

Note: Values appear high because the dataset includes wholesale buyers. The quartile split correctly reflects the actual distribution.
