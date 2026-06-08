# Agent Handoff — ML Module (Member 3)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/ml`
**Dataset:** `data/online+retail/Online Retail.xlsx`
**Deadline:** 27 June 2026, 7 PM

---

## What This Module Does

Member 3 owns the machine learning core of the recommendation system. This means two things:

1. **Training** — running `notebooks/ml_model.ipynb` to produce trained model artefacts (`.pkl` files) that all other members load at inference time.
2. **Inference functions** — `predict_product()` and `recommend_products()`, which are the two functions every other module depends on for personalised recommendations.

Every recommendation the system produces either passes through `predict_product()` to rank categories, or through `recommend_products()` to pick specific products. Nothing in the system produces a final recommendation without calling one of these two functions.

---

## How This Module Fits Into the Full Pipeline

The system has four components working in sequence. ML sits in the third position:

```
User Profile
     │
     ▼
Rules Engine (Member 2)
  apply_rules(profile) → eligible: list of up to 8 categories
     │
     ▼
Search Module (Member 1)
  find_reachable_categories(profile, eligible) → shortlist: narrowed categories
     │  (or for new users: find_popular_categories → bypass ML entirely)
     ▼
ML Module (Member 3)  ◄─── YOU ARE HERE
  predict_product(profile, candidates=shortlist) → [(category, score), ...]
  recommend_products(profile, category) → [{"category","product","price","score"}, ...]
     │
     ▼
Integration (Member 4)
  recommend(profile) → final output dict
```

**Key point:** The ML module never sees the raw transaction data at inference time. It only sees the user profile dict and the shortlist from the search module. All the heavy computation is done once at training time and stored in pkl files.

---

## Architecture of the ML Module

Two completely separate model types serve two separate purposes:

### Model 1 — ALS (Alternating Least Squares)
**Purpose:** Score specific *products* within a category  
**Used by:** `recommend_products()`  
**How it works:**
- Trained on the full user-item binary matrix (4,338 customers × 3,665 products)
- ALS compresses this into 50-dimensional "factor" vectors for every product
- Two products with similar factor vectors are likely bought by the same type of customer
- At inference, a customer's owned items are used as seed — products most similar to those seeds get recommended
- Falls back to global popularity ranking if the customer has no purchase history

### Model 2 — Random Forest Blend (Models A + B)
**Purpose:** Score *categories* — predict which product types a customer will buy next  
**Used by:** `predict_product()`  
**How it works:**
- **Model A (Context):** 3 features — customer segment, price range, current month. Works for *all* users including brand new ones with no history.
- **Model B (History):** 15 features — purchase count, recency, spend, favourite category. Only meaningful for returning users.
- Both models output a probability (0–1) that the customer will buy from each of the 8 categories
- The outputs are blended by purchase confidence: `confidence = 1 - 1/(1 + n_purchases)`
  - New user (0 purchases) → 100% Model A
  - 5 purchases → 83% Model B, 17% Model A
  - 10+ purchases → ~91% Model B, ~9% Model A

---

## Training Data Construction

The RF models are trained on supervised examples built from the temporal split:

- **Features** come from each customer's transaction history **before November 2011**
- **Labels** are which categories they actually bought from in **November–December 2011**

This is called a temporal split. It mirrors real deployment: the model learns from the past and is judged on the future.

A customer-level 80/20 split is applied:
- **80% of returning customers** → training set for the RF models
- **20% of returning customers** → held out for honest evaluation
- **All cold-start customers** → training set (they have no pre-November history to evaluate against)

The ALS model is trained on the **full dataset** (no split needed — it learns purchase patterns, not future predictions).

---

## Evaluation Strategy

Three evaluation steps, each measuring something different:

| Section | What it measures | Data used |
|---|---|---|
| 6e: Classification report | Are the RF classifiers accurate? | 20% held-out customers |
| 7a: Category HR@3 | Does the top-3 category list include something correct? | 20% held-out customers |
| 7b: Product HR@K | Does the top-K product list include something the customer actually bought? | 20% held-out customers |

**HR@K (Hit Rate@K):** For each customer, the system recommends K items. If at least one is something the customer actually bought, that's a hit. HR@K is the fraction of customers who got at least one hit.

---

## Saved Artefacts (8 pkl files)

All saved to `models/` directory. All gitignored — regenerate by running `Kernel → Restart & Run All`.

| File | Size | Used by | Contents |
|---|---|---|---|
| `als_model.pkl` | ~1.6 MB | Internal | Trained ALS model object |
| `similarity_matrix.pkl` | ~54 MB | `recommend_products()` | 3,665 × 3,665 item cosine similarity (float32) |
| `category_similarity.pkl` | <1 MB | Member 1 search module | 8 × 8 category cosine similarity DataFrame |
| `model_context.pkl` | <1 MB | `predict_product()` | MultiOutputClassifier RF, 3 features |
| `model_history.pkl` | <1 MB | `predict_product()` | MultiOutputClassifier RF, 15 features |
| `product_catalogue.pkl` | ~1 MB | Both functions | StockCode, Description, category, avg_price, popularity_rank |
| `customer_features.pkl` | ~1 MB | Reference | Per-customer aggregated features |
| `encoder_category.pkl` | <1 MB | Reference | LabelEncoder for PRODUCT_CATEGORIES |

---

## Public Interface

### `predict_product(user_profile, candidates=None)`

Scores candidate categories for a user. Works for all users including cold-start.

```python
predict_product(user_profile: dict, candidates: list = None) -> list[tuple[str, float]]
```

- `candidates`: optional subset of `PRODUCT_CATEGORIES` (passed by search module). Defaults to all 8.
- Returns `[(category, score), ...]` sorted by score descending. Scores are RF probabilities (0–1).
- **Never raises ValueError** — cold-start users get context model scores.

### `recommend_products(user_profile, category, top_n=3)`

Recommends specific products within one category using ALS item similarity.

```python
recommend_products(user_profile: dict, category: str, top_n: int = 3) -> list[dict]
```

- Returns `[{"category": str, "product": str, "price": float, "score": float}, ...]`
- `price` is the average unit price in £, rounded to 2 decimal places
- `score` is cosine similarity (0–1) for returning users, or `0.0` for popularity fallback
- Never returns products the user already owns (filters `purchase_history`)

---

## Notebook Structure (10 Sections)

| Section | What happens |
|---|---|
| 1. Load Data | Read Online Retail.xlsx (397,884 rows) |
| 2. EDA | Explore distributions, missing values, cancellations |
| 3. Data Cleaning | Remove guest checkouts, cancellations, zero-price rows |
| 4. Preprocessing | Build binary user-item matrix (4,338 × 3,665) |
| 5. Feature Engineering | Category mapping via keywords, customer features, product catalogue |
| 6a. ALS | Train ALS, build item similarity matrix and category similarity matrix |
| 6b. Training Data | Temporal split, build supervised training rows |
| 6b (Train/Test Split) | Split customers 80/20 |
| 6c. Model A | Train context RF on 80% training customers |
| 6d. Model B | Train history RF on 80% returning customers |
| 6e. Evaluation | Classification report + confusion matrices on 20% held-out |
| 6f. Feature Importance | Charts showing which features drive each model |
| 7. Recommendation Evaluation | Category HR@3 and Product HR@K on 20% held-out |
| 8. Save Artefacts | Save 8 pkl files to models/ |
| 9. Inference Functions | Reload artefacts, define predict_product and recommend_products |
| 10. Demo + Trace | Run all 5 SAMPLE_PROFILES, step-by-step pipeline trace |

---

## Current Status

- `notebooks/ml_model.ipynb` — **fully written, NOT YET EXECUTED**
- Run `Kernel → Restart & Run All` to generate the pkl artefacts
- `src/constants.py` — complete and correct
- `models/` — empty until notebook is run

### Immediate Next Step

Run the notebook. After execution verify:
1. All 8 pkl files exist in `models/`
2. Section 6e classification report prints without error
3. Section 7b product HR@K completes (uses fast numpy, should be ~30–60 seconds)
4. Section 10 demo shows 3 categories × 3 products × 5 profiles = 45 product recommendations with prices
5. `new_customer` profile produces `recommendation_type = "popular"` (no ValueError)

---

## Rules

- `predict_product()` and `recommend_products()` signatures are frozen — Member 4 calls them
- Both functions must return values from `PRODUCT_CATEGORIES` only
- Never return items in `purchase_history`
- All constants import from `src/constants.py`
- Do not push — user pushes manually
- PRs must be draft: `gh pr create --draft`
