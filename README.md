# CIC6314 - Smart Product Recommendation System

> **Course:** CIC6314 Artificial Intelligence | **Lecturer:** Prabha Kumaresan
> **Session:** March/April 2026 | **Deadline:** 27 June 2026, 7 PM

An intelligent product recommendation system that combines rule-based reasoning, A* graph search, and machine learning to suggest relevant products to customers based on their purchase history and spending behaviour. Built on the UCI Online Retail dataset - real transaction data from a UK gift and novelty retailer.

---

## Team Members

| Student ID | Name | Branch | Module |
|---|---|---|---|
| 1231303279 | Issye Lailiyah | `feature/search` | Search Algorithm (A*) + GitHub |
| 1211106853 | Thineshraj A/L Chandrasegaran | `feature/rules` | Knowledge Representation & Logic |
| 1211112109 | Wan Arief Najwan | `feature/ml` | Machine Learning Model |
|1231302416 | MUHAMMAD ADAM HADZIQ | `feature/integration` | Integration & System Design |

---

## Dataset

**File:** `data/online+retail/Online Retail.xlsx`
**Source:** UCI Machine Learning Repository - Online Retail Dataset
**Transactions:** 541,909 rows | **After cleaning:** 397,884 rows
**Customers:** 4,338 unique | **Products:** 3,665 unique | **Categories:** 8
**Period:** December 2010 to December 2011
**Country:** Primarily United Kingdom, currency in GBP

The dataset contains real invoices from a UK-based online retailer that sells gift and novelty items. Each row is one line item from an invoice, with columns for InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, and Country.

### Product Categories

Products are mapped to 8 categories based on keyword matching on the Description field. The mapping is stored in `data/online+retail/product_categories.csv`.

| Category | Description |
|---|---|
| Home Decor | Decorative items, wall art, frames, candles |
| Kitchen & Dining | Kitchenware, mugs, storage, cooking tools |
| Seasonal & Gifts | Christmas, holiday, gift wrap, seasonal items |
| Toys & Games | Children's toys, games, puzzles |
| Stationery & Craft | Notebooks, pens, craft supplies, paper goods |
| Fashion & Accessories | Bags, jewellery, clothing accessories |
| Garden & Outdoor | Planters, outdoor decor, garden tools |
| Food & Confectionery | Food items, sweets, confectionery |

---

## System Architecture

```
Customer Profile
      |
      v
+---------------------+
|   Rules Engine      |  <- (src/rules_engine.py)
|   apply_rules()     |     10 rules: spend tier gating + behavioural signals
|                     |     Output: list of eligible categories
+----------+----------+
           |
           |  Has purchase history?
     ------+------
     |            |
    YES           NO
     |            |
     v            v
+----------+   +--------------------+
| A* Search|   | Popularity Ranking |  <-  (src/search_module.py)
|find_reach|   | find_popular_cat() |
|able_cats()|  +--------------------+
|           |          |
| Traverses |     Top 3 popular
| ALS graph |     categories
| f = g + h |
+-----+-----+
      |
      v
+---------------------+
|   ML Model          |  <- (notebooks/ml_model.ipynb)
|   predict_product() |     RF blend: context model + history model
|                     |     Output: categories ranked by purchase probability
+----------+----------+
           |
           v
+---------------------+
|   ALS Item Lookup   |  <- 
| recommend_products()|     Item-item cosine similarity
|                     |     3 products per top category
+----------+----------+
           |
           v
+---------------------+
|  Integration Layer  |  <-  (smart_product_recommendation.ipynb)
|   recommend()       |     Single entry point, routes both paths
|                     |     Returns 9 products across top 3 categories
+---------------------+
```

---

## Repository Structure

```
cic6314-smart-product-recommendation/
|
|-- smart_product_recommendation.ipynb   # MAIN NOTEBOOK - full pipeline integration
|                                        # Member 4: recommend() orchestrator, demo,
|                                        # schema verification, pipeline trace,
|                                        # A* justification (Section 8)
|
|-- notebooks/
|   |-- search_demo.ipynb                #  A* demo with step-by-step f/g/h
|   |                                    # trace, graph visualisation, integration demo
|   |-- ml_model.ipynb                   #  ALS + RF training pipeline,
|   |                                    # evaluation (F1, HR@K), saves 8 pkl files
|   |-- rules_demo.ipynb                 # rules engine demo and test cases
|   |-- fig_rules_heatmap.png            # Generated heatmap from rules demo
|   `-- test_rules.py                    # Unit tests for the rules engine
|
|-- src/
|   |-- constants.py                     # Shared constants - ALL modules import from here
|   |                                    # PRODUCT_CATEGORIES, SAMPLE_PROFILES,
|   |                                    # SPEND_THRESHOLDS, build_user_profile()
|   |-- search_module.py                 # A* search on ALS similarity graph (Member 1)
|   |                                    # find_reachable_categories() - personalised path
|   |                                    # find_popular_categories()   - cold-start path
|   `-- rules_engine.py                  # 10-rule inference engine (Member 2)
|                                        # apply_rules() - spend tier + behavioural gating
|
|-- data/
|   `-- online+retail/
|       |-- Online Retail.xlsx           # Raw dataset (UCI, 541,909 rows)
|       |-- product_categories.csv       # StockCode to category mapping (3,665 products)
|       `-- product_categories_README.md # How the keyword-based category mapping was built
|
|-- models/                              # Pre-trained artefacts (generated by ml_model.ipynb)
|   |-- als_model.pkl                    # Trained ALS model (50 latent factors, 50 iterations)
|   |-- similarity_matrix.pkl            # 3,665 x 3,665 item-item cosine similarity matrix
|   |-- category_similarity.pkl          # 8 x 8 category-category ALS similarity matrix
|   |-- model_context.pkl                # RF context model (3 features, all customers)
|   |-- model_history.pkl                # RF history model (15 features, returning customers)
|   |-- product_catalogue.pkl            # Products with category, avg price, popularity rank
|   |-- customer_features.pkl            # Customer-level features used during training
|   `-- encoder_category.pkl             # Label encoder for category columns
|
|-- poster/
|   `-- poster.pdf                       # Presentation poster (Member 4)
|
|-- docs/
|   |-- ARCHITECTURE.md                  # System architecture notes
|   `-- HANDOFF.md                       # Cross-member handoff documentation
|
|-- agent_handoff/                       # Handoff notes between team members
|
|-- scripts/                             # Utility scripts (not part of submission)
|
|-- requirements.txt                     # Python dependencies
`-- README.md                            # This file
```

---

## Module Interfaces

All members import from `src/constants.py`. Never hardcode category names, spend thresholds, or field values anywhere else.

```python
from src.constants import (
    PRODUCT_CATEGORIES,    # list of 8 category strings
    SAMPLE_PROFILES,       # 5 test profiles built from real CustomerIDs
    build_user_profile,    # builds and validates a profile dict
    SPEND_THRESHOLDS,      # Low/Mid-Low/Mid-High/High quartile boundaries (GBP)
    get_price_range,       # maps avg_order_value -> price tier string
    get_customer_segment,  # maps invoice count -> New/Occasional/Frequent
)
```

### User Profile Schema

```python
profile = build_user_profile(
    customer_id      = "17850",              # CustomerID from Online Retail
    purchase_history = ["85123A", "71053"],  # StockCodes of products bought
    avg_order_value  = 293.90,               # mean basket value in GBP
    total_invoices   = 8,                    # number of distinct invoices
    recency_days     = 25,                   # days since last order
)
# Derived fields added automatically:
# price_range          -> "Mid-High"   (from avg_order_value quartile)
# customer_segment     -> "Occasional" (from total_invoices)
# favourite_category   -> most frequently bought category (from purchase_history)
# purchased_categories -> unique categories bought, in order
```

Note: there are no demographics fields (no age, gender, city) because the UCI Online Retail dataset does not include them. Customer identity is derived entirely from purchase behaviour.

### Function Signatures

```python
# Member 2 - Rules Engine
from src.rules_engine import apply_rules
apply_rules(user_profile)
# Returns: list[str] - eligible categories from PRODUCT_CATEGORIES

# Member 1 - A* Search (personalised path)
from src.search_module import find_reachable_categories, find_popular_categories
find_reachable_categories(user_profile, eligible, max_hops=2, threshold=0.06)
# Returns: list[str] - eligible categories ordered by A* f-score (lowest f = most optimal)

find_popular_categories(eligible, price_range, top_n=3)
# Returns: list[tuple[str, int]] - [(category, buyer_count), ...] sorted descending

# Member 3 - ML scoring
predict_product(user_profile, candidates=None)
# Returns: list[tuple[str, float]] - [(category, probability), ...] sorted descending

recommend_products(user_profile, category, top_n=3)
# Returns: list[dict] - [{'category', 'product', 'price', 'score'}, ...]

# Member 4 - Integration entry point
recommend(user_profile)
# Returns dict:
#   recommendation_type  -> 'personalised' or 'popular'
#   top_3_categories     -> [(category, score), ...]          3 items
#   recommended_products -> [{'category','product','price','score'}, ...] 9 items
#   eligible             -> [category, ...]
```

---

## How the A* Search Works

The search module models the 8 product categories as nodes in a weighted graph. Two categories are connected by an edge if their ALS-derived cosine similarity is at least 0.06. This threshold was chosen because Fashion & Accessories has a maximum similarity of 0.059 to all other categories, making it correctly isolated unless explicitly unlocked by the rules engine.

A* starts from the customer's `favourite_category` and finds all eligible categories reachable within 2 hops. The cost function is:

```
g(n) = cumulative edge cost = sum of (1 - similarity) along the path
h(n) = heuristic            = 1 - normalised_popularity(n)
f(n) = g(n) + h(n)          = A* minimises this value

Admissibility: normalised_popularity is in [0, 1], so h is always in [0, 1]
               and never overestimates the true remaining cost. A* is guaranteed
               to return the minimum-cost ordering.
```

Categories with lower f-score are ranked first because they are both close to the customer's favourite (low g) and globally popular (low h). Results are passed to the RF model for final scoring.

---

## How the ML Model Works

Two Random Forest classifiers are trained and blended with a confidence weight.

**Model A (Context Model)** - 3 features: customer segment (encoded 0/1/2), price range (encoded 0/1/2/3), and current month. Works for all customers.

**Model B (History Model)** - 15 features: number of purchases, unique categories, recency in days, average order value, segment (encoded), price range (encoded), and a 9-element one-hot vector for favourite category. Only used for returning customers.

Blend formula:

```
confidence = 1 - 1 / (1 + n_purchases)
final_score = (1 - confidence) x context_score + confidence x history_score

n = 0  -> confidence = 0.00 -> 100% context model
n = 1  -> confidence = 0.50 -> 50% context + 50% history
n = 5  -> confidence = 0.83 -> 17% context + 83% history
n = 14 -> confidence = 0.93 -> 7%  context + 93% history
```

Product recommendations use ALS item-item cosine similarity. The mean similarity between each candidate product and all items in the customer's purchase history is computed, and the top 3 products per category are returned.

**Evaluation results:**
- Model B macro-F1: 0.71 on held-out test set (temporal split, Nov-Dec 2011 as test)
- ALS Hit Rate@10: 0.4207 vs popularity baseline 0.2589

---

## How the Rules Engine Works

Ten rules gate the eligible product categories for each customer.

**Spend Tier Rules (Rules 1-4)** assign a base set of categories from the customer's average order value quartile:
- Rule 1: Low (below GBP 178.62) -> Home Decor, Stationery & Craft, Seasonal & Gifts
- Rule 2: Mid-Low (GBP 178.62 to 293.90) -> Home Decor, Kitchen & Dining, Seasonal & Gifts, Fashion & Accessories
- Rule 3: Mid-High (GBP 293.90 to 430.11) -> Kitchen & Dining, Home Decor, Toys & Games, Garden & Outdoor
- Rule 4: High (GBP 430.11 and above) -> all 8 categories

**Behavioural Rules (Rules 5-10)** adjust the eligible set based on purchase behaviour:
- Rule 5: Always include the customer's favourite category (retention signal)
- Rule 6: Customers who bought from 3 or more categories see all 8 categories (broad explorer)
- Rule 7: Customers who bought from only 1 category see just that category plus its nearest ALS neighbour (prevents choice paralysis for narrow shoppers)
- Rule 8: Customers inactive for 90 or more days get Seasonal & Gifts and Food & Confectionery added (re-engagement hook)
- Rule 9: Frequent segment customers (more than 10 invoices) see all 8 categories (power shoppers)
- Rule 10: New customers with no history are restricted to Home Decor, Seasonal & Gifts, Kitchen & Dining (cold-start containment - overrides all other rules)

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<team-lead-username>/cic6314-smart-product-recommendation.git
cd cic6314-smart-product-recommendation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install implicit    # ALS collaborative filtering
pip install networkx    # graph visualisation in search_demo
```

### 3. Run the ML training notebook first

Open `notebooks/ml_model.ipynb` and run all cells (Kernel -> Restart & Run All). This generates all 8 pkl files in the `models/` folder. The main notebook will not work without these.

### 4. Run the main integration notebook

Open `smart_product_recommendation.ipynb` and run all cells. This is the notebook that demonstrates the full pipeline and is the primary submission deliverable.

### 5. Run individual demo notebooks (optional)

- `notebooks/search_demo.ipynb` - A* step-by-step trace and graph visualisation
- `notebooks/rules_demo.ipynb` - Rules engine demo

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Submission-ready code only. Team lead merges here before submission. |
| `dev` | Shared integration branch. All features merge here first. |
| `feature/search` | Member 1 - A* search algorithm |
| `feature/rules` | Member 2 - knowledge rules and inference engine |
| `feature/ml` | Member 3 - machine learning model |
| `feature/integration` | Member 4 - full pipeline integration and poster |

---

## Daily Workflow

```bash
# Start of every session - sync first
git checkout dev && git pull origin dev
git checkout feature/your-branch
git merge dev

# Do your work, then commit
git add .
git commit -m "feat(module): short description"
git push origin feature/your-branch
```

---

## Commit Message Convention

| Prefix | When to use |
|---|---|
| `feat(search):` | A* search algorithm work |
| `feat(rules):` | Rules engine work |
| `feat(ml):` | ML model work |
| `feat(integration):` | Integration pipeline work |
| `fix(module):` | Bug fix |
| `docs:` | Documentation updates |
| `data:` | Dataset changes |
| `style:` | Formatting or cleanup only |

---

## Dependencies

```
python >= 3.9
jupyter             # notebook environment
scikit-learn        # RandomForestClassifier, MultiOutputClassifier, train_test_split
pandas              # data manipulation
numpy               # array operations
matplotlib          # plots and charts
seaborn             # heatmaps and styled plots
joblib              # model serialisation
openpyxl            # reading .xlsx files
implicit            # ALS collaborative filtering (install separately)
networkx            # graph construction and visualisation in search_demo.ipynb
```
