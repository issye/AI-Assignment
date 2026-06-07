# Agent Handoff — ML / CF Module (Member 3)
**Project:** CIC6314 Smart Product Recommendation System
**Branch:** `feature/ml`
**Dataset:** `data/online+retail/Online Retail.xlsx`
**Deadline:** 27 June 2026, 7 PM

---

## What Changed and Why

The project has **fully pivoted** from a binary purchase-probability classifier (suvroo)
to **item-based collaborative filtering (CF)** on the UCI Online Retail dataset.

| What changed | Old (suvroo) | New (Online Retail) |
|---|---|---|
| Dataset | Synthetic, 10K customers, Indian, ₹ | Real transactions, 4,338 customers, UK, £ |
| Approach | Binary ML classifier (Random Forest) | Item-based cosine similarity CF |
| Training pairs | 240K synthetic (customer×product) | Real user-item purchase matrix |
| Model artefact | 231 MB Random Forest pkl | ~50 MB similarity matrix (DataFrame) |
| Feature engineering | 29 features incl. demographics | User-item matrix (binary) only |
| Demographics | Age, gender, city used | Removed — not in dataset, not needed |
| Output scores | ML purchase probability | Cosine similarity (0–1) |
| Data leakage | category_browsed shortcut | No construction rule — real data |

---

## Your Responsibilities

1. Rewrite `notebooks/ml_model.ipynb` end-to-end (see sections below)
2. Rewrite `src/constants.py` (already done — pull from `main`)
3. Save 5 artefact files to `models/`
4. Generate `data/online+retail/product_categories.csv` (already done — pull from `main`)
5. Rewrite all 4 agent handoff documents (already done — pull from `main`)

---

## Notebook Structure (8 sections)

### Section 1 — Data Loading & Cleaning

```python
import pandas as pd, numpy as np, pickle, warnings
warnings.filterwarnings('ignore')

df = pd.read_excel('data/online+retail/Online Retail.xlsx')
df = df[df['CustomerID'].notna()]
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df['StockCode']   = df['StockCode'].astype(str)
df['CustomerID']  = df['CustomerID'].astype(int).astype(str)
df['Description'] = df['Description'].str.strip().str.upper()
df['line_total']  = df['Quantity'] * df['UnitPrice']

# Expected: ~397,884 rows | 4,338 customers | 3,665 products
print(f"Rows: {len(df):,} | Customers: {df.CustomerID.nunique():,} | Products: {df.StockCode.nunique():,}")
```

### Section 2 — Category Engineering

```python
CATEGORY_KEYWORDS = {
    'Home Decor':            ['LANTERN','FRAME','CANDLE','VASE','MIRROR','SIGN','CLOCK','LIGHT','HOLDER','WALL'],
    'Kitchen & Dining':      ['MUG','CUP','PLATE','BOWL','TEAPOT','JUG','KITCHEN','CAKE','SPOON','JAR'],
    'Seasonal & Gifts':      ['CHRISTMAS','XMAS','EASTER','HALLOWEEN','VALENTINE','BIRTHDAY','GIFT','WRAP'],
    'Toys & Games':          ['TOY','GAME','PUZZLE','DOLL','BEAR','PLAY','CHILDREN','KIDS'],
    'Stationery & Craft':    ['PEN','CARD','NOTEBOOK','CRAFT','PAPER','STAMP','STICKER','TAPE'],
    'Fashion & Accessories': ['BAG','SCARF','JEWEL','NECKLACE','BRACELET','PURSE','UMBRELLA','WALLET'],
    'Garden & Outdoor':      ['GARDEN','PLANT','OUTDOOR','WATERING','POT','BIRD','FLOWER'],
    'Food & Confectionery':  ['FOOD','CHOCOLATE','SWEET','BISCUIT','JAM','HONEY','TEA','COFFEE'],
}

def assign_category(desc):
    if not isinstance(desc, str): return 'Home Decor'
    d = desc.upper()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in d for kw in keywords):
            return cat
    return 'Home Decor'   # default

df['category'] = df['Description'].apply(assign_category)
print(df.groupby('category')['StockCode'].nunique().sort_values(ascending=False))
```

### Section 3 — Customer Feature Engineering

```python
REF_DATE = pd.Timestamp('2011-12-09')

basket = (df.groupby(['CustomerID','InvoiceNo'])['line_total']
            .sum().reset_index(name='basket_value'))

customer_features = df.groupby('CustomerID').agg(
    avg_order_value      = ('line_total',   'mean'),
    total_invoices       = ('InvoiceNo',    'nunique'),
    recency_days         = ('InvoiceDate',  lambda x: (REF_DATE - x.max()).days),
    favourite_category   = ('category',     lambda x: x.mode()[0]),
    purchased_categories = ('category',     lambda x: list(x.unique())),
    n_categories         = ('category',     'nunique'),
).reset_index()

def get_segment(n):
    if n <= 2:  return 'New'
    if n <= 10: return 'Occasional'
    return 'Frequent'

def get_price_range(v):
    if v < 178.62:  return 'Low'
    if v < 293.90:  return 'Mid-Low'
    if v < 430.11:  return 'Mid-High'
    return 'High'

customer_features['customer_segment'] = customer_features['total_invoices'].apply(get_segment)
customer_features['price_range']      = customer_features['avg_order_value'].apply(get_price_range)
print(customer_features.head())
```

### Section 4 — User-Item Matrix

```python
# Binary: 1 if customer bought product at least once, 0 otherwise
user_item = (df.groupby(['CustomerID','StockCode'])['Quantity']
               .sum()
               .unstack(fill_value=0)
               .clip(upper=1))

# Shape: ~4,338 customers × ~3,665 products
print(f"User-item matrix shape: {user_item.shape}")
print(f"Matrix density: {user_item.values.mean():.4f}")   # expect ~0.005–0.01
```

### Section 5 — Item-Item Similarity Matrix

```python
from sklearn.metrics.pairwise import cosine_similarity

# Item-item cosine similarity (transpose: items as rows)
item_sim_values = cosine_similarity(user_item.T)
item_sim_df = pd.DataFrame(
    item_sim_values,
    index   = user_item.columns,
    columns = user_item.columns,
)
print(f"Similarity matrix shape: {item_sim_df.shape}")
print(f"Diagonal (should be 1.0): {item_sim_df.values.diagonal().mean():.4f}")

# ── Category-category similarity ──────────────────────────────────────────
# Aggregate item-level similarity to category level
# Used by Member 1's BFS search

stock_to_cat = df[['StockCode','category']].drop_duplicates().set_index('StockCode')['category']

cat_sim_rows = {}
for cat_a in CATEGORY_KEYWORDS:
    items_a = stock_to_cat[stock_to_cat == cat_a].index.intersection(item_sim_df.index)
    row = {}
    for cat_b in CATEGORY_KEYWORDS:
        items_b = stock_to_cat[stock_to_cat == cat_b].index.intersection(item_sim_df.columns)
        if len(items_a) == 0 or len(items_b) == 0:
            row[cat_b] = 0.0
        else:
            row[cat_b] = float(item_sim_df.loc[items_a, items_b].values.mean())
    cat_sim_rows[cat_a] = row

category_sim_df = pd.DataFrame(cat_sim_rows).T
print("Category similarity matrix:")
print(category_sim_df.round(3))
```

### Section 6 — Global Popularity Table

```python
# Unique buyers per product — used for cold-start recommendations
popularity = (df.groupby(['StockCode','Description','category'])['CustomerID']
                .nunique()
                .reset_index(name='popularity_rank'))

avg_price = (df.groupby('StockCode')['UnitPrice']
               .mean()
               .reset_index(name='avg_price'))

product_catalogue = popularity.merge(avg_price, on='StockCode')
product_catalogue = product_catalogue.sort_values('popularity_rank', ascending=False)
print(f"Product catalogue: {len(product_catalogue)} products")
print(product_catalogue.head(10))
```

### Section 7 — Evaluation: Hit Rate@K

```python
# Hold out each customer's last purchase, check if it appears in top-K recs

def get_top_k_for_customer(customer_id, k=5):
    bought = user_item.loc[customer_id]
    bought_items = bought[bought == 1].index.tolist()
    if len(bought_items) < 2:
        return []
    # Score all unowned items
    scores = item_sim_df.loc[:, bought_items].mean(axis=1)
    scores = scores[~scores.index.isin(bought_items)]
    return scores.nlargest(k).index.tolist()

hits_at_1, hits_at_3, hits_at_5, total = 0, 0, 0, 0

for cid in customer_features[customer_features['total_invoices'] >= 3]['CustomerID']:
    if cid not in user_item.index:
        continue
    # Hold out last-purchased item
    last_item = df[df['CustomerID']==cid].sort_values('InvoiceDate').iloc[-1]['StockCode']
    # Remove last item from history temporarily
    temp_bought = user_item.loc[cid].copy()
    temp_bought[last_item] = 0
    bought_items = temp_bought[temp_bought == 1].index.tolist()
    if not bought_items:
        continue
    scores = item_sim_df.loc[:, bought_items].mean(axis=1)
    scores = scores[scores.index != last_item]
    top5 = scores.nlargest(5).index.tolist()
    if last_item in top5[:1]: hits_at_1 += 1
    if last_item in top5[:3]: hits_at_3 += 1
    if last_item in top5[:5]: hits_at_5 += 1
    total += 1

print(f"Hit Rate@1: {hits_at_1/total:.4f}")
print(f"Hit Rate@3: {hits_at_3/total:.4f}")
print(f"Hit Rate@5: {hits_at_5/total:.4f}")
print(f"Total evaluated customers: {total}")
```

### Section 8 — Save Artefacts

```python
import os
os.makedirs('models', exist_ok=True)

pickle.dump(item_sim_df,        open('models/similarity_matrix.pkl',    'wb'))
pickle.dump(product_catalogue,  open('models/product_catalogue.pkl',    'wb'))
pickle.dump(category_sim_df,    open('models/category_similarity.pkl',  'wb'))
pickle.dump(customer_features,  open('models/customer_features.pkl',    'wb'))

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder().fit(PRODUCT_CATEGORIES)
pickle.dump(enc, open('models/encoder_category.pkl', 'wb'))

print("Saved 5 artefacts to models/:")
for f in ['similarity_matrix.pkl','product_catalogue.pkl','category_similarity.pkl',
          'customer_features.pkl','encoder_category.pkl']:
    size = os.path.getsize(f'models/{f}') / 1e6
    print(f"  {f}: {size:.1f} MB")
```

### Section 9 — Inference Functions

```python
# ── Load artefacts ────────────────────────────────────────────────────────
item_sim_df       = pickle.load(open('models/similarity_matrix.pkl',   'rb'))
product_catalogue = pickle.load(open('models/product_catalogue.pkl',   'rb'))
category_sim_df   = pickle.load(open('models/category_similarity.pkl', 'rb'))

from src.constants import PRODUCT_CATEGORIES, SAMPLE_PROFILES


def predict_product(user_profile, candidates=None):
    """
    Score candidate categories by mean item-similarity to user's purchase history.

    Parameters
    ----------
    user_profile : dict  — from build_user_profile()
    candidates   : list  — subset of PRODUCT_CATEGORIES to score (default: all)

    Returns
    -------
    list[tuple[str, float]] — [(category, score), ...] sorted descending
    """
    bought = user_profile['purchase_history']
    cats   = candidates or PRODUCT_CATEGORIES

    if not bought:
        raise ValueError("predict_product() called with empty purchase_history. "
                         "Use find_popular_categories() for cold-start users.")

    # Filter bought to items present in similarity matrix
    valid_bought = [sc for sc in bought if sc in item_sim_df.columns]
    if not valid_bought:
        return [(cat, 0.0) for cat in cats]

    scores = {}
    for cat in cats:
        cat_items = product_catalogue[product_catalogue['category'] == cat]['StockCode']
        valid_cat = [sc for sc in cat_items if sc in item_sim_df.index]
        if not valid_cat:
            scores[cat] = 0.0
        else:
            scores[cat] = float(item_sim_df.loc[valid_cat, valid_bought].values.mean())

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def recommend_products(user_profile, category, top_n=3):
    """
    Recommend specific products within a category using item-similarity scoring.

    Parameters
    ----------
    user_profile : dict  — from build_user_profile()
    category     : str   — one of PRODUCT_CATEGORIES
    top_n        : int   — number of products to return

    Returns
    -------
    list[dict] — [{"category": str, "product": str, "score": float}, ...]
                  products not in purchase_history, sorted by score descending
    """
    bought = user_profile['purchase_history']
    valid_bought = [sc for sc in bought if sc in item_sim_df.columns]

    cat_items = product_catalogue[product_catalogue['category'] == category].copy()
    unowned   = cat_items[~cat_items['StockCode'].isin(bought)]

    if not valid_bought or unowned.empty:
        # Fallback to popularity if no history or all items owned
        top = unowned.nlargest(top_n, 'popularity_rank')
        return [{"category": category, "product": row['Description'], "score": 0.0}
                for _, row in top.iterrows()]

    valid_unowned = unowned[unowned['StockCode'].isin(item_sim_df.index)].copy()
    if valid_unowned.empty:
        top = unowned.nlargest(top_n, 'popularity_rank')
        return [{"category": category, "product": row['Description'], "score": 0.0}
                for _, row in top.iterrows()]

    valid_unowned['score'] = (item_sim_df
                               .loc[valid_unowned['StockCode'], valid_bought]
                               .mean(axis=1).values)
    top = valid_unowned.nlargest(top_n, 'score')
    return [{"category": category, "product": row['Description'], "score": round(float(row['score']), 4)}
            for _, row in top.iterrows()]
```

### Section 10 — Demo on SAMPLE_PROFILES

```python
# Run all 5 profiles to verify end-to-end
from src.constants import SAMPLE_PROFILES

for name, profile in SAMPLE_PROFILES.items():
    print(f"\n{'='*50}")
    print(f"Profile: {name}")
    if profile['purchase_history']:
        cats     = predict_product(profile)[:3]
        products = []
        for cat, _ in cats:
            products += recommend_products(profile, category=cat, top_n=3)
        print(f"Top categories: {cats}")
        print(f"Products:")
        for p in products:
            print(f"  [{p['category']}] {p['product']} — {p['score']:.4f}")
    else:
        print("  Cold-start user — no CF scores available")
        print("  (Member 1 handles this via find_popular_categories)")
```

---

## Saved Artefacts (5 files)

| File | Contents | Size (approx) |
|---|---|---|
| `models/similarity_matrix.pkl` | item_sim_df — StockCode × StockCode cosine similarity | ~50 MB |
| `models/product_catalogue.pkl` | StockCode, Description, category, avg_price, popularity_rank | ~500 KB |
| `models/category_similarity.pkl` | 8×8 category-category cosine similarity DataFrame | ~1 KB |
| `models/customer_features.pkl` | Per-customer: avg_spend, segment, favourite_category, etc. | ~200 KB |
| `models/encoder_category.pkl` | LabelEncoder for 8 PRODUCT_CATEGORIES | ~1 KB |

**Delete old suvroo artefacts** (no longer used):
`product_recommendation_model.pkl`, `rf_product_model.pkl`, `encoder_gender.pkl`,
`encoder_city.pkl`, `encoder_brand.pkl`, `encoder_subcat.pkl`, `encoder_device.pkl`,
`encoder_payment.pkl`, `encoder_season.pkl`

---

## Public Interface (unchanged signatures)

```python
# Category-level CF scoring — for returning users only
predict_product(user_profile: dict, candidates: list = None) -> list[tuple[str, float]]
# Returns: [("Home Decor", 0.82), ("Kitchen & Dining", 0.71), ...]

# Product-level CF scoring — for returning users only
recommend_products(user_profile: dict, category: str, top_n: int = 3) -> list[dict]
# Returns: [{"category": "Home Decor", "product": "PICTURE FRAME", "score": 0.81}, ...]
```

---

## What To Do

1. `Kernel → Restart & Run All` on `notebooks/ml_model.ipynb`
2. Verify clean row count (~397,884), customer count (~4,338), product count (~3,665)
3. Check category distribution — Home Decor dominant (~56%), others present
4. Check similarity matrix diagonal = 1.0, all values in [0,1]
5. Note Hit Rate@K from Section 7 — record in notebook commentary
6. Verify all 5 artefacts saved to `models/`
7. Run Section 10 demo — all 4 returning profiles should produce non-zero scores
8. Commit artefacts: `git add models/ notebooks/ml_model.ipynb src/constants.py`
9. Create draft PR: `gh pr create --draft`

## Rules

- `predict_product()` and `recommend_products()` signatures are fixed — Member 4 calls them
- `recommend_products()` must return flat list of dicts with keys: category, product, score
- Never return items that appear in `purchase_history`
- Do not push — user pushes manually
