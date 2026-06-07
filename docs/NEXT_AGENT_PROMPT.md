# Next Agent Prompt — ML Model Search via Continuous Backtesting

## Your Task

Run a systematic backtest comparing multiple traditional machine learning recommendation
models on the UCI Online Retail dataset. Find the best performing model by Hit Rate@K
and update the project to use it.

---

## Project Context

Read `docs/HANDOFF.md` in full before doing anything else. It contains the full system
architecture, file locations, dataset details, and decisions made in the previous session.

Short summary:
- **Dataset:** `data/online+retail/Online Retail.xlsx` — 397,884 real UK transactions,
  4,338 customers, 3,665 products, Dec 2010–Dec 2011
- **System:** Smart product recommendation system, 4-member team, CIC6314 assignment
- **Deadline:** 27 June 2026, 7 PM (19 days remaining)
- **Current approach:** Item-based CF using raw cosine similarity on a binary user-item matrix
- **Your job:** Find a better ML model through backtesting

---

## What Was Already Tried (Do Not Repeat)

These were backtested in the previous session using a temporal split
(train < 2011-11-01, test >= 2011-11-01, n=1,544 customers):

| Model | HR@5 | Notes |
|---|---|---|
| Raw Cosine CF | 0.2247 | Current best — this is your baseline to beat |
| SVD only | 0.1652 | Too sparse for SVD to work well |
| TF-IDF + SVD | 0.1464 | TF-IDF made SVD worse |
| Popularity | 0.1451 | Lower bound |

The dataset has 1.54% density — very sparse. SVD struggles because it can only explain
~24–38% of variance with 100 components. Any model you try must handle sparse implicit
feedback data.

---

## Models to Try (Traditional ML Only)

Try these in order of expected suitability. All are from scikit-learn or the `implicit`
library. No deep learning, no neural networks, no transformers.

### Tier 1 — Most Promising (try these first)

1. **ALS — Alternating Least Squares** (`implicit` library)
   - Designed specifically for sparse implicit feedback (binary purchase data)
   - Iterative optimisation — genuinely a trained ML model
   - `pip install implicit`
   ```python
   import implicit
   model = implicit.als.AlternatingLeastSquares(factors=50, iterations=20, regularization=0.1)
   model.fit(sparse_matrix.T)  # expects items × users sparse matrix
   ```
   - Tune: `factors` in [20, 50, 100], `iterations` in [10, 20, 50], `regularization` in [0.01, 0.1, 1.0]

2. **BPR — Bayesian Personalised Ranking** (`implicit` library)
   - Optimises directly for ranking (not reconstruction) — well suited to recommendation
   - `implicit.bpr.BayesianPersonalizedRanking(factors=50)`
   - Tune: same hyperparameters as ALS

3. **NMF — Non-Negative Matrix Factorisation** (`sklearn`)
   - Learns non-negative latent factors — more interpretable than SVD
   - `sklearn.decomposition.NMF(n_components=50, max_iter=200)`
   - Similarity from learned item factors (same as SVD approach)

### Tier 2 — Worth Trying

4. **KNN Item-Based CF** (`sklearn`)
   - Explicit nearest-neighbour lookup instead of cosine average
   - `sklearn.neighbors.NearestNeighbors(metric='cosine', algorithm='brute')`
   - For each query item, find K most similar items, aggregate

5. **TF-IDF + Cosine (no SVD)**
   - Apply TF-IDF weighting but skip dimensionality reduction
   - Preserves all signal, removes popularity bias
   - May beat raw cosine by downweighting ubiquitous items

6. **BM25 weighting + Cosine**
   - `implicit` library has a BM25 weighting utility
   - `implicit.nearest_neighbours.BM25Recommender`
   - Stronger than TF-IDF for implicit feedback

---

## Backtest Protocol (USE THIS EXACTLY)

Use the existing script at `scripts/run_backtest.py` as your starting template.

**Temporal split — DO NOT CHANGE:**
- Train: `InvoiceDate < 2011-11-01`
- Test: `InvoiceDate >= 2011-11-01`
- Evaluate on customers present in BOTH splits (n ≈ 1,544)

**Evaluation metric:**
- Primary: **Hit Rate@5** — did any test purchase appear in top-5 recommendations?
- Also report: HR@1, HR@3, HR@10
- A "hit" = at least one actual test purchase appears in the top-K list

**For implicit/ALS models**, the recommendation generation changes:
```python
# ALS returns scores directly
user_ids  = [ui_train.index.get_loc(cid)]
item_ids, scores = model.recommend(user_id, user_items[user_id], N=k, filter_already_liked=True)
top_k = [ui_train.columns[i] for i in item_ids]
```

**Always include these baselines in every comparison table:**
- Raw Cosine CF: HR@5 = 0.2247 (pre-computed, include as reference row)
- Popularity: HR@5 = 0.1451 (pre-computed, include as reference row)

---

## Hyperparameter Tuning

For each model that shows promise (within 5% of raw cosine or better), tune its
key hyperparameters. Use a simple grid — not cross-validation, just re-run the
temporal backtest with different values:

Example for ALS:
```
factors:        [20, 50, 100]
iterations:     [10, 20, 50]
regularization: [0.01, 0.1, 1.0]
→ 27 combinations — run all, pick best HR@5
```

---

## Output Required

1. **A results table** showing every model + every hyperparameter variant tested,
   with HR@1, HR@3, HR@5, HR@10

2. **A clear winner** — the single best model configuration

3. **Updated `notebooks/ml_model.ipynb`** — replace Section 5 with the winning model.
   Use `scripts/build_notebook.py` to regenerate the notebook (do not edit the .ipynb
   JSON directly). The section should:
   - Explain why this model was chosen (cite backtest numbers)
   - Show the training step clearly (this satisfies the assignment ML rubric)
   - Save the trained model to `models/` as a pkl artefact

4. **Updated `docs/HANDOFF.md`** — append a new section with:
   - Full results table
   - Winning model + configuration
   - HR@K numbers for the winner
   - What was updated in the codebase

---

## Constraints

- **No deep learning** — no neural networks, no embeddings, no transformers
- **No external APIs** — all computation local
- **Traditional ML only** — scikit-learn, implicit, scipy are all fine
- **Keep the interface intact** — `predict_product()` and `recommend_products()` signatures
  must not change. Only the similarity matrix / scoring internals change.
- **Do not touch** `src/rules_engine.py` or `src/search_module.py` — those are other
  members' files
- **Do not push** — user pushes manually

---

## File Locations

```
Project root:   C:\Users\12111\Desktop\STUDIES\Projects\Career-Recommender-System\AI-Assignment\
Notebook:       notebooks/ml_model.ipynb
Build script:   scripts/build_notebook.py   ← use this to rebuild the notebook
Backtest script:scripts/run_backtest.py     ← use this as your template
Dataset:        data/online+retail/Online Retail.xlsx
Constants:      src/constants.py
Handoff:        docs/HANDOFF.md
```

---

## Definition of Done

- [ ] At least 6 model variants backtested (including ALS and BPR)
- [ ] Best model beats or matches raw cosine HR@5 = 0.2247
- [ ] `notebooks/ml_model.ipynb` Section 5 updated with winning model
- [ ] Winning model artefact saved to `models/`
- [ ] Full results table documented in `docs/HANDOFF.md`
