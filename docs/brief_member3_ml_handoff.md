# Claude Handoff — Member 3 ML Module
# CIC6314 Smart Product Recommendation System
# For: next Claude session continuing this work

---

## Project state as of 2026-06-06

**Branch:** `feature/ml`  
**Repo:** `issye/AI-Assignment` (private, GitHub MCP may need re-auth)  
**Working directory:** `C:\Users\12111\Desktop\STUDIES\Projects\Career-Recommender-System\AI-Assignment`  
**Deadline:** 27 June 2026, 7 PM  
**User:** Member 3 (ML Engineer), Windows CMD environment

---

## What has been completed

### Files committed on `feature/ml`
| File | Status | Notes |
|---|---|---|
| `src/constants.py` | ✅ Complete | Product domain, all constants locked |
| `notebooks/ml_model.ipynb` | ✅ Built, not yet run | Needs Kernel → Restart & Run All |
| `data/ecommerce_customer_behavior_dataset_v2.csv` | ✅ Committed | 17,049 rows, 5,000 customers |
| `docs/brief_member1_search.md` | ✅ Complete | Agent brief for Member 1 |
| `docs/brief_member2_rules.md` | ✅ Complete | Agent brief for Member 2 |
| `docs/brief_member4_integration.md` | ✅ Complete | Agent brief for Member 4 |
| `docs/brief_member3_ml_handoff.md` | ✅ This file | |

### Files NOT yet committed
- `data/career_dataset_large.xlsx` — old career dataset, still in data/ folder. Can be deleted.
- `data/content_based_recommendation_dataset.csv` — rejected dataset, can be deleted.
- `models/` — will be created when notebook is run

---

## Domain & Architecture

**Domain:** Smart Product Recommendation (switched from Career Recommendation)  
**Reason for switch:** Career dataset had randomly assigned labels — all models scored 8.3% (random chance for 12 classes). No learnable signal.

**Architecture (reachability-first):**
```
Rules Engine → A* Search → ML Model → Integration
```

**Why this order:** `median_spend` (MI=0.20) is the only strong feature. A* ranks candidates by price reachability using spend as heuristic. ML re-ranks A*'s shortlist — easier than classifying from scratch across 8 categories.

---

## Dataset facts

**File:** `data/ecommerce_customer_behavior_dataset_v2.csv`  
**Raw:** 17,049 transaction rows × 18 columns  
**After aggregation:** 5,000 user profiles  
**Target:** `favourite_category` = product category where customer spent most total money (NOT mode — mode is unreliable with avg 3.4 orders/customer)  
**8 classes:** Books, Food, Beauty, Toys, Fashion, Sports, Home & Garden, Electronics  
**Class imbalance:** Beauty=~1200, Toys=~300 (4:1 ratio) → handled with `class_weight='balanced'`  

**Feature signal (mutual information vs target):**
- `median_spend`: 0.2027 — strong ✓
- `avg_pages`: 0.0269 — weak
- `avg_quantity`: 0.0119 — weak
- `Age`, `Gender`, `City`: < 0.008 — near zero (indirect via spend)

**Spend quartile thresholds (from data):**
- Low: < 262
- Mid-Low: 262–502
- Mid-High: 502–989
- High: ≥ 989

**Category avg prices (from data):**
Books=56, Food=71, Beauty=112, Toys=169, Fashion=276, Sports=493, Home & Garden=691, Electronics=1767

---

## ML notebook structure (notebooks/ml_model.ipynb)

14 sections already written. The notebook has NOT been run yet — outputs are empty.

1. Setup & Imports
2. Data Loading (raw CSV)
3. EDA (category distribution, demographics, spend)
4. Aggregation (transaction log → user profiles, target engineering)
5. Feature Engineering & Signal Verification (MI scores, cross-tabs)
6. Preprocessing (LabelEncoders, feature matrix assembly)
7. Train/Test Split (80/20 stratified)
8. Baseline Models (Decision Tree, KNN)
9. Primary Model (Random Forest default + GridSearchCV tuned)
10. Model Evaluation (comparison chart, classification report, confusion matrix, feature importance, cross-val)
11. Results Discussion (pre-written markdown — update numbers after running)
12. Save Artefacts (to `models/`)
13. `predict_product()` function (team interface)
14. Testing with SAMPLE_PROFILES + re-ranking mode test

---

## predict_product() interface

```python
predict_product(user_profile: dict, candidates: list = None) -> list[tuple[str, float]]
# If candidates=None: scores all 8 PRODUCT_CATEGORIES
# If candidates provided: re-ranks only those (A* waterfall mode)
# Returns: [(category, confidence), ...] sorted descending
```

Key implementation detail: when `candidates` is provided, the function still runs `predict_proba()` on all 8 classes internally, then filters the output to only the requested candidates before returning. This ensures probabilities are calibrated correctly.

---

## constants.py — product domain

```python
PRODUCT_CATEGORIES = ["Books","Food","Beauty","Toys","Fashion","Sports","Home & Garden","Electronics"]

CATEGORY_AVG_PRICES = {"Books":56,"Food":71,"Beauty":112,"Toys":169,"Fashion":276,"Sports":493,"Home & Garden":691,"Electronics":1767}

SPEND_THRESHOLDS = {"Low":(0,262),"Mid-Low":(262,502),"Mid-High":(502,989),"High":(989,inf)}

URBAN_CITIES = ["Istanbul","Ankara","Izmir"]

SAMPLE_PROFILES = {
    "budget_reader":      age=22, gender=Female, city=Konya,    spend=55.0,   tier=Low
    "beauty_shopper":     age=30, gender=Female, city=Istanbul, spend=180.0,  tier=Low
    "fashion_enthusiast": age=27, gender=Male,   city=Ankara,   spend=380.0,  tier=Mid-Low
    "sports_buyer":       age=35, gender=Male,   city=Izmir,    spend=650.0,  tier=Mid-High
    "tech_spender":       age=42, gender=Male,   city=Istanbul, spend=1500.0, tier=High
}
```

---

## What the next session should do first

1. **Run the notebook** — `Kernel → Restart & Run All` in `notebooks/ml_model.ipynb`
2. **Check model accuracy** — should be well above 12.5% (random chance for 8 classes). Expected 40–60%.
3. **Update Section 11** (Results Discussion markdown) with actual numbers from the run
4. **Delete old datasets** — `career_dataset_large.xlsx` and `content_based_recommendation_dataset.csv`
5. **Commit and push** — then create draft PR to `dev`
6. **Check teammate progress** — Member 1 and 2 need to update their modules. Their briefs are in `docs/`

---

## Known issues / things to watch for

- **`data/career_dataset_large.xlsx` still in repo** — delete it, it's the old broken dataset
- **`models/` directory** — doesn't exist yet, notebook creates it on run
- **GitHub MCP** — repo is `issye/AI-Assignment`, was intermittently unreachable. Use local git commands as fallback.
- **Branch switching** — switching away from `feature/ml` may fail due to untracked `data/` and `models/` folders. Move them temporarily first.
- **Windows CMD** — user is on Windows. Use `move` not `mv`, `del` not `rm`. Bash tool doesn't have Windows commands — use PowerShell or instruct user to run CMD commands manually.
- **Git push** — user pushes manually. Never run `git push`.
- **PRs** — always draft: `gh pr create --draft`

---

## Git state

```
branch: feature/ml
ahead of origin/feature/ml by: 3 commits (not yet pushed)
untracked: data/career_dataset_large.xlsx, data/content_based_recommendation_dataset.csv
```

---

## Team structure

| Member | Branch | Module | Status |
|---|---|---|---|
| Issye (lead) | `feature/search` | A* Search | Written for career domain — needs product domain update. Brief: `docs/brief_member1_search.md` |
| Member 2 | `feature/rules` | Rules Engine | Unknown status. Brief: `docs/brief_member2_rules.md` |
| Member 3 (user) | `feature/ml` | ML Model | Notebook built, not yet run |
| Member 4 | `feature/integration` | Integration | Unknown status. Brief: `docs/brief_member4_integration.md` |

---

## Key decisions made (don't re-debate these)

1. **Reachability-first** (Logic → A* → ML) chosen over data-first (Logic → ML → A*) because demographic features are too weak for ML to be primary recommender
2. **Target = highest-spend category** chosen over mode because avg 3.4 orders/customer makes mode unreliable
3. **class_weight='balanced'** used to handle 4:1 class imbalance
4. **No dataset generation** — real dataset has sufficient signal via median_spend
5. **No domain-specific feature scaling** — Random Forest doesn't require it
6. **Agent briefs replace handoff doc** — single source of truth per member
