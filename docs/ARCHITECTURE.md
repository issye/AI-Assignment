# System Architecture — CIC6314 Smart Product Recommendation System

> **Pivoted 2026-06-07:** suvroo (synthetic) → UCI Online Retail (real UK transactions)
> Approach: Item-Based Collaborative Filtering + Popularity Fallback

```mermaid
flowchart TD
    %% ── Data Layer ──────────────────────────────────────────────
    subgraph DATA["Data Layer  (data/online+retail/)"]
        OR[("Online Retail.xlsx\n397,884 transactions\n4,338 customers · 3,665 products\nRead-only — never modified")]
        PC[("product_categories.csv\nDerived lookup\nStockCode → 8 categories")]
    end

    %% ── Shared Config ───────────────────────────────────────────
    subgraph CONST["src/constants.py  (shared config)"]
        BUP["build_user_profile()\ncustomer_id · purchase_history\navg_order_value · segment · recency"]
        SP["SAMPLE_PROFILES\n5 real Online Retail customers"]
        CFG["PRODUCT_CATEGORIES (8)\nSPEND_THRESHOLDS (£)\nCUSTOMER_SEGMENTS"]
    end

    %% ── Offline Training ────────────────────────────────────────
    subgraph TRAIN["notebooks/ml_model.ipynb  (Member 3 — run once)"]
        CLEAN["S1-2: Clean + Category Engineering"]
        FEAT["S3-4: Customer Features\nUser-Item Matrix (binary)"]
        SIM["S5: Item-Item Cosine Similarity\n3,665 × 3,665 matrix\n+ 8×8 Category Similarity"]
        POP["S6: Global Popularity Table\nunique buyers per product"]
        EVAL["S7: Hit Rate@K evaluation"]
        ART["S8: Save 5 artefacts\nsimilarity_matrix · product_catalogue\ncategory_similarity · customer_features\nencoder_category"]
    end

    %% ── Inference Modules ───────────────────────────────────────
    subgraph INFER["Inference Layer  (src/)"]
        RE["rules_engine.py  (Member 2)\napply_rules(user_profile)\n→ eligible: list[str]\n10 behavioural rules\nno demographics"]
        SM["search_module.py  (Member 1)\nfind_reachable_categories(...)\n→ shortlist: list[str]\n\nfind_popular_categories(...)\n→ list[tuple[str,int]]"]
        CF["CF Inference  (Member 3, notebook)\npredict_product(user_profile, candidates)\n→ [(category, score)]\n\nrecommend_products(user_profile, category)\n→ [dict(category,product,score)]"]
    end

    %% ── Two Paths ───────────────────────────────────────────────
    subgraph PATHS["Two Recommendation Paths  (Member 4 routes)"]
        PERS["Personalised Path\npurchase_history non-empty\nCF drives scoring"]
        COLD["Popular Path\npurchase_history empty\nPopularity drives scoring"]
    end

    %% ── Orchestration ───────────────────────────────────────────
    subgraph ORCH["notebooks/career_recommender.ipynb  (Member 4)\nrecommend(user_profile) → dict"]
        OUT["{\n  recommendation_type: personalised|popular\n  top_3_categories: [(cat, score), ...]\n  recommended_products: [dict × 9]\n  eligible: [str]\n}"]
    end

    %% ── Edges: offline ──────────────────────────────────────────
    OR --> CLEAN --> FEAT --> SIM --> EVAL
    OR --> POP
    SIM --> ART
    POP --> ART
    PC --> CONST

    %% ── Edges: shared config ─────────────────────────────────────
    OR --> CONST
    CONST --> BUP

    %% ── Edges: inference ────────────────────────────────────────
    BUP --> RE
    ART -->|"category_similarity.pkl\nproduct_catalogue.pkl"| SM
    ART -->|"similarity_matrix.pkl\nproduct_catalogue.pkl"| CF

    RE -->|"eligible categories"| SM
    RE -->|"eligible categories"| CF

    SM -->|"shortlist (returning)"| PERS
    SM -->|"popular ranking (new)"| COLD
    CF -->|"category + product scores"| PERS

    PERS --> ORCH
    COLD --> ORCH
    ORCH --> OUT
```

---

## Module Status

| Module | Owner | Status |
|---|---|---|
| `src/constants.py` | Member 3 | ✅ Updated — 8 categories, £ thresholds, new profile schema |
| `notebooks/ml_model.ipynb` | Member 3 | ✅ Rewritten — ❌ needs `Restart & Run All` to save artefacts |
| `data/online+retail/product_categories.csv` | Member 3 | ✅ Generated |
| `src/rules_engine.py` | Member 2 | ❌ Not yet implemented (brief: `agent_handoff/rules_handoff.md`) |
| `src/search_module.py` | Member 1 | ❌ Not yet implemented (brief: `agent_handoff/search_handoff.md`) |
| `notebooks/career_recommender.ipynb` | Member 4 | ❌ Not yet implemented (brief: `agent_handoff/integration_handoff.md`) |

---

## Data Flow Summary

```
Online Retail.xlsx  (read-only)
   └─► notebooks/ml_model.ipynb  (run once offline)
          ├─► similarity_matrix.pkl      (3,665 × 3,665 item cosine similarity)
          ├─► category_similarity.pkl    (8 × 8 category cosine similarity)
          ├─► product_catalogue.pkl      (descriptions + categories + popularity)
          ├─► customer_features.pkl      (per-customer derived features)
          └─► encoder_category.pkl

At inference time:
   build_user_profile()
          │
          ▼
   apply_rules()           → eligible categories  (Member 2)
          │
          ├─ purchase_history EMPTY ──────────────────────────
          │   find_popular_categories()              (Member 1)
          │   → popular categories + products
          │   recommendation_type = "popular"
          │
          └─ purchase_history NON-EMPTY ──────────────────────
              find_reachable_categories()            (Member 1)
              → shortlist (BFS on category graph)
              predict_product()                      (Member 3)
              → ranked categories by CF score
              recommend_products() × 3               (Member 3)
              → 3 products per top-3 category
              recommendation_type = "personalised"
                         │
                         ▼
                   recommend()                       (Member 4)
                   → {recommendation_type, top_3_categories,
                      recommended_products, eligible}
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Item-based CF instead of ML classifier | Real co-purchase signal; no data leakage; no synthetic pairs; model is a 50 MB matrix vs 231 MB Random Forest |
| No demographics | Online Retail has no age/gender; CF doesn't need them |
| Two-path routing | CF cannot score users with zero purchase history; popularity is the honest cold-start fallback |
| 8 categories (not 6) | Better reflects Online Retail's natural product distribution |
| Products from top-3 categories | Avoids over-reliance on single top category; 9 products total per recommendation |
| `recommendation_type` flag | Signals to display layer whether scores are CF similarity or popularity counts |
