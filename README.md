# CIC6314 — Career Recommendation System

> **Course:** CIC6314 Artificial Intelligence | **Lecturer:** Prabha Kumaresan  
> **Session:** March/April 2026 | **Deadline:** 27 June 2026, 7 PM

An intelligent Career Recommendation System that integrates three core AI techniques — rule-based reasoning, A\* search, and machine learning — to recommend suitable career paths based on a user's education, skills, and interests.

---

## Team Members

| Student ID | Name | Branch | Module |
|---|---|---|---|
| | (Team Lead — Issye) | `feature/search` | Search Algorithm (A\*) + GitHub |
| | Member 2 | `feature/rules` | Knowledge Representation & Logic |
| | Member 3 | `feature/ml` | Machine Learning Model |
| | Member 4 | `feature/integration` | Integration & System Design + Poster |

---

## Repository Structure

```
cic6314-career-recommender/
│
├── notebooks/
│   └── career_recommender.ipynb   # Main Jupyter notebook (all sections)
│
├── data/
│   └── career_dataset.csv         # Kaggle dataset (Member 3 adds this)
│
├── src/
│   ├── constants.py               # Shared career categories + user profile schema
│   ├── search_module.py           # A* search (Member 1)
│   ├── rules_engine.py            # Inference rules (Member 2)
│   └── ml_model.py                # ML predictor + career_model.pkl (Member 3)
│
├── poster/
│   └── poster.pdf                 # Presentation poster (Member 4)
│
├── requirements.txt
└── README.md
```

---

## System Architecture

```
User Profile Input
       │
       ▼
┌─────────────────────┐
│  Rules Engine       │  ← Member 2
│  apply_rules(       │    IF degree='CS' AND 'Python' in skills
│    user_profile)    │    → filters eligible career categories
└────────┬────────────┘
         │  filtered careers
         ▼
┌─────────────────────┐
│  A* Search          │  ← Member 1
│  find_career_path(  │    traverses career graph
│    user, target)    │    heuristic = skill gap count
└────────┬────────────┘
         │  optimal path
         ▼
┌─────────────────────┐
│  ML Classifier      │  ← Member 3
│  predict_career(    │    Random Forest trained on Kaggle data
│    user_profile)    │    → ranked career list with confidence scores
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Integration Layer  │  ← Member 4
│  + Final Output     │    Top-3 careers + skill roadmap
└─────────────────────┘
```

---

## Module Interfaces (from `constants.py`)

All members must use the **same** user profile schema and career categories:

```python
# Standard user profile
user_profile = {
    "education_level": "bachelor",      # high_school | diploma | bachelor | master | phd
    "degree_field":    "CS",            # CS | IT | Engineering | Business | Design | Finance
    "gpa":             3.7,             # 0.0 – 4.0
    "skills":          ["Python", "SQL", "Machine Learning"],
    "years_experience": 1,
    "interests":       ["AI", "data"],
    "leadership":      False,
    "target_career":   "Data Scientist" # or None
}

# Module function signatures
apply_rules(user_profile)          # → list of eligible career strings
find_career_path(user_profile,     # → list of path steps
                 target_career)
predict_career(user_profile)       # → [(career_name, confidence_score), ...]
```

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Final submission-ready code only. Team lead merges here before submission. |
| `dev` | Shared integration branch. All features merge here first. |
| `feature/search` | Member 1 — A\* search algorithm |
| `feature/rules` | Member 2 — knowledge rules & inference engine |
| `feature/ml` | Member 3 — machine learning model |
| `feature/integration` | Member 4 — full pipeline integration + poster |

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<team-lead-username>/cic6314-career-recommender.git
cd cic6314-career-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Switch to your feature branch
```bash
git checkout dev
git pull origin dev
git checkout -b feature/your-module-name
```

### 4. Start Jupyter
```bash
jupyter notebook
```

---

## Daily Workflow

```bash
# Start of every session — sync first
git checkout dev && git pull origin dev
git checkout feature/your-branch
git merge dev

# Do your work, then commit
git add .
git commit -m "feat(module): short description of what you did"
git push origin feature/your-branch
```

---

## Commit Message Convention

| Prefix | When to use |
|---|---|
| `feat(search):` | New search algorithm work |
| `feat(rules):` | New rules engine work |
| `feat(ml):` | New ML model work |
| `feat(integration):` | Integration pipeline work |
| `fix(module):` | Bug fix |
| `docs:` | Markdown documentation cells |
| `data:` | Dataset changes |
| `style:` | Formatting/cleanup only |

---

## Dependencies

```
python >= 3.9
jupyter
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
experta          # or pyknow — for rules engine
```

---

## Submission Checklist

- [ ] Notebook runs clean: **Kernel → Restart & Run All** with zero errors
- [ ] All 3 AI components present and integrated
- [ ] 8–10 well-defined inference rules documented
- [ ] ML model evaluated with accuracy, F1, confusion matrix
- [ ] A\* heuristic clearly explained in markdown
- [ ] Full pipeline demo with 3+ test user profiles
- [ ] Poster file in `poster/` folder
- [ ] Rubric appendix printed and attached
- [ ] ZIP named after team name
- [ ] Submitted by **27 June 2026, 7 PM**
