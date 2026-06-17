# CIC6314 — Career Recommendation System

> **Course:** CIC6314 Artificial Intelligence | **Lecturer:** Prabha Kumaresan  
> **Session:** March/April 2026 | **Deadline:** 27 June 2026, 7 PM

An intelligent Career Recommendation System that combines rule-based reasoning, A\* search, and machine learning to suggest suitable career paths based on a student's education, skills, and academic background.

---

## Team Members

| Student ID | Name | Branch | Module |
|---|---|---|---|
| | Issye Lailiyah| `feature/search` | Search Algorithm (A\*) + GitHub |
| | Thinesh | `feature/rules` | Knowledge Representation & Logic |
| | Wan Arief | `feature/ml` | Machine Learning Model |
| | Adam Hadziq | `feature/integration` | Integration & System Design + Poster |

---

## Repository Structure

```
cic6314-career-recommender/
│
├── notebooks/
│   └── career_recommender.ipynb   # Main Jupyter notebook (all sections)
│
├── data/
│   └── career_dataset_large.xlsx  # Dataset (5000 rows, 6 features)
│
├── src/
│   ├── constants.py               # Shared constants — import from here only
│   ├── search_module.py           # A* search (Member 1)
│   ├── rules_engine.py            # Inference rules (Member 2)
│   └── ml_model.py                # ML predictor (Member 3)
│
├── scripts/
│   └── dataset_audit.py           # Dataset validation script
│
├── poster/
│   └── poster.pdf                 # Presentation poster (Member 4)
│
├── requirements.txt
└── README.md
```

---

## Dataset

**File:** `data/career_dataset_large.xlsx`  
**Source:** Kaggle — AI-based Career Recommendation System  
**Rows:** 5,000 | **Features:** 6 | **Target:** `Recommended Career` (12 classes)

| Column | Type | Notes |
|---|---|---|
| `Education Level` | categorical | 5 levels: Matric → PhD |
| `Specialization` | categorical | 8 fields (e.g. Computer Science, Finance) |
| `Skills` | multi-label string | comma-separated, 10 unique tokens |
| `Certifications` | categorical | 7 options + None (~12% missing) |
| `CGPA/Percentage` | int | range 60–95 |
| `Recommended Career` | **target** | 12 balanced classes (~395–432 each) |

**Career labels:**
```
Business Analyst, Clerk, Data Entry Operator, Financial Analyst,
Junior Accountant, Marketing Executive, ML Engineer, Professor,
Research Scientist, Sales Assistant, School Counselor, Software Engineer
```

**Preprocessing notes for Member 3:**
- `Skills` → use `MultiLabelBinarizer` to encode the 10 skill tokens
- `Certifications` → fill nulls with `"None"` before encoding
- `CGPA/Percentage` → already numeric, use as-is or normalise
- Drop the 1 duplicate row

---

## System Architecture

```
User Profile Input
       │
       ▼
┌─────────────────────┐
│  Rules Engine       │  ← Member 2
│  apply_rules(       │    IF Education='Bachelor's' AND 'Python' in skills
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
│  predict_career(    │    Random Forest trained on career_dataset_large.xlsx
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

## Module Interfaces

All members import from `src/constants.py`. Never hardcode career names, skill strings, or field values anywhere else.

```python
from src.constants import (
    CAREER_CATEGORIES, EDUCATION_LEVELS, SPECIALIZATIONS,
    ALL_SKILLS, CERTIFICATIONS, SAMPLE_PROFILES, build_user_profile
)
```

### User Profile Schema

```python
user_profile = build_user_profile(
    education_level = "Bachelor's",       # one of EDUCATION_LEVELS
    specialization  = "Computer Science", # one of SPECIALIZATIONS
    cgpa            = 82,                 # int, 60–95
    skills          = ["Python", "SQL"],  # subset of ALL_SKILLS
    certifications  = "AWS Certified",    # one of CERTIFICATIONS, or None
    target_career   = "ML Engineer",      # one of CAREER_CATEGORIES, or None
)
```

### Function Signatures

```python
# Member 2 — rules engine
apply_rules(user_profile)
# returns: list[str] — eligible careers from CAREER_CATEGORIES

# Member 1 — A* search
find_career_path(user_profile, target_career)
# returns: list[str] — ordered career steps from current to target

# Member 3 — ML model
predict_career(user_profile)
# returns: list[tuple[str, float]] — [(career, confidence), ...]
#          sorted by confidence descending
```

### Valid Field Values

```python
EDUCATION_LEVELS  = ["Matric", "Intermediate", "Bachelor's", "Master's", "PhD"]

SPECIALIZATIONS   = ["Arts", "Business", "Commerce", "Computer Science",
                      "Engineering", "Finance", "Psychology", "Science"]

ALL_SKILLS        = ["Accounting", "Communication", "Counseling", "Data Analysis",
                      "Financial Analysis", "Machine Learning", "Marketing",
                      "MS Office", "Python", "SQL"]

CERTIFICATIONS    = ["AWS Certified", "CFA Level 1", "Creative Writing",
                      "Digital Marketing", "Google Data Analytics",
                      "Mental Health Basics", "Tally ERP", None]

CAREER_CATEGORIES = ["Business Analyst", "Clerk", "Data Entry Operator",
                      "Financial Analyst", "Junior Accountant", "Marketing Executive",
                      "ML Engineer", "Professor", "Research Scientist",
                      "Sales Assistant", "School Counselor", "Software Engineer"]
```

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Submission-ready code only. Team lead merges here before submission. |
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

### 4. Run the notebook
```bash
jupyter notebook
# open notebooks/career_recommender.ipynb
```

> **VS Code users:** Install the Jupyter extension and open the `.ipynb` file directly. Works the same way.

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
| `feat(search):` | A\* search algorithm work |
| `feat(rules):` | Rules engine work |
| `feat(ml):` | ML model work |
| `feat(integration):` | Integration pipeline work |
| `fix(module):` | Bug fix |
| `docs:` | Markdown / documentation updates |
| `data:` | Dataset changes |
| `style:` | Formatting or cleanup only |

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
openpyxl
```

---

## Submission Checklist

- [ ] Notebook runs clean: **Kernel → Restart & Run All** with zero errors
- [ ] All 3 AI components present and integrated
- [ ] 8–10 well-defined inference rules documented
- [ ] ML model evaluated with accuracy, F1, confusion matrix
- [ ] A\* heuristic clearly explained in markdown
- [ ] Full pipeline demo using `SAMPLE_PROFILES`
- [ ] Poster file in `poster/` folder
- [ ] Rubric appendix printed and attached
- [ ] ZIP named after team name
- [ ] Submitted by **27 June 2026, 7 PM**
