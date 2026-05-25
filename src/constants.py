# =============================================================================
# constants.py
# Shared constants for the Career Recommendation System
# CIC6314 Artificial Intelligence — Group Project
#
# ALL MODULES MUST IMPORT FROM THIS FILE.
# Do not hardcode career names, field values, or skill strings anywhere else.
# If you need to add/change something, update here and notify the team.
#
# DATASET: career_dataset_large.xlsx (5000 rows, 6 features)
# All values below are derived from this dataset — do not change them
# without updating the dataset and notifying the team.
# =============================================================================


# -----------------------------------------------------------------------------
# CAREER CATEGORIES
# 12 career labels from the dataset — exact strings used as ML class labels.
#   - Rules engine  (apply_rules)      must return values from this list
#   - A* graph      (find_career_path) must use these as node names
#   - ML model      (predict_career)   must use these as class labels
# -----------------------------------------------------------------------------
CAREER_CATEGORIES = [
    "Business Analyst",
    "Clerk",
    "Data Entry Operator",
    "Financial Analyst",
    "Junior Accountant",
    "Marketing Executive",
    "ML Engineer",
    "Professor",
    "Research Scientist",
    "Sales Assistant",
    "School Counselor",
    "Software Engineer",
]


# -----------------------------------------------------------------------------
# EDUCATION LEVELS
# Exact strings from the dataset "Education Level" column.
# Ordered from lowest to highest for comparison logic.
# -----------------------------------------------------------------------------
EDUCATION_LEVELS = [
    "Matric",           # ~high school
    "Intermediate",     # ~A-levels / diploma
    "Bachelor's",
    "Master's",
    "PhD",
]

# Numeric mapping for rules/A* that need to compare levels
EDUCATION_RANK = {level: i for i, level in enumerate(EDUCATION_LEVELS)}
# Usage: EDUCATION_RANK["Bachelor's"] → 2, EDUCATION_RANK["Master's"] → 3


# -----------------------------------------------------------------------------
# SPECIALIZATIONS  (degree field / major)
# Exact strings from the dataset "Specialization" column.
# Use these in the user profile "specialization" field.
# -----------------------------------------------------------------------------
SPECIALIZATIONS = [
    "Arts",
    "Business",
    "Commerce",
    "Computer Science",
    "Engineering",
    "Finance",
    "Psychology",
    "Science",
]


# -----------------------------------------------------------------------------
# SKILLS
# Exact 10 skill tokens from the dataset "Skills" column.
# The dataset stores skills as comma-separated strings; these are the tokens
# after splitting. Member 3 uses MultiLabelBinarizer on these for ML encoding.
# -----------------------------------------------------------------------------
ALL_SKILLS = [
    "Accounting",
    "Communication",
    "Counseling",
    "Data Analysis",
    "Financial Analysis",
    "Machine Learning",
    "Marketing",
    "MS Office",
    "Python",
    "SQL",
]


# -----------------------------------------------------------------------------
# CERTIFICATIONS
# Exact strings from the dataset "Certifications" column.
# 11.92% of rows have no certification — represented as None in the profile
# and filled with "None" during ML preprocessing.
# -----------------------------------------------------------------------------
CERTIFICATIONS = [
    "AWS Certified",
    "CFA Level 1",
    "Creative Writing",
    "Digital Marketing",
    "Google Data Analytics",
    "Mental Health Basics",
    "Tally ERP",
    None,   # no certification — valid value
]


# -----------------------------------------------------------------------------
# USER PROFILE SCHEMA
# Standard input format for ALL three modules.
#
#   apply_rules(user_profile)         → list[str]  eligible careers
#   find_career_path(user_profile,    → list[str]  path of roles
#                    target_career)
#   predict_career(user_profile)      → list[tuple[str, float]]
#                                        [(career, confidence), ...]
#                                        sorted by confidence descending
#
# Use build_user_profile() below to construct a validated profile dict.
#
# NOTE ON CGPA: the dataset uses a 60–95 integer scale (not 0.0–4.0).
#               Pass cgpa as an integer in that range (e.g. 78).
# -----------------------------------------------------------------------------

def build_user_profile(
    education_level: str,
    specialization: str,
    cgpa: int,
    skills: list,
    certifications: str = None,
    target_career: str = None,
) -> dict:
    """
    Build and validate a user profile dict.
    Raises ValueError if any field contains an unrecognised value.

    Parameters
    ----------
    education_level : one of EDUCATION_LEVELS
    specialization  : one of SPECIALIZATIONS
    cgpa            : int 60–95 (dataset scale)
    skills          : list of strings from ALL_SKILLS
    certifications  : one of CERTIFICATIONS, or None
    target_career   : one of CAREER_CATEGORIES, or None

    Returns
    -------
    dict with all fields validated
    """
    if education_level not in EDUCATION_LEVELS:
        raise ValueError(f"education_level must be one of {EDUCATION_LEVELS}")
    if specialization not in SPECIALIZATIONS:
        raise ValueError(f"specialization must be one of {SPECIALIZATIONS}")
    if not (60 <= cgpa <= 95):
        raise ValueError("cgpa must be between 60 and 95 (dataset scale)")
    invalid_skills = [s for s in skills if s not in ALL_SKILLS]
    if invalid_skills:
        raise ValueError(f"Unrecognised skills: {invalid_skills}. Must be from ALL_SKILLS.")
    if certifications not in CERTIFICATIONS:
        raise ValueError(f"certifications must be one of {CERTIFICATIONS}")
    if target_career is not None and target_career not in CAREER_CATEGORIES:
        raise ValueError(f"target_career must be one of {CAREER_CATEGORIES} or None")

    return {
        "education_level": education_level,
        "specialization":  specialization,
        "cgpa":            int(cgpa),
        "skills":          list(skills),
        "certifications":  certifications,
        "target_career":   target_career,
    }


# -----------------------------------------------------------------------------
# SAMPLE USER PROFILES
# Use these to test your module during development.
# Member 4 will also use these for the final integration demo.
# Values are chosen to realistically represent each career path.
# -----------------------------------------------------------------------------

SAMPLE_PROFILES = {

    "cs_ml_student": build_user_profile(
        education_level = "Bachelor's",
        specialization  = "Computer Science",
        cgpa            = 82,
        skills          = ["Python", "Machine Learning", "Data Analysis", "SQL"],
        certifications  = "Google Data Analytics",
        target_career   = "ML Engineer",
    ),

    "finance_graduate": build_user_profile(
        education_level = "Bachelor's",
        specialization  = "Finance",
        cgpa            = 75,
        skills          = ["Accounting", "Financial Analysis", "MS Office"],
        certifications  = "CFA Level 1",
        target_career   = "Financial Analyst",
    ),

    "business_analyst": build_user_profile(
        education_level = "Master's",
        specialization  = "Business",
        cgpa            = 80,
        skills          = ["Data Analysis", "Communication", "MS Office", "SQL"],
        certifications  = "Google Data Analytics",
        target_career   = "Business Analyst",
    ),

    "psychology_grad": build_user_profile(
        education_level = "Bachelor's",
        specialization  = "Psychology",
        cgpa            = 72,
        skills          = ["Counseling", "Communication", "MS Office"],
        certifications  = "Mental Health Basics",
        target_career   = "School Counselor",
    ),

    "software_engineer": build_user_profile(
        education_level = "Bachelor's",
        specialization  = "Engineering",
        cgpa            = 85,
        skills          = ["Python", "SQL", "Machine Learning", "Data Analysis"],
        certifications  = "AWS Certified",
        target_career   = "Software Engineer",
    ),
}
