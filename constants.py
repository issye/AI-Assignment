# =============================================================================
# constants.py
# Shared constants for the Career Recommendation System
# CIC6314 Artificial Intelligence — Group Project
#
# ALL MODULES MUST IMPORT FROM THIS FILE.
# Do not hardcode career names, field values, or skill strings anywhere else.
# If you need to add/change something, update here and notify the team.
# =============================================================================


# -----------------------------------------------------------------------------
# CAREER CATEGORIES
# These are the 10 supported career outputs across all three modules:
#   - Rules engine  (apply_rules)      must return values from this list
#   - A* graph      (find_career_path) must use these as node names
#   - ML model      (predict_career)   must use these as class labels
# -----------------------------------------------------------------------------
CAREER_CATEGORIES = [
    "Software Engineer",
    "Data Scientist",
    "Data Analyst",
    "Machine Learning Engineer",
    "Cybersecurity Analyst",
    "Product Manager",
    "UX Designer",
    "Business Analyst",
    "Finance Analyst",
    "DevOps Engineer",
]


# -----------------------------------------------------------------------------
# EDUCATION LEVELS
# Use these exact strings in the user profile "education_level" field.
# Ordered from lowest to highest for comparison logic.
# -----------------------------------------------------------------------------
EDUCATION_LEVELS = [
    "high_school",
    "diploma",
    "bachelor",
    "master",
    "phd",
]

# Numeric mapping for rules/ML that need to compare levels
EDUCATION_RANK = {level: i for i, level in enumerate(EDUCATION_LEVELS)}
# Usage: EDUCATION_RANK["bachelor"] → 2, EDUCATION_RANK["master"] → 3


# -----------------------------------------------------------------------------
# DEGREE FIELDS
# Use these exact strings in the user profile "degree_field" field.
# -----------------------------------------------------------------------------
DEGREE_FIELDS = [
    "Computer Science",
    "Information Technology",
    "Software Engineering",
    "Data Science",
    "Cybersecurity",
    "Electrical Engineering",
    "Business Administration",
    "Finance",
    "Accounting",
    "Graphic Design",
    "Psychology",
    "Other",
]


# -----------------------------------------------------------------------------
# SKILLS
# Canonical skill names. Use these exact strings in the skills list.
# Grouped by domain for readability — the actual value is just the string.
# -----------------------------------------------------------------------------

# Programming & Data
SKILLS_PROGRAMMING = [
    "Python", "Java", "C++", "JavaScript", "R",
    "SQL", "NoSQL", "MATLAB",
]

# Data & AI
SKILLS_DATA_AI = [
    "Machine Learning", "Deep Learning", "Data Analysis",
    "Data Visualization", "Statistics", "NLP",
    "Computer Vision", "pandas", "scikit-learn", "TensorFlow",
]

# Web & DevOps
SKILLS_DEVOPS = [
    "Linux", "Docker", "Kubernetes", "CI/CD",
    "Cloud (AWS)", "Cloud (Azure)", "Cloud (GCP)",
    "Networking", "Git",
]

# Security
SKILLS_SECURITY = [
    "Cybersecurity", "Penetration Testing", "Network Security",
    "SIEM", "Incident Response", "Cryptography",
]

# Business & Soft
SKILLS_BUSINESS = [
    "Project Management", "Communication", "Leadership",
    "Excel", "PowerPoint", "Business Analysis",
    "Financial Modeling", "Accounting",
]

# Design
SKILLS_DESIGN = [
    "UI/UX Design", "Figma", "Adobe XD", "Prototyping",
    "User Research",
]

# Flat list of all skills (use for validation / ML encoding)
ALL_SKILLS = (
    SKILLS_PROGRAMMING +
    SKILLS_DATA_AI +
    SKILLS_DEVOPS +
    SKILLS_SECURITY +
    SKILLS_BUSINESS +
    SKILLS_DESIGN
)


# -----------------------------------------------------------------------------
# INTERESTS
# Use these exact strings in the user profile "interests" list.
# -----------------------------------------------------------------------------
INTERESTS = [
    "Artificial Intelligence",
    "Data and Analytics",
    "Cybersecurity",
    "Web Development",
    "Cloud Computing",
    "Product and Strategy",
    "Design and Creativity",
    "Finance and Economics",
    "Business and Management",
    "Research and Academia",
]


# -----------------------------------------------------------------------------
# USER PROFILE SCHEMA
# This is the standard input format for ALL three modules.
#
#   apply_rules(user_profile)         → list[str]  eligible careers
#   find_career_path(user_profile,    → list[str]  path of roles
#                    target_career)
#   predict_career(user_profile)      → list[tuple[str, float]]
#                                        [(career, confidence), ...]
#                                        sorted by confidence descending
#
# Use build_user_profile() below to construct a valid profile dict.
# -----------------------------------------------------------------------------

def build_user_profile(
    education_level: str,
    degree_field: str,
    gpa: float,
    skills: list,
    years_experience: int,
    interests: list,
    leadership: bool = False,
    target_career: str = None,
) -> dict:
    """
    Build and validate a user profile dict.
    Raises ValueError if any field contains an unrecognised value.

    Parameters
    ----------
    education_level  : one of EDUCATION_LEVELS
    degree_field     : one of DEGREE_FIELDS
    gpa              : float 0.0 – 4.0
    skills           : list of strings from ALL_SKILLS
    years_experience : int 0–40
    interests        : list of strings from INTERESTS
    leadership       : bool, has leadership/management experience
    target_career    : one of CAREER_CATEGORIES, or None

    Returns
    -------
    dict with all fields validated
    """
    if education_level not in EDUCATION_LEVELS:
        raise ValueError(f"education_level must be one of {EDUCATION_LEVELS}")
    if degree_field not in DEGREE_FIELDS:
        raise ValueError(f"degree_field must be one of {DEGREE_FIELDS}")
    if not (0.0 <= gpa <= 4.0):
        raise ValueError("gpa must be between 0.0 and 4.0")
    if target_career is not None and target_career not in CAREER_CATEGORIES:
        raise ValueError(f"target_career must be one of {CAREER_CATEGORIES} or None")

    return {
        "education_level":  education_level,
        "degree_field":     degree_field,
        "gpa":              float(gpa),
        "skills":           list(skills),
        "years_experience": int(years_experience),
        "interests":        list(interests),
        "leadership":       bool(leadership),
        "target_career":    target_career,
    }


# -----------------------------------------------------------------------------
# SAMPLE USER PROFILES
# Use these to test your module during development.
# Member 4 will also use these for the final integration demo.
# -----------------------------------------------------------------------------

SAMPLE_PROFILES = {

    "cs_graduate": build_user_profile(
        education_level  = "bachelor",
        degree_field     = "Computer Science",
        gpa              = 3.6,
        skills           = ["Python", "Machine Learning", "SQL",
                            "Data Analysis", "scikit-learn"],
        years_experience = 1,
        interests        = ["Artificial Intelligence", "Data and Analytics"],
        leadership       = False,
        target_career    = "Data Scientist",
    ),

    "cybersecurity_student": build_user_profile(
        education_level  = "bachelor",
        degree_field     = "Cybersecurity",
        gpa              = 3.2,
        skills           = ["Cybersecurity", "Python", "Networking",
                            "Linux", "Penetration Testing"],
        years_experience = 0,
        interests        = ["Cybersecurity", "Cloud Computing"],
        leadership       = False,
        target_career    = "Cybersecurity Analyst",
    ),

    "business_with_tech": build_user_profile(
        education_level  = "bachelor",
        degree_field     = "Business Administration",
        gpa              = 3.4,
        skills           = ["Excel", "SQL", "Business Analysis",
                            "Communication", "PowerPoint"],
        years_experience = 2,
        interests        = ["Business and Management", "Data and Analytics"],
        leadership       = True,
        target_career    = None,
    ),

    "design_focused": build_user_profile(
        education_level  = "diploma",
        degree_field     = "Graphic Design",
        gpa              = 3.0,
        skills           = ["UI/UX Design", "Figma", "Prototyping",
                            "User Research", "Adobe XD"],
        years_experience = 1,
        interests        = ["Design and Creativity", "Product and Strategy"],
        leadership       = False,
        target_career    = "UX Designer",
    ),

    "experienced_dev": build_user_profile(
        education_level  = "bachelor",
        degree_field     = "Software Engineering",
        gpa              = 3.1,
        skills           = ["Python", "Java", "Docker", "Kubernetes",
                            "CI/CD", "Linux", "Cloud (AWS)", "Git"],
        years_experience = 4,
        interests        = ["Cloud Computing", "Web Development"],
        leadership       = True,
        target_career    = "DevOps Engineer",
    ),
}
