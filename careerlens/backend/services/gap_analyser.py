"""
CareerLens — Skill Gap Analyzer Service
Extracts skills from JD and checks presence in resume.
"""

import re
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .model_manager import model_manager
from models.schemas import SkillMatch


# ── Curated skill taxonomy ────────────────────────────────────────────────────
SKILL_TAXONOMY = {
    "Technical": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
        "Kotlin", "Swift", "R", "MATLAB", "Scala", "Ruby", "PHP",
        "React", "Vue", "Angular", "Next.js", "FastAPI", "Django", "Flask",
        "Spring Boot", "Node.js", "Express", "GraphQL", "REST API",
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "TensorFlow", "PyTorch", "Keras", "scikit-learn", "Hugging Face",
        "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
        "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Terraform", "CI/CD",
        "Git", "Linux", "Bash", "Agile", "Scrum",
    ],
    "Data": [
        "Data Analysis", "Data Engineering", "ETL", "Spark", "Kafka", "Airflow",
        "dbt", "Tableau", "Power BI", "Looker", "Pandas", "NumPy",
        "A/B Testing", "Statistics", "Feature Engineering", "Data Modeling",
        "Data Warehousing", "BigQuery", "Snowflake", "Data Pipeline",
    ],
    "Soft Skill": [
        "Leadership", "Communication", "Teamwork", "Problem Solving",
        "Critical Thinking", "Time Management", "Adaptability", "Collaboration",
        "Project Management", "Stakeholder Management", "Mentoring", "Presentation",
    ],
}

# Flat list with category mapping
ALL_SKILLS: List[Tuple[str, str]] = [
    (skill, cat)
    for cat, skills in SKILL_TAXONOMY.items()
    for skill in skills
]


def extract_skills_from_text(text: str) -> List[str]:
    """Extract skill mentions from text using pattern matching."""
    found = []
    text_lower = text.lower()
    for skill, _ in ALL_SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return list(set(found))


def analyze_skill_gaps(resume_text: str, job_description: str) -> List[SkillMatch]:
    """
    Compare skills in JD vs resume.
    Uses both keyword matching AND semantic similarity for a hybrid approach.
    """
    embedder = model_manager.embedder

    jd_skills = extract_skills_from_text(job_description)
    resume_skills = extract_skills_from_text(resume_text)
    resume_lower = resume_text.lower()

    # Encode resume as a whole for semantic matching
    resume_embedding = embedder.encode([resume_text], convert_to_numpy=True)

    results: List[SkillMatch] = []

    # Only analyze skills mentioned in the JD
    target_skills = jd_skills if jd_skills else [s for s, _ in ALL_SKILLS[:30]]

    for skill in target_skills:
        # Find category
        category = next(
            (cat for s, cat in ALL_SKILLS if s.lower() == skill.lower()), "Technical"
        )

        # Keyword match (strong signal)
        in_resume_keyword = skill.lower() in resume_lower

        if in_resume_keyword:
            # Exact keyword match → strong
            status = "strong"
            score = _compute_skill_score(skill, resume_text, embedder, base=0.78)
        else:
            # Semantic match
            skill_embedding = embedder.encode([skill], convert_to_numpy=True)
            sem_score = float(
                cosine_similarity(resume_embedding, skill_embedding)[0][0]
            )
            if sem_score > 0.55:
                status = "partial"
                score = round(float(np.clip(sem_score, 0.4, 0.72)), 2)
            else:
                status = "missing"
                score = round(float(np.clip(sem_score * 0.5, 0.0, 0.35)), 2)

        results.append(
            SkillMatch(skill=skill, status=status, score=score, category=category)
        )

    # Sort: missing first, then partial, then strong
    order = {"missing": 0, "partial": 1, "strong": 2}
    results.sort(key=lambda x: (order[x.status], -x.score))

    return results[:25]  # Cap at 25 for UI readability


def _compute_skill_score(
    skill: str, resume_text: str, embedder, base: float = 0.75
) -> float:
    """Compute a nuanced score when skill is present (count mentions, context)."""
    count = len(
        re.findall(r"\b" + re.escape(skill.lower()) + r"\b", resume_text.lower())
    )
    # More mentions → slightly higher score, capped at 0.98
    return round(min(base + count * 0.04, 0.98), 2)


def detect_resume_weaknesses(resume_text: str, sections: dict) -> list:
    """
    Heuristic detection of common resume weaknesses.
    """
    from models.schemas import ResumeWeakness

    weaknesses = []

    # 1. No quantified achievements
    has_numbers = bool(re.search(r"\d+%|\$\d+|\d+x|\d+\+", resume_text))
    if not has_numbers:
        weaknesses.append(ResumeWeakness(
            section="Experience",
            issue="No quantified achievements found",
            severity="high",
            suggestion="Add metrics: 'Increased revenue by 30%', 'Reduced latency by 2x'"
        ))

    # 2. Weak action verbs
    weak_verbs = ["responsible for", "helped", "worked on", "assisted", "was involved"]
    found_weak = [v for v in weak_verbs if v in resume_text.lower()]
    if found_weak:
        weaknesses.append(ResumeWeakness(
            section="Experience",
            issue=f"Weak action verbs detected: {', '.join(found_weak[:3])}",
            severity="medium",
            suggestion="Replace with strong verbs: Led, Built, Architected, Delivered, Drove"
        ))

    # 3. Missing summary section
    if "summary" not in sections and "objective" not in resume_text.lower():
        weaknesses.append(ResumeWeakness(
            section="Summary",
            issue="No professional summary detected",
            severity="medium",
            suggestion="Add a 3-line summary highlighting your value proposition and target role"
        ))

    # 4. Too short
    word_count = len(resume_text.split())
    if word_count < 200:
        weaknesses.append(ResumeWeakness(
            section="Overall",
            issue=f"Resume is very short ({word_count} words)",
            severity="high",
            suggestion="Expand experience, projects, and skills sections for stronger signal"
        ))
    elif word_count > 1000:
        weaknesses.append(ResumeWeakness(
            section="Overall",
            issue=f"Resume may be too long ({word_count} words)",
            severity="low",
            suggestion="Trim to 1 page for < 5 years experience, 2 pages for senior roles"
        ))

    # 5. Missing skills section
    if "skills" not in sections:
        weaknesses.append(ResumeWeakness(
            section="Skills",
            issue="No dedicated skills section found",
            severity="medium",
            suggestion="Add a skills section with key technical and soft skills for ATS scanning"
        ))

    return weaknesses