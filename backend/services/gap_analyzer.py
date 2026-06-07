"""
CareerLens -- Gap Analyzer v3
Works with scorer v3: implied skills, seniority-aware, strengths + gaps.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .model_manager import model_manager
from .scorer import (
    SKILL_ALIASES,
    _ALIAS_TO_CANONICAL,
    _extract_jd_skills_with_importance,
    _skill_present_score,
    _smart_chunk,
    compute_implied_skills,
    detect_seniority,
)

try:
    from backend.models.schemas import SkillMatch, ResumeWeakness
except ImportError:
    from models.schemas import SkillMatch, ResumeWeakness


# Category labels for UI grouping
_CATEGORY_MAP = {
    "python": "Language", "javascript": "Language", "typescript": "Language",
    "java": "Language", "c++": "Language", "c#": "Language",
    "golang": "Language", "r language": "Language",
    "backend api development": "Framework", "microservices": "Architecture",
    "machine learning": "AI/ML", "deep learning": "AI/ML",
    "natural language processing": "AI/ML", "computer vision": "AI/ML",
    "mlops": "AI/ML", "large language models": "AI/ML",
    "reinforcement learning": "AI/ML", "time series analysis": "AI/ML",
    "pytorch": "ML Library", "tensorflow": "ML Library",
    "scikit-learn": "ML Library", "hugging face": "ML Library",
    "numpy": "ML Library", "pandas": "ML Library",
    "sql": "Data", "nosql": "Data", "data analysis": "Data",
    "data engineering": "Data", "apache spark": "Data",
    "aws": "Cloud", "gcp": "Cloud", "azure": "Cloud",
    "docker": "DevOps", "kubernetes": "DevOps", "ci/cd": "DevOps",
    "git": "Tool", "object oriented programming": "Concept",
    "data structures": "Concept", "agile": "Process",
    "system design": "Architecture", "problem solving": "Concept",
}


def _category(canonical: str) -> str:
    return _CATEGORY_MAP.get(canonical, "Technical")


def _display_name(canonical: str, aliases: list, resume_text: str) -> str:
    """Show the alias actually found in the resume, fallback to title-cased canonical."""
    resume_lower = resume_text.lower()
    for alias in aliases:
        if re.search(r"\b" + re.escape(alias.lower()) + r"\b", resume_lower):
            return alias.title() if alias.islower() else alias
    return canonical.title()


def analyze_skill_gaps(resume_text: str, job_description: str) -> List[SkillMatch]:
    """
    Build the skill-gap heatmap.
    Now includes 'implied' status so the UI can show skills the candidate
    probably has even without explicit keywords.

    Sort order:
      1. Required + missing   (most urgent -- shown first)
      2. Required + partial
      3. Optional + missing
      4. Required + implied
      5. Required + strong
      6. Optional + implied / strong
    """
    embedder = model_manager.embedder

    resume_chunks    = _smart_chunk(resume_text)
    resume_embs      = embedder.encode(resume_chunks, convert_to_numpy=True)
    resume_embedding = np.mean(resume_embs, axis=0, keepdims=True)
    implied_scores   = compute_implied_skills(resume_text)
    jd_skills        = _extract_jd_skills_with_importance(job_description)

    results: List[SkillMatch] = []

    for canonical, aliases, importance in jd_skills:
        status, confidence = _skill_present_score(
            canonical, aliases, resume_text,
            resume_embedding, embedder, implied_scores,
        )

        display  = _display_name(canonical, aliases, resume_text)
        category = f"{importance.title()} - {_category(canonical)}"

        results.append(SkillMatch(
            skill=display,
            status=status,
            score=confidence,
            category=category,
        ))

    # Sort
    _priority = {
        ("required", "missing"):  0,
        ("required", "partial"):  1,
        ("optional", "missing"):  2,
        ("required", "implied"):  3,
        ("required", "strong"):   4,
        ("optional", "implied"):  5,
        ("optional", "partial"):  5,
        ("optional", "strong"):   6,
    }

    imp_lookup = {
        _display_name(c, a, resume_text): i
        for c, a, i in jd_skills
    }

    def sort_key(sm: SkillMatch):
        imp = imp_lookup.get(sm.skill, "required")
        pri = _priority.get((imp, sm.status), 7)
        return (pri, -sm.score)

    results.sort(key=sort_key)
    return results[:28]


def get_missing_required_skills(resume_text: str, job_description: str) -> list:
    """Return list of required skill canonical names that are missing or partial."""
    embedder         = model_manager.embedder
    resume_chunks    = _smart_chunk(resume_text)
    resume_embs      = embedder.encode(resume_chunks, convert_to_numpy=True)
    resume_embedding = np.mean(resume_embs, axis=0, keepdims=True)
    implied_scores   = compute_implied_skills(resume_text)
    jd_skills        = _extract_jd_skills_with_importance(job_description)

    missing = []
    for canonical, aliases, importance in jd_skills:
        if importance != "required":
            continue
        status, _ = _skill_present_score(
            canonical, aliases, resume_text,
            resume_embedding, embedder, implied_scores,
        )
        if status in ("missing", "partial"):
            missing.append(canonical)
    return missing


def detect_resume_weaknesses(resume_text: str, sections: dict) -> List[ResumeWeakness]:
    """Context-aware weakness detection with actionable, specific suggestions."""
    weaknesses: List[ResumeWeakness] = []
    rl         = resume_text.lower()
    word_count = len(resume_text.split())

    # 1. No quantified achievements
    # Matches: 30%, $50, 2x, 50+, 50K users, "approximately 30 percent", "2 times", etc.
    has_numbers = bool(re.search(
        r"\d+\s*%"
        r"|\$\s*\d+"
        r"|\d+\s*x\b"
        r"|\d+\s*\+"
        r"|\d+\s*(million|billion|k\b|users|customers|accuracy|f1|requests)"
        r"|approximately\s+\d+"
        r"|\d+\s+percent"
        r"|\d+\s+times\b"
        r"|\d+\s*(seconds?|minutes?|hours?)\b",
        resume_text, re.I
    ))
    if not has_numbers:
        weaknesses.append(ResumeWeakness(
            section="Experience",
            issue="No quantified achievements found",
            severity="high",
            suggestion=(
                "Add impact numbers to every bullet: "
                "'Achieved 92% model accuracy', 'Processed 50K+ records', "
                "'Reduced runtime by 40%', 'Served 1000+ users'"
            ),
        ))

    # 2. Weak action verbs
    weak = [p for p in [
        "responsible for", "worked on", "helped with",
        "assisted", "was involved in", "participated in",
    ] if p in rl]
    if weak:
        weaknesses.append(ResumeWeakness(
            section="Experience",
            issue=f"Passive phrasing: '{weak[0]}'",
            severity="medium",
            suggestion=(
                "Use strong action verbs: "
                "Built, Developed, Designed, Implemented, Engineered, Optimised, Deployed"
            ),
        ))

    # 3. Missing summary
    has_summary = (
        "summary" in sections
        or any(k in rl[:400] for k in ["summary", "objective", "profile", "about me"])
    )
    if not has_summary:
        weaknesses.append(ResumeWeakness(
            section="Summary",
            issue="No professional summary section",
            severity="medium",
            suggestion=(
                "Add 2-3 sentences: (1) Who you are + year of study, "
                "(2) Core technical strengths, (3) What you're seeking"
            ),
        ))

    # 4. No deployment signals
    has_deploy = any(k in rl for k in [
        "deployed", "live", "hosted", "github pages", "render", "heroku",
        "aws", "docker", "production", "api endpoint",
    ])
    if not has_deploy:
        weaknesses.append(ResumeWeakness(
            section="Projects",
            issue="No deployed projects detected",
            severity="medium",
            suggestion=(
                "Deploy at least one project — even a simple Flask/FastAPI app on Render (free). "
                "Add the live URL to your resume. This massively differentiates you."
            ),
        ))

    # 5. Resume length
    if word_count < 200:
        weaknesses.append(ResumeWeakness(
            section="Overall",
            issue=f"Resume is very short ({word_count} words)",
            severity="high",
            suggestion=(
                "Expand each project bullet to 3-4 lines: "
                "what you built, how you built it, what it achieved. "
                "Add a Technical Skills section with grouped keywords."
            ),
        ))
    elif word_count > 1200:
        weaknesses.append(ResumeWeakness(
            section="Overall",
            issue=f"Resume may be too long ({word_count} words)",
            severity="low",
            suggestion="Keep to 1 page for students / < 3 years experience.",
        ))

    # 6. No skills section
    if "skills" not in sections and "technical" not in rl[:500]:
        weaknesses.append(ResumeWeakness(
            section="Skills",
            issue="No dedicated Technical Skills section",
            severity="medium",
            suggestion=(
                "Add a grouped Skills section: "
                "Languages | Frameworks | Tools | Concepts. "
                "Critical for ATS keyword scanning."
            ),
        ))

    # 7. No GitHub / portfolio link
    has_github = "github.com" in rl or "github:" in rl
    if not has_github:
        weaknesses.append(ResumeWeakness(
            section="Projects",
            issue="No GitHub profile or project links",
            severity="medium",
            suggestion=(
                "Add your GitHub URL to the header and link each project. "
                "Recruiters click GitHub links — make sure repos have good READMEs."
            ),
        ))

    return weaknesses