"""
CareerLens — Skill Gap Analyzer  (v2)
======================================
Key improvements over v1
------------------------
* Uses the shared SKILL_ALIASES + normalise_skill from scorer.py
  so Flask/FastAPI/Express all resolve to "backend api development" — no
  more false negatives from framework mismatches.
* Splits JD skills into REQUIRED vs OPTIONAL before building the heatmap.
* Missing OPTIONAL skills are shown but clearly labelled — they do not drive
  a "you failed" perception.
* Semantic threshold tuned to 0.42 to reduce false positives.
* Returns rich per-skill metadata: importance, category, matched aliases.
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
    normalise_skill,
)

try:
    from backend.models.schemas import SkillMatch, ResumeWeakness
except ImportError:
    from models.schemas import SkillMatch, ResumeWeakness


# ── Skill category lookup (for UI colour grouping) ────────────────────────────
_CATEGORY_MAP: dict[str, str] = {
    "python": "Language", "javascript": "Language", "typescript": "Language",
    "c++": "Language", "c#": "Language", "golang": "Language", "r language": "Language",

    "backend api development": "Framework", "microservices": "Architecture",
    "system design": "Architecture", "object oriented programming": "Concept",
    "functional programming": "Concept", "agile": "Process",
    "version control": "Tool",

    "machine learning": "AI/ML", "deep learning": "AI/ML",
    "natural language processing": "AI/ML", "computer vision": "AI/ML",
    "mlops": "AI/ML", "large language models": "AI/ML",
    "reinforcement learning": "AI/ML",
    "pytorch": "ML Library", "tensorflow": "ML Library",
    "scikit-learn": "ML Library", "hugging face": "ML Library",

    "sql": "Data", "nosql": "Data", "data analysis": "Data",
    "data engineering": "Data", "apache spark": "Data",
    "apache kafka": "Data", "apache airflow": "Data",

    "aws": "Cloud", "gcp": "Cloud", "azure": "Cloud",
    "docker": "DevOps", "kubernetes": "DevOps", "ci/cd": "DevOps",
    "infrastructure as code": "DevOps",
}

def _category(canonical: str) -> str:
    return _CATEGORY_MAP.get(canonical, "Technical")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def analyze_skill_gaps(resume_text: str, job_description: str) -> List[SkillMatch]:
    """
    Build a skill-gap heatmap for the dashboard.

    Returns up to 28 SkillMatch objects sorted as:
      1. Required + missing  (red — most actionable)
      2. Required + partial  (amber)
      3. Optional + missing  (softer red)
      4. Required + strong   (green)
      5. Optional + strong   (lighter green)

    The SkillMatch.score field is now a true 0-1 confidence,
    NOT a penalty-inflated distance.
    """
    embedder = model_manager.embedder

    # Encode resume once
    resume_chunks   = _smart_chunk(resume_text)
    resume_embs     = embedder.encode(resume_chunks, convert_to_numpy=True)
    resume_embedding = np.mean(resume_embs, axis=0, keepdims=True)

    # Get JD skills with importance classification
    jd_skills = _extract_jd_skills_with_importance(job_description)

    results: List[SkillMatch] = []

    for canonical, aliases, importance in jd_skills:
        status, confidence = _skill_present_score(
            canonical, aliases,
            resume_text, resume_embedding, embedder,
        )

        # Display name: use canonical but capitalise nicely
        display_name = _display_name(canonical, aliases, resume_text)

        results.append(SkillMatch(
            skill=display_name,
            status=status,
            score=confidence,
            category=f"{importance.title()} · {_category(canonical)}",
        ))

    # ── Sort ─────────────────────────────────────────────────────────────────
    _sort_key = {
        ("required", "missing"):  0,
        ("required", "partial"):  1,
        ("optional", "missing"):  2,
        ("required", "strong"):   3,
        ("optional", "partial"):  4,
        ("optional", "strong"):   5,
    }
    importance_by_display = {
        _display_name(c, a, resume_text): i
        for c, a, i in jd_skills
    }

    def sort_fn(sm: SkillMatch) -> tuple:
        imp = importance_by_display.get(sm.skill, "required")
        pri = _sort_key.get((imp, sm.status), 6)
        return (pri, -sm.score)

    results.sort(key=sort_fn)
    return results[:28]


def get_missing_required_skills(resume_text: str, job_description: str) -> list[str]:
    """
    Return a flat list of canonical skill names that are REQUIRED in the JD
    but absent or partial in the resume. Used by the insights engine.
    """
    embedder = model_manager.embedder
    resume_chunks    = _smart_chunk(resume_text)
    resume_embs      = embedder.encode(resume_chunks, convert_to_numpy=True)
    resume_embedding = np.mean(resume_embs, axis=0, keepdims=True)

    jd_skills = _extract_jd_skills_with_importance(job_description)
    missing: list[str] = []

    for canonical, aliases, importance in jd_skills:
        if importance != "required":
            continue
        status, _ = _skill_present_score(
            canonical, aliases, resume_text, resume_embedding, embedder
        )
        if status in ("missing", "partial"):
            missing.append(canonical)

    return missing


def detect_resume_weaknesses(resume_text: str, sections: dict) -> List[ResumeWeakness]:
    """
    Heuristic detection of common resume anti-patterns.
    Returns a list of ResumeWeakness objects, each with:
      section, issue, severity (high/medium/low), suggestion.
    """
    weaknesses: List[ResumeWeakness] = []
    word_count = len(resume_text.split())
    rl = resume_text.lower()

    # 1. No quantified achievements
    if not re.search(r"\d+\s*%|\$\s*\d+|\d+\s*x\b|\d+\s*\+|\d+\s*(million|billion|k\b)",
                     resume_text, re.I):
        weaknesses.append(ResumeWeakness(
            section="Experience",
            issue="No quantified achievements detected",
            severity="high",
            suggestion=(
                "Add impact metrics to every bullet: "
                "'Reduced inference latency by 40%', 'Trained model on 50M+ samples', "
                "'Increased conversion rate by $120K ARR'"
            ),
        ))

    # 2. Weak action verbs
    weak_phrases = [
        "responsible for", "worked on", "helped with",
        "assisted", "was involved in", "participated in",
    ]
    found_weak = [p for p in weak_phrases if p in rl]
    if found_weak:
        weaknesses.append(ResumeWeakness(
            section="Experience",
            issue=f"Passive phrasing detected: '{found_weak[0]}'",
            severity="medium",
            suggestion=(
                "Replace with strong action verbs: "
                "Architected · Engineered · Delivered · Drove · Spearheaded · Optimised"
            ),
        ))

    # 3. Missing professional summary
    has_summary = (
        "summary" in sections
        or "objective" in rl
        or "profile" in rl
        or "about" in sections
    )
    if not has_summary:
        weaknesses.append(ResumeWeakness(
            section="Summary",
            issue="No professional summary detected",
            severity="medium",
            suggestion=(
                "Add a 3-sentence summary: "
                "(1) Years of experience + title, "
                "(2) Core technical expertise, "
                "(3) Key achievement or career goal"
            ),
        ))

    # 4. Length checks
    if word_count < 200:
        weaknesses.append(ResumeWeakness(
            section="Overall",
            issue=f"Resume is very thin ({word_count} words)",
            severity="high",
            suggestion=(
                "Expand experience bullets (4–6 per role), add a projects section, "
                "and include a dedicated skills list for ATS parsing"
            ),
        ))
    elif word_count > 1200:
        weaknesses.append(ResumeWeakness(
            section="Overall",
            issue=f"Resume may be too long ({word_count} words)",
            severity="low",
            suggestion=(
                "Target 1 page for < 5 years XP, 2 pages for senior roles. "
                "Cut older or irrelevant roles."
            ),
        ))

    # 5. No dedicated skills section
    if "skills" not in sections and "technical" not in rl[:300]:
        weaknesses.append(ResumeWeakness(
            section="Skills",
            issue="No dedicated skills section found",
            severity="medium",
            suggestion=(
                "Add a 'Technical Skills' section with grouped keywords "
                "(Languages · Frameworks · Cloud · Tools) — critical for ATS keyword scanning"
            ),
        ))

    # 6. No project / portfolio signals
    has_projects = any(k in rl for k in ["github", "project", "built", "portfolio", "open source"])
    if not has_projects:
        weaknesses.append(ResumeWeakness(
            section="Projects",
            issue="No project or GitHub signals found",
            severity="low",
            suggestion=(
                "Add a Projects section with 2–3 shipped projects, "
                "each with tech stack + measurable outcome"
            ),
        ))

    return weaknesses


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _display_name(canonical: str, aliases: list[str], resume_text: str) -> str:
    """
    If an alias was actually found in the resume, show that alias
    (e.g. 'Flask' instead of 'backend api development').
    Otherwise return a title-cased canonical.
    """
    resume_lower = resume_text.lower()
    for alias in aliases:
        if re.search(r"\b" + re.escape(alias.lower()) + r"\b", resume_lower):
            return alias.title() if alias.islower() else alias
    # Title-case canonical
    return canonical.title()