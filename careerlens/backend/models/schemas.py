"""
CareerLens — Data Models (Pydantic)
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


# ── Request Models ────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    job_description: str = Field(..., min_length=50, description="The full job description text")

class RewriteRequest(BaseModel):
    bullet_point: str = Field(..., description="Resume bullet point to rewrite")
    job_description: str = Field(..., description="Target job description for context")
    context: Optional[str] = Field(None, description="Optional: section name (e.g., 'Experience')")

class InsightsRequest(BaseModel):
    missing_skills: List[str] = Field(..., description="Skills identified as missing")
    resume_score: float = Field(..., ge=0, le=100, description="Current semantic match score")
    job_title: Optional[str] = Field(None, description="Target job title extracted from JD")


# ── Response Models ───────────────────────────────────────────────────────────

class SkillMatch(BaseModel):
    skill: str
    status: str        # "strong" | "partial" | "missing"
    score: float       # 0.0 – 1.0
    category: str      # e.g. "Technical", "Soft Skill", "Tool"

class ResumeWeakness(BaseModel):
    section: str
    issue: str
    severity: str      # "high" | "medium" | "low"
    suggestion: str

class AnalyzeResponse(BaseModel):
    match_score: float
    match_label: str   # "Excellent" | "Good" | "Fair" | "Poor"
    skills: List[SkillMatch]
    weaknesses: List[ResumeWeakness]
    resume_text_preview: str
    job_title: str
    analysis_id: str

class RewriteResponse(BaseModel):
    original: str
    rewritten: str
    improvement_reason: str
    impact_keywords: List[str]

class LearningPath(BaseModel):
    skill: str
    priority: int       # 1 = highest
    resources: List[str]
    estimated_time: str
    impact_score: float  # how much the score improves if learned

class InsightsResponse(BaseModel):
    top_skills_to_learn: List[LearningPath]
    estimated_score_after_improvement: float
    career_growth_tips: List[str]
    job_market_insight: str