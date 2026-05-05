"""
CareerLens — /api/analyze Router  (v2)
Uses weighted ATS scoring + importance-aware gap analysis.
"""

import asyncio
import uuid
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

try:
    from backend.models.schemas import AnalyzeResponse
    from backend.services import (
        extract_text_from_pdf, detect_resume_sections,
        compute_match_score, compute_detailed_score,
        get_match_label, extract_job_title,
        analyze_skill_gaps, detect_resume_weaknesses,
    )
except ImportError:
    from models.schemas import AnalyzeResponse
    from services import (
        extract_text_from_pdf, detect_resume_sections,
        compute_match_score, compute_detailed_score,
        get_match_label, extract_job_title,
        analyze_skill_gaps, detect_resume_weaknesses,
    )

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(..., min_length=50),
):
    """
    Full async analysis pipeline (v2 — ATS-realistic scoring).

    Steps (concurrent where possible):
      1. PDF parse
      2. Section detection
      3. Weighted semantic + skill score
      4. Importance-aware gap analysis
      5. Weakness detection
    """
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    file_bytes = await resume.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10 MB).")

    loop = asyncio.get_event_loop()

    # Step 1: parse PDF
    resume_text, _ = await loop.run_in_executor(
        None, extract_text_from_pdf, file_bytes
    )
    if len(resume_text.strip()) < 50:
        raise HTTPException(422, "Could not extract text. Use a text-based PDF.")

    # Step 2+3+4 in parallel
    sections_task   = loop.run_in_executor(None, detect_resume_sections, resume_text)
    score_task      = loop.run_in_executor(None, compute_match_score, resume_text, job_description)
    job_title_task  = loop.run_in_executor(None, extract_job_title, job_description)

    sections, match_score, job_title = await asyncio.gather(
        sections_task, score_task, job_title_task
    )

    # Step 5+6 — depend on sections
    skills_task     = loop.run_in_executor(None, analyze_skill_gaps, resume_text, job_description)
    weaknesses_task = loop.run_in_executor(None, detect_resume_weaknesses, resume_text, sections)

    skills, weaknesses = await asyncio.gather(skills_task, weaknesses_task)

    return AnalyzeResponse(
        match_score=match_score,
        match_label=get_match_label(match_score),
        skills=skills,
        weaknesses=weaknesses,
        resume_text_preview=resume_text[:800],
        job_title=job_title,
        analysis_id=str(uuid.uuid4())[:8],
    )