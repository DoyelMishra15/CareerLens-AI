"""
CareerLens — /api/analyze Router
Accepts PDF + JD, returns full analysis asynchronously.
"""

import asyncio
import uuid
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from models.schemas import AnalyzeResponse
from services import (
    extract_text_from_pdf, detect_resume_sections,
    compute_match_score, get_match_label, extract_job_title,
    analyze_skill_gaps, detect_resume_weaknesses,
)

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resume(
    resume: UploadFile = File(..., description="Resume PDF"),
    job_description: str = Form(..., min_length=50, description="Job description text"),
):
    """
    Full async analysis pipeline:
    1. Parse PDF
    2. Compute semantic match score
    3. Analyze skill gaps
    4. Detect resume weaknesses
    All steps run concurrently where possible.
    """
    start = time.monotonic()

    # ── 1. Parse PDF ──────────────────────────────────────────────────────────
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await resume.read()
    if len(file_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large (max 10MB).")

    # Run CPU-bound parsing in thread pool
    loop = asyncio.get_event_loop()
    resume_text, page_count = await loop.run_in_executor(
        None, extract_text_from_pdf, file_bytes
    )

    if len(resume_text.strip()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Could not extract readable text from PDF. Try a text-based PDF."
        )

    # ── 2. Async parallel processing ──────────────────────────────────────────
    sections_task = loop.run_in_executor(None, detect_resume_sections, resume_text)
    score_task = loop.run_in_executor(None, compute_match_score, resume_text, job_description)
    job_title_task = loop.run_in_executor(None, extract_job_title, job_description)

    sections, match_score, job_title = await asyncio.gather(
        sections_task, score_task, job_title_task
    )

    # Skills and weaknesses depend on sections
    skills_task = loop.run_in_executor(None, analyze_skill_gaps, resume_text, job_description)
    weaknesses_task = loop.run_in_executor(
        None, detect_resume_weaknesses, resume_text, sections
    )

    skills, weaknesses = await asyncio.gather(skills_task, weaknesses_task)

    elapsed = time.monotonic() - start

    return AnalyzeResponse(
        match_score=match_score,
        match_label=get_match_label(match_score),
        skills=skills,
        weaknesses=weaknesses,
        resume_text_preview=resume_text[:800],  # First 800 chars for UI
        job_title=job_title,
        analysis_id=str(uuid.uuid4())[:8],
    )