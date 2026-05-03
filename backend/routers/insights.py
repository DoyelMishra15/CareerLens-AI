"""
CareerLens — /api/insights Router
Career Growth Intelligence endpoint.
"""

import asyncio
from fastapi import APIRouter
from models.schemas import InsightsRequest, InsightsResponse
from services import generate_insights

router = APIRouter()


@router.post("/insights", response_model=InsightsResponse)
async def get_career_insights(request: InsightsRequest):
    """
    Generate Career Growth Intelligence:
    - Top 3 skills to learn next (with resources + time estimates)
    - Estimated score after improvement
    - Career growth tips
    - Job market insight
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        generate_insights,
        request.missing_skills,
        request.resume_score,
        request.job_title or "",
    )
    return result