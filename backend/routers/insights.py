"""
CareerLens — /api/insights Router  (v2)
Now receives pre-classified missing_required_skills from frontend,
falling back to the gap analyzer if not provided.
"""

import asyncio
from fastapi import APIRouter

try:
    from backend.models.schemas import InsightsRequest, InsightsResponse
    from backend.services import generate_insights
except ImportError:
    from models.schemas import InsightsRequest, InsightsResponse
    from services import generate_insights

router = APIRouter()


@router.post("/insights", response_model=InsightsResponse)
async def get_career_insights(request: InsightsRequest):
    """
    Career Growth Intelligence:
      - Top 3 skills to learn (prioritised by impact)
      - Projected score after improvement
      - Career tips
      - Market insight
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