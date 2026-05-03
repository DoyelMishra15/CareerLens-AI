"""
CareerLens — /api/rewrite Router
Rewrites individual resume bullet points for maximum impact.
"""

import asyncio
from fastapi import APIRouter
from models.schemas import RewriteRequest, RewriteResponse
from services import rewrite_bullet_point

router = APIRouter()


@router.post("/rewrite", response_model=RewriteResponse)
async def rewrite_bullet(request: RewriteRequest):
    """
    Rewrite a single resume bullet point.
    Uses rule-based enhancement + keyword injection.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        rewrite_bullet_point,
        request.bullet_point,
        request.job_description,
        request.context or "Experience",
    )
    return result