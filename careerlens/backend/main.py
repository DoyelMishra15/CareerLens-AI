"""
CareerLens — AI-Powered Career Intelligence & Resume Optimization Platform
FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os

from routers import analyze, rewrite, insights

# ── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CareerLens API",
    description="AI-Powered Career Intelligence & Resume Optimization Platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(analyze.router, prefix="/api", tags=["Analysis"])
app.include_router(rewrite.router, prefix="/api", tags=["Rewrite"])
app.include_router(insights.router, prefix="/api", tags=["Insights"])

# ── Static Files ──────────────────────────────────────────────────────────────
static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "static")
templates_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "templates")

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(os.path.join(templates_dir, "index.html"))

@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    return FileResponse(os.path.join(templates_dir, "dashboard.html"))

@app.get("/health")
async def health():
    return {"status": "ok", "service": "CareerLens API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)