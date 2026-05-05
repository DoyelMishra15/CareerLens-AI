"""
CareerLens — AI-Powered Career Intelligence & Resume Optimization Platform
FastAPI Application Entry Point

Run from project root:
    python -m uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os

# ── Package-relative imports (works whether run as module or directly) ────────
try:
    from backend.routers import analyze, rewrite, insights
except ImportError:
    from routers import analyze, rewrite, insights

# ── Resolve paths relative to THIS file (robust regardless of cwd) ────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))  # .../backend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                    # .../CareerLens-AI/
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# Support two frontend layouts:
#   Layout A (zip):  frontend/static/ + frontend/templates/
#   Layout B (repo): frontend/css|js/ + frontend/*.html  (flat)
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
if not os.path.isdir(STATIC_DIR):
    STATIC_DIR = FRONTEND_DIR          # flat — serve whole frontend dir

TEMPLATES_DIR = os.path.join(FRONTEND_DIR, "templates")
if not os.path.isdir(TEMPLATES_DIR):
    TEMPLATES_DIR = FRONTEND_DIR       # flat — html files at root

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
app.include_router(analyze.router,  prefix="/api", tags=["Analysis"])
app.include_router(rewrite.router,  prefix="/api", tags=["Rewrite"])
app.include_router(insights.router, prefix="/api", tags=["Insights"])

# ── Static Files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(os.path.join(TEMPLATES_DIR, "index.html"))

@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    return FileResponse(os.path.join(TEMPLATES_DIR, "dashboard.html"))

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "CareerLens API",
        "frontend_dir": FRONTEND_DIR,
        "static_dir": STATIC_DIR,
    }

if __name__ == "__main__":
    # python backend/main.py  — runs from project root
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)