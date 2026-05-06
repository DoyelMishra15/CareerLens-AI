# CareerLens — Dockerfile
# Multi-stage, optimized for CPU-only inference on free tiers

FROM python:3.11-slim AS base

# System deps for PyMuPDF + spaCy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Download spaCy model ─────────────────────────────────────────────────────
RUN python -m spacy download en_core_web_sm

# ── Pre-download transformer models (baked into image) ──────────────────────
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('sentence-transformers cached')"

RUN python -c "\
from transformers import pipeline; \
pipeline('zero-shot-classification', model='cross-encoder/nli-MiniLM2-L6-H768', device=-1); \
print('zero-shot classifier cached')"

# ── Copy application code ────────────────────────────────────────────────────
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# ── Run ─────────────────────────────────────────────────────────────────────
EXPOSE 8000
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]