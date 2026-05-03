"""
CareerLens — Semantic Scorer Service
Computes cosine similarity between resume and JD embeddings.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .model_manager import model_manager
import re


def compute_match_score(resume_text: str, job_description: str) -> float:
    """
    Compute semantic similarity score between resume and JD.
    Returns a float 0–100.
    """
    embedder = model_manager.embedder

    # Chunk texts for better representation
    resume_chunks = _chunk_text(resume_text, max_words=200)
    jd_chunks = _chunk_text(job_description, max_words=200)

    resume_embeddings = embedder.encode(resume_chunks, convert_to_numpy=True)
    jd_embeddings = embedder.encode(jd_chunks, convert_to_numpy=True)

    # Mean-pool chunks
    resume_vec = np.mean(resume_embeddings, axis=0, keepdims=True)
    jd_vec = np.mean(jd_embeddings, axis=0, keepdims=True)

    similarity = cosine_similarity(resume_vec, jd_vec)[0][0]

    # Scale to 0–100 with calibration
    # Raw cosine for text is typically 0.3–0.9; map to 0–100
    score = float(np.clip((similarity - 0.2) / 0.7 * 100, 0, 100))
    return round(score, 1)


def get_match_label(score: float) -> str:
    if score >= 75:
        return "Excellent"
    elif score >= 55:
        return "Good"
    elif score >= 35:
        return "Fair"
    else:
        return "Poor"


def _chunk_text(text: str, max_words: int = 200) -> list:
    """Split text into overlapping chunks of max_words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    step = max_words // 2
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
        if i + max_words >= len(words):
            break
    return chunks if chunks else [text]


def extract_job_title(job_description: str) -> str:
    """Attempt to extract job title from JD text."""
    lines = job_description.strip().split("\n")
    # First non-empty line is often the title
    for line in lines[:5]:
        line = line.strip()
        if 5 < len(line) < 80 and not line.endswith(":"):
            return line

    # Fallback: look for "Position:" or "Role:" keywords
    for line in lines:
        match = re.search(
            r"(?:position|role|title|job)[:\s]+(.+)", line, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()[:80]

    return "Target Role"