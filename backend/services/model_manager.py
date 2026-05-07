"""
CareerLens — Optimized ML Model Manager
Lazy-loaded singleton for low-memory deployments (Render free tier safe)
"""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Lazy-loaded singleton model manager.
    Models load ONLY when first accessed.
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self._embedder: Optional[SentenceTransformer] = None

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def embedder(self) -> SentenceTransformer:
        """
        Load embedding model only when needed.
        """
        if self._embedder is None:
            logger.info("Loading embedding model...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded.")

        return self._embedder


# Global singleton
model_manager = ModelManager.get()