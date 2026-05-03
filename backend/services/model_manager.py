"""
CareerLens — ML Model Manager
Singleton loader for all ML models. Ensures models are loaded ONCE and reused.
"""

import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import spacy

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton class that loads and caches all ML models.
    Call ModelManager.get() to get the shared instance.
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self._embedder: Optional[SentenceTransformer] = None
        self._classifier = None
        self._nlp = None
        self._loaded = False

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_all(self):
        """Load all models (called once at startup)."""
        if self._loaded:
            return
        logger.info("🔄 Loading ML models (one-time)...")

        # 1. Sentence Transformer — lightweight, fast on CPU
        logger.info("  Loading sentence-transformers/all-MiniLM-L6-v2...")
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # 2. Zero-shot classifier — for skill categorization
        logger.info("  Loading zero-shot classifier...")
        self._classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-MiniLM2-L6-H768",
            device=-1,  # CPU
        )

        # 3. spaCy — for NLP (NER, keyword extraction)
        logger.info("  Loading spaCy en_core_web_sm...")
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self._nlp = None

        self._loaded = True
        logger.info("✅ All models loaded.")

    @property
    def embedder(self) -> SentenceTransformer:
        if not self._embedder:
            self.load_all()
        return self._embedder

    @property
    def classifier(self):
        if not self._classifier:
            self.load_all()
        return self._classifier

    @property
    def nlp(self):
        if not self._nlp:
            self.load_all()
        return self._nlp


# Global singleton
model_manager = ModelManager.get()