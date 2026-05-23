"""
CareerLens — ML Model Manager
Singleton loader. Classifier removed to fit Render free tier (512MB RAM).
"""

import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
import spacy

logger = logging.getLogger(__name__)


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self._embedder: Optional[SentenceTransformer] = None
        self._nlp = None
        self._loaded = False

    @classmethod
    def get(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_all(self):
        if self._loaded:
            return
        logger.info("🔄 Loading ML models...")

        logger.info("  Loading sentence-transformers/all-MiniLM-L6-v2...")
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

        logger.info("  Loading spaCy en_core_web_sm...")
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self._nlp = None

        self._loaded = True
        logger.info("✅ Models loaded.")

    @property
    def embedder(self) -> SentenceTransformer:
        if not self._embedder:
            self.load_all()
        return self._embedder

    @property
    def classifier(self):
        # Classifier disabled — uses too much RAM on free tier.
        # classify_jd_skill_importance() uses regex anyway, not this model.
        return None

    @property
    def nlp(self):
        if not self._nlp:
            self.load_all()
        return self._nlp


model_manager = ModelManager.get()