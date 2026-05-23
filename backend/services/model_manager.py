"""
CareerLens — ML Model Manager
Classifier removed; CPU-only torch keeps memory under 512MB.
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
        logger.info("Loading ML models...")
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found.")
            self._nlp = None
        self._loaded = True
        logger.info("Models loaded.")

    @property
    def embedder(self) -> SentenceTransformer:
        if not self._embedder:
            self.load_all()
        return self._embedder

    @property
    def classifier(self):
        # Removed — not used anywhere, saves ~400MB RAM
        return None

    @property
    def nlp(self):
        if not self._nlp:
            self.load_all()
        return self._nlp


model_manager = ModelManager.get()