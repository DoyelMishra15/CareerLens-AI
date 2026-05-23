"""
CareerLens — ML Model Manager
Uses fastembed (ONNX-based, no PyTorch) to stay under 512MB RAM.
Wraps fastembed to match the sentence-transformers .encode() interface
so no other file needs changing.
"""

import logging
import numpy as np
from typing import Optional, List
import spacy

logger = logging.getLogger(__name__)


class EmbedderWrapper:
    """Wraps fastembed.TextEmbedding to match sentence-transformers API."""

    def __init__(self):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def encode(self, texts: List[str], convert_to_numpy: bool = True) -> np.ndarray:
        embeddings = list(self._model.embed(texts))
        return np.array(embeddings)


class ModelManager:
    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self._embedder: Optional[EmbedderWrapper] = None
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

        self._embedder = EmbedderWrapper()
        logger.info("fastembed model loaded.")

        try:
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy loaded.")
        except OSError:
            logger.warning("spaCy model not found.")
            self._nlp = None

        self._loaded = True
        logger.info("All models ready.")

    @property
    def embedder(self) -> EmbedderWrapper:
        if not self._embedder:
            self.load_all()
        return self._embedder

    @property
    def classifier(self):
        return None  # Not used anywhere

    @property
    def nlp(self):
        if not self._nlp:
            self.load_all()
        return self._nlp


model_manager = ModelManager.get()