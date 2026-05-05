"""
CareerLens — Model Preloader
Run this once after install to download + cache all models.
Usage: python preload_models.py
"""

import time
import sys

print("\n◈ CareerLens — Model Preloader")
print("══════════════════════════════════════")
print("This runs once and caches models to disk.\n")

# ── sentence-transformers ──────────────────────────────────────────────────
print("→ [1/3] Downloading sentence-transformers/all-MiniLM-L6-v2...")
t = time.time()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
# Warm up
_ = model.encode(["test sentence"])
print(f"   ✅ Done ({time.time()-t:.1f}s)\n")

# ── transformers zero-shot ─────────────────────────────────────────────────
print("→ [2/3] Downloading cross-encoder/nli-MiniLM2-L6-H768 (zero-shot)...")
t = time.time()
from transformers import pipeline
clf = pipeline(
    "zero-shot-classification",
    model="cross-encoder/nli-MiniLM2-L6-H768",
    device=-1,
)
# Warm up
_ = clf("test", candidate_labels=["technical skill", "soft skill"])
print(f"   ✅ Done ({time.time()-t:.1f}s)\n")

# ── spaCy ──────────────────────────────────────────────────────────────────
print("→ [3/3] Loading spaCy en_core_web_sm...")
t = time.time()
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    _ = nlp("Hello world")
    print(f"   ✅ Done ({time.time()-t:.1f}s)\n")
except OSError:
    print("   ⚠️  spaCy model not found. Run: python -m spacy download en_core_web_sm\n")

print("══════════════════════════════════════")
print("✅ All models cached. First API request will be fast.")
print("\nNow run: cd backend && python main.py\n")