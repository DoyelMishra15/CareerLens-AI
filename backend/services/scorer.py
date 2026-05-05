"""
CareerLens — ATS-Realistic Scoring Engine  (v2)
================================================
Formula:
    final = 0.50 * semantic_sim
          + 0.30 * required_skill_match   (only adds, never penalises)
          + 0.20 * optional_skill_bonus   (pure bonus — zero if all missing)

Why this works like a real ATS
-------------------------------
• Semantic sim catches paraphrased experience even with no exact keywords.
• Required skill match rewards covering the must-haves proportionally.
• Optional bonus only adds — a candidate missing all nice-to-haves still gets
  the full semantic + required contribution (no cliff-edge penalty).
• Raw cosine is re-calibrated: professional résumé <-> JD pairs typically land
  between 0.25–0.75. We map that range linearly to 0–100 so a genuinely
  aligned doc lands proper score.
"""

from __future__ import annotations

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .model_manager import model_manager


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SKILL NORMALISATION MAP
#     canonical form -> list of aliases that map to it.
#     All comparison is done on normalised forms.
# ══════════════════════════════════════════════════════════════════════════════
SKILL_ALIASES: dict[str, list[str]] = {
    # Programming languages
    "python":                       ["python3", "py"],
    "javascript":                   ["js", "es6", "es2015", "ecmascript"],
    "typescript":                   ["ts"],
    "c++":                          ["cpp", "c plus plus"],
    "c#":                           ["csharp", "c sharp", ".net"],
    "golang":                       ["go"],
    "r language":                   ["r programming"],

    # Backend / API
    "backend api development":      [
        "fastapi", "flask", "django", "express", "spring boot",
        "rest api", "restful api", "graphql", "api development",
        "web api", "node.js", "nodejs",
    ],
    "microservices":                ["micro-services", "service oriented architecture", "soa"],

    # ML / AI
    "machine learning":             ["ml", "statistical learning", "predictive modelling"],
    "deep learning":                ["dl", "neural networks", "ann", "dnn"],
    "natural language processing":  ["nlp", "text processing", "computational linguistics"],
    "computer vision":              ["cv", "image recognition", "object detection"],
    "mlops":                        ["ml ops", "ml pipeline", "model deployment", "model serving"],
    "large language models":        ["llm", "llms", "generative ai", "gen ai", "gpt"],
    "reinforcement learning":       ["rl", "rlhf"],

    # ML libraries
    "pytorch":                      ["torch"],
    "tensorflow":                   ["tf", "keras"],
    "scikit-learn":                 ["sklearn", "scikit learn"],
    "hugging face":                 ["huggingface", "transformers library", "hf"],

    # Data
    "sql":                          ["mysql", "postgresql", "postgres", "sqlite",
                                     "t-sql", "pl/sql", "database querying"],
    "nosql":                        ["mongodb", "cassandra", "dynamodb", "couchdb"],
    "data analysis":                ["data analytics", "eda", "exploratory data analysis"],
    "data engineering":             ["data pipelines", "etl", "elt", "data integration"],
    "apache spark":                 ["spark", "pyspark"],
    "apache kafka":                 ["kafka", "event streaming"],
    "apache airflow":               ["airflow", "workflow orchestration"],

    # Cloud / DevOps
    "aws":                          ["amazon web services", "s3", "ec2", "lambda",
                                     "sagemaker", "boto3"],
    "gcp":                          ["google cloud", "google cloud platform", "bigquery",
                                     "vertex ai"],
    "azure":                        ["microsoft azure", "azure ml"],
    "docker":                       ["containerisation", "containerization", "containers"],
    "kubernetes":                   ["k8s", "container orchestration"],
    "ci/cd":                        ["continuous integration", "continuous deployment",
                                     "github actions", "jenkins", "gitlab ci",
                                     "circle ci", "travis ci"],
    "infrastructure as code":       ["iac", "terraform", "pulumi", "cloudformation"],

    # Concepts
    "object oriented programming":  ["oop", "object oriented", "oo design"],
    "functional programming":       ["fp"],
    "agile":                        ["scrum", "kanban", "sprint planning"],
    "system design":                ["distributed systems", "high availability",
                                     "scalable architecture"],
    "version control":              ["git", "github", "gitlab", "bitbucket"],
}

# Reverse map:  alias -> canonical
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in SKILL_ALIASES.items():
    _ALIAS_TO_CANONICAL[_canonical] = _canonical
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias.lower()] = _canonical


def normalise_skill(raw: str) -> str:
    """Return the canonical form of a skill string."""
    return _ALIAS_TO_CANONICAL.get(raw.strip().lower(), raw.strip().lower())


# ══════════════════════════════════════════════════════════════════════════════
# 2.  REQUIRED vs OPTIONAL SIGNAL DETECTION
# ══════════════════════════════════════════════════════════════════════════════
_REQUIRED_RE = re.compile(
    r"\b(required|must\s+have|must[\s-]know|mandatory|essential|"
    r"you\s+must|we\s+require|minimum\s+requirement|"
    r"years?\s+of\s+experience|qualification[s]?)\b",
    re.IGNORECASE,
)
_OPTIONAL_RE = re.compile(
    r"\b(preferred|nice[\s-]to[\s-]have|bonus|plus|desirable|"
    r"good[\s-]to[\s-]have|ideally|advantage|optionally|"
    r"not\s+required|would\s+be\s+a\s+plus)\b",
    re.IGNORECASE,
)


def classify_jd_skill_importance(skill_term: str, job_description: str) -> str:
    """
    Scan every sentence/line that contains the skill term in the JD.
    Return 'optional' if optional signals dominate, else 'required'.
    """
    skill_lower = skill_term.lower()
    sentences = re.split(r"[.\n;]", job_description)
    opt_hits = req_hits = 0
    for sent in sentences:
        if skill_lower in sent.lower():
            if _OPTIONAL_RE.search(sent):
                opt_hits += 1
            if _REQUIRED_RE.search(sent):
                req_hits += 1
    if opt_hits > req_hits:
        return "optional"
    return "required"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SKILL PRESENCE CHECK  (keyword + semantic hybrid)
# ══════════════════════════════════════════════════════════════════════════════
_SEM_CACHE: dict[tuple, float] = {}   # (canonical, resume_hash) -> cosine


def _skill_present_score(
    skill_canonical: str,
    all_aliases: list[str],
    resume_text: str,
    resume_embedding: np.ndarray,
    embedder,
    sem_threshold: float = 0.42,
) -> tuple[str, float]:
    """
    Returns (status, confidence_0_1).

    Status:
      "strong"  -> keyword exact-match (canonical or alias) in resume
      "partial" -> no keyword but semantic similarity >= sem_threshold
      "missing" -> neither
    """
    resume_lower = resume_text.lower()
    terms = [skill_canonical] + [a.lower() for a in all_aliases]

    # a) Keyword pass
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, resume_lower):
            n = len(re.findall(pattern, resume_lower))
            confidence = min(0.75 + n * 0.05, 0.98)
            return "strong", round(confidence, 2)

    # b) Semantic pass
    cache_key = (skill_canonical, hash(resume_text[:600]))
    if cache_key not in _SEM_CACHE:
        skill_emb = embedder.encode([skill_canonical], convert_to_numpy=True)
        sim = float(cosine_similarity(resume_embedding, skill_emb)[0][0])
        _SEM_CACHE[cache_key] = sim
    sim = _SEM_CACHE[cache_key]

    if sim >= sem_threshold:
        conf = round(float(np.clip((sim - sem_threshold) / (1 - sem_threshold), 0.35, 0.70)), 2)
        return "partial", conf

    # c) Missing
    conf = round(float(np.clip(sim * 0.4, 0.0, 0.30)), 2)
    return "missing", conf


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MAIN SCORING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_match_score(resume_text: str, job_description: str) -> float:
    """
    ATS-realistic weighted score -> 0-100.

    Weights
    -------
    50 %  semantic similarity   (sentence-transformer cosine, re-calibrated)
    30 %  required skill match  (proportion covered, scaled 0-100)
    20 %  optional skill bonus  (pure additive, never penalises absence)
    """
    embedder = model_manager.embedder

    # ── 4a. Semantic component ────────────────────────────────────────────────
    resume_chunks = _smart_chunk(resume_text)
    jd_chunks     = _smart_chunk(job_description)

    resume_embs = embedder.encode(resume_chunks, convert_to_numpy=True)
    jd_embs     = embedder.encode(jd_chunks,     convert_to_numpy=True)

    # Best-match pooling per JD chunk -> average maximums
    sims        = cosine_similarity(jd_embs, resume_embs)   # (jd_n, res_n)
    raw_sim     = float(sims.max(axis=1).mean())

    # Re-calibrate: [0.25, 0.78] -> [0, 100]
    sem_score = float(np.clip((raw_sim - 0.25) / 0.53 * 100, 0, 100))

    # ── 4b. Skill components ─────────────────────────────────────────────────
    resume_embedding = np.mean(resume_embs, axis=0, keepdims=True)
    jd_skills = _extract_jd_skills_with_importance(job_description)

    required_sk = [(s, a) for s, a, i in jd_skills if i == "required"]
    optional_sk = [(s, a) for s, a, i in jd_skills if i == "optional"]

    def _score_list(skill_list):
        out = []
        for canonical, aliases in skill_list:
            status, conf = _skill_present_score(
                canonical, aliases, resume_text, resume_embedding, embedder
            )
            if status == "strong":
                out.append(conf)
            elif status == "partial":
                out.append(conf * 0.6)
            else:
                out.append(0.0)
        return out

    req_scores = _score_list(required_sk)
    opt_scores = _score_list(optional_sk)

    # Required: average match rate scaled to 0-100
    # Fallback 60 when no required skills detected (neutral, not punishing)
    req_component = (sum(req_scores) / len(req_scores) * 100) if req_scores else 60.0
    # Optional: average match rate — ONLY adds value
    opt_component = (sum(opt_scores) / len(opt_scores) * 100) if opt_scores else 0.0

    # ── 4c. Combine ──────────────────────────────────────────────────────────
    final = (
        0.50 * sem_score
        + 0.30 * req_component
        + 0.20 * opt_component
    )
    return round(float(np.clip(final, 0, 100)), 1)


def compute_detailed_score(resume_text: str, job_description: str) -> dict:
    """
    Full breakdown returned to the API so the frontend can show transparency.
    Returns everything compute_match_score computes, plus per-skill details.
    """
    embedder = model_manager.embedder

    resume_chunks = _smart_chunk(resume_text)
    jd_chunks     = _smart_chunk(job_description)
    resume_embs   = embedder.encode(resume_chunks, convert_to_numpy=True)
    jd_embs       = embedder.encode(jd_chunks,     convert_to_numpy=True)
    sims          = cosine_similarity(jd_embs, resume_embs)
    raw_sim       = float(sims.max(axis=1).mean())
    sem_score     = float(np.clip((raw_sim - 0.25) / 0.53 * 100, 0, 100))

    resume_embedding = np.mean(resume_embs, axis=0, keepdims=True)
    jd_skills = _extract_jd_skills_with_importance(job_description)

    skill_results = []
    req_scores, opt_scores = [], []

    for canonical, aliases, importance in jd_skills:
        status, conf = _skill_present_score(
            canonical, aliases, resume_text, resume_embedding, embedder
        )
        weighted = conf if status == "strong" else (conf * 0.6 if status == "partial" else 0.0)
        (req_scores if importance == "required" else opt_scores).append(weighted)

        skill_results.append({
            "skill":      canonical,
            "aliases":    aliases[:3],
            "importance": importance,
            "status":     status,
            "confidence": conf,
        })

    req_component = (sum(req_scores) / len(req_scores) * 100) if req_scores else 60.0
    opt_component = (sum(opt_scores) / len(opt_scores) * 100) if opt_scores else 0.0
    final = float(np.clip(
        0.50 * sem_score + 0.30 * req_component + 0.20 * opt_component,
        0, 100,
    ))

    return {
        "final_score":     round(final, 1),
        "semantic_score":  round(sem_score, 1),
        "required_score":  round(req_component, 1),
        "optional_score":  round(opt_component, 1),
        "skill_breakdown": skill_results,
        "weights":         {"semantic": 0.50, "required": 0.30, "optional": 0.20},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SKILL EXTRACTION FROM JD
# ══════════════════════════════════════════════════════════════════════════════

def _extract_jd_skills_with_importance(
    job_description: str,
) -> list[tuple[str, list[str], str]]:
    """
    Returns list of (canonical, aliases, importance).
    Scans the JD text for every known canonical skill and its aliases.
    """
    jd_lower = job_description.lower()
    found: list[tuple[str, list[str], str]] = []
    seen: set[str] = set()

    for canonical, aliases in SKILL_ALIASES.items():
        if canonical in seen:
            continue
        all_terms = [canonical] + [a.lower() for a in aliases]
        for term in all_terms:
            pattern = r"\b" + re.escape(term) + r"\b"
            if re.search(pattern, jd_lower):
                importance = classify_jd_skill_importance(term, job_description)
                found.append((canonical, aliases, importance))
                seen.add(canonical)
                break

    if not found:
        # Generic fallback — neutral required skills
        found = [
            ("python",           SKILL_ALIASES["python"],           "required"),
            ("machine learning", SKILL_ALIASES["machine learning"], "required"),
            ("sql",              SKILL_ALIASES["sql"],              "required"),
        ]

    return found


# ══════════════════════════════════════════════════════════════════════════════
# 6.  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _smart_chunk(text: str, max_words: int = 180) -> list[str]:
    """Sentence-aware chunking with 50 % overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, word_count = [], [], 0

    for sent in sentences:
        wc = len(sent.split())
        if word_count + wc > max_words and current:
            chunks.append(" ".join(current))
            half = len(current) // 2
            current = current[half:]
            word_count = sum(len(s.split()) for s in current)
        current.append(sent)
        word_count += wc

    if current:
        chunks.append(" ".join(current))
    return chunks if chunks else [text]


def get_match_label(score: float) -> str:
    if score >= 78:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 42:
        return "Fair"
    else:
        return "Poor"


def extract_job_title(job_description: str) -> str:
    lines = job_description.strip().split("\n")
    for line in lines[:6]:
        line = line.strip()
        if 5 < len(line) < 80 and not line.endswith(":") and not line.startswith("•"):
            if not re.search(r"(we are|our company|about us|join us)", line, re.I):
                return line
    for line in lines:
        m = re.search(r"(?:position|role|title|job)[:\s]+(.+)", line, re.IGNORECASE)
        if m:
            return m.group(1).strip()[:80]
    return "Target Role"