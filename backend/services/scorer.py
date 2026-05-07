"""
CareerLens — ATS-Realistic Scoring Engine  (v3)
================================================

Scoring philosophy
------------------
Mimics how a smart technical recruiter reads a resume — not a grep tool.

Formula (dynamically weighted by seniority):
─────────────────────────────────────────────
INTERN / STUDENT
  40 % semantic similarity
  25 % required skill match   (keyword + implied)
  20 % project / learning signal
  15 % optional skill bonus   (pure additive)

EXPERIENCED (3 + yrs)
  40 % semantic similarity
  35 % required skill match
  15 % production / deployment signal
  10 % optional skill bonus

Key improvements over v2
─────────────────────────
1. IMPLIED SKILL INFERENCE  — "Flask app" → infers REST APIs; "NLP project" → infers
   text preprocessing, tokenisation, embeddings; "ML project" → infers NumPy/Pandas.
2. SENIORITY DETECTION      — detects intern / junior / senior from JD text and
   calibrates weights + thresholds accordingly.
3. EXPERIENCE SIGNALS       — hackathons, GitHub, internships, CGPA, certifications
   all contribute to a "learning potential" bonus used in intern scoring.
4. CALIBRATED RANGES        — intern strong-fit → 70-85, moderate → 55-70, no longer
   capped at 62 for a clearly qualified candidate.
5. PROJECTED SCORE FIX      — virtual skill injection re-runs the skill component so
   the projected score actually moves.
6. EXPLAINABILITY           — every score includes strengths[], gaps[], reasoning str.
"""

from __future__ import annotations

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .model_manager import model_manager


# ══════════════════════════════════════════════════════════════════════════════
#  1.  SKILL ALIAS MAP  +  IMPLIED SKILL GRAPH
# ══════════════════════════════════════════════════════════════════════════════

SKILL_ALIASES: dict[str, list[str]] = {
    "python":                       ["python3", "py"],
    "javascript":                   ["js", "es6", "ecmascript"],
    "typescript":                   ["ts"],
    "java":                         ["java8", "java11", "java17"],
    "c++":                          ["cpp"],
    "c#":                           ["csharp", ".net"],
    "golang":                       ["go lang", "go programming"],
    "r language":                   ["r programming"],

    "backend api development":      [
        "fastapi", "flask", "django", "express", "spring boot",
        "rest api", "restful", "graphql", "api development",
        "web api", "node.js", "nodejs", "backend development",
    ],
    "microservices":                ["micro-services", "soa"],

    "machine learning":             ["ml", "statistical learning", "predictive modelling",
                                     "supervised learning", "unsupervised learning"],
    "deep learning":                ["dl", "neural networks", "ann", "dnn", "cnn", "rnn", "lstm"],
    "natural language processing":  ["nlp", "text processing", "text classification",
                                     "sentiment analysis", "named entity recognition",
                                     "language model", "computational linguistics"],
    "computer vision":              ["cv", "image recognition", "object detection",
                                     "image classification", "opencv"],
    "mlops":                        ["ml ops", "ml pipeline", "model deployment",
                                     "model serving", "model monitoring"],
    "large language models":        ["llm", "generative ai", "gen ai", "gpt",
                                     "prompt engineering", "langchain"],
    "reinforcement learning":       ["rl", "rlhf"],
    "time series analysis":         ["time series", "forecasting", "arima", "lstm forecasting"],

    "pytorch":                      ["torch"],
    "tensorflow":                   ["tf", "keras"],
    "scikit-learn":                 ["sklearn", "scikit learn"],
    "hugging face":                 ["huggingface", "transformers library"],
    "numpy":                        ["np", "numerical python"],
    "pandas":                       ["dataframes", "pd"],
    "data analysis":                ["data analytics", "eda", "exploratory data analysis",
                                     "data visualization", "matplotlib", "seaborn"],
    "data engineering":             ["data pipelines", "etl", "elt"],
    "apache spark":                 ["spark", "pyspark"],

    "sql":                          ["mysql", "postgresql", "postgres", "sqlite",
                                     "database", "relational database", "t-sql"],
    "nosql":                        ["mongodb", "cassandra", "dynamodb"],

    "aws":                          ["amazon web services", "s3", "ec2", "lambda",
                                     "sagemaker", "boto3"],
    "gcp":                          ["google cloud", "bigquery", "vertex ai"],
    "azure":                        ["microsoft azure", "azure ml"],
    "docker":                       ["containerisation", "containerization", "containers"],
    "kubernetes":                   ["k8s", "container orchestration"],
    "ci/cd":                        ["continuous integration", "github actions",
                                     "jenkins", "gitlab ci"],

    "git":                          ["github", "gitlab", "bitbucket", "version control"],
    "object oriented programming":  ["oop", "object oriented", "oops"],
    "data structures":              ["dsa", "algorithms", "data structures and algorithms"],
    "agile":                        ["scrum", "kanban"],
    "system design":                ["distributed systems", "scalable architecture"],
    "problem solving":              ["competitive programming", "leetcode", "hackerrank",
                                     "debugging"],
}

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _c, _al in SKILL_ALIASES.items():
    _ALIAS_TO_CANONICAL[_c] = _c
    for _a in _al:
        _ALIAS_TO_CANONICAL[_a.lower()] = _c


def normalise_skill(raw: str) -> str:
    return _ALIAS_TO_CANONICAL.get(raw.strip().lower(), raw.strip().lower())


# ── Implied skill graph ───────────────────────────────────────────────────────
# "if resume contains KEY, infer these skills at given confidence"
# confidence: 0.0–1.0 (applied as fractional match score)
IMPLIED_SKILLS: dict[str, list[tuple[str, float]]] = {
    # ML ecosystem
    "machine learning":            [("numpy", 0.70), ("pandas", 0.70),
                                    ("scikit-learn", 0.65), ("data analysis", 0.60)],
    "deep learning":               [("numpy", 0.75), ("pytorch", 0.50),
                                    ("tensorflow", 0.50)],
    "natural language processing": [("numpy", 0.70), ("pandas", 0.65),
                                    ("scikit-learn", 0.60), ("data analysis", 0.55)],
    "computer vision":             [("numpy", 0.75), ("pytorch", 0.45)],
    "time series analysis":        [("numpy", 0.70), ("pandas", 0.75),
                                    ("data analysis", 0.65)],
    "data analysis":               [("numpy", 0.70), ("pandas", 0.75)],

    # Backend
    "backend api development":     [("backend api development", 0.85),
                                    ("sql", 0.45)],
    "flask":                       [("backend api development", 0.90)],
    "fastapi":                     [("backend api development", 0.90)],

    # LLM / Prompt
    "large language models":       [("natural language processing", 0.55),
                                    ("python", 0.80)],

    # Deployment signals
    "docker":                      [("mlops", 0.40)],
    "aws":                         [("mlops", 0.45)],
    "git":                         [("version control", 0.95)],
}

# Context phrases that signal deployment / project maturity
_DEPLOY_SIGNALS = [
    "deployed", "github pages", "render", "heroku", "aws", "gcp", "azure",
    "docker", "production", "live project", "api endpoint", "hosted",
]
_HACKATHON_SIGNALS = [
    "hackathon", "sih", "smart india", "national level", "winner", "finalist",
    "competed", "competition", "ieee", "acm",
]
_INTERNSHIP_SIGNALS = [
    "intern", "internship", "training", "isro", "industrial",
]
_GITHUB_SIGNALS = ["github.com/", "github:", "open source", "open-source"]
_LEADERSHIP_SIGNALS = [
    "led", "managed", "spearheaded", "founded", "organised", "organized",
    "team lead", "president", "captain",
]


# ══════════════════════════════════════════════════════════════════════════════
#  2.  SENIORITY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_seniority(job_description: str) -> str:
    """Returns 'intern', 'junior', or 'senior'."""
    jd_lower = job_description.lower()
    intern_kw  = ["intern", "internship", "fresher", "fresh graduate",
                  "entry level", "entry-level", "graduate", "trainee", "stipend"]
    senior_kw  = ["senior", "lead", "principal", "staff", "architect",
                  "manager", "director", "5+ years", "7+ years", "10+ years"]
    junior_kw  = ["junior", "associate", "1-2 years", "1-3 years", "2+ years"]

    if any(k in jd_lower for k in intern_kw):
        return "intern"
    if any(k in jd_lower for k in senior_kw):
        return "senior"
    if any(k in jd_lower for k in junior_kw):
        return "junior"

    # Heuristic: if no years required mentioned, lean junior
    if not re.search(r"\d+\+?\s*years?\s*(of\s*)?(experience|exp)", jd_lower):
        return "junior"
    return "senior"


# ══════════════════════════════════════════════════════════════════════════════
#  3.  REQUIRED vs OPTIONAL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

_REQUIRED_RE = re.compile(
    r"\b(required|must\s+have|must[\s-]know|mandatory|essential|"
    r"you\s+must|we\s+require|minimum|qualification[s]?|"
    r"strong\s+knowledge|strong\s+understanding)\b",
    re.IGNORECASE,
)
_OPTIONAL_RE = re.compile(
    r"\b(preferred|nice[\s-]to[\s-]have|bonus|plus|desirable|"
    r"good[\s-]to[\s-]have|ideally|advantage|optionally|"
    r"not\s+required|would\s+be\s+a\s+plus|familiarity|exposure)\b",
    re.IGNORECASE,
)


def classify_jd_skill_importance(skill_term: str, job_description: str) -> str:
    skill_lower = skill_term.lower()
    sentences = re.split(r"[.\n;]", job_description)
    opt_hits = req_hits = 0
    for sent in sentences:
        if skill_lower in sent.lower():
            if _OPTIONAL_RE.search(sent):
                opt_hits += 1
            if _REQUIRED_RE.search(sent):
                req_hits += 1
    return "optional" if opt_hits > req_hits else "required"


# ══════════════════════════════════════════════════════════════════════════════
#  4.  SKILL PRESENCE CHECK  (keyword + implied + semantic)
# ══════════════════════════════════════════════════════════════════════════════

_SEM_CACHE: dict[tuple, float] = {}


def _skill_present_score(
    skill_canonical: str,
    all_aliases: list[str],
    resume_text: str,
    resume_embedding: np.ndarray,
    embedder,
    implied_scores: dict[str, float],
    sem_threshold: float = 0.40,
) -> tuple[str, float]:
    """
    Returns (status, confidence).
    Status: 'strong' | 'implied' | 'partial' | 'missing'
    """
    resume_lower = resume_text.lower()
    terms = [skill_canonical] + [a.lower() for a in all_aliases]

    # a) Keyword exact match
    for term in terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, resume_lower):
            n = len(re.findall(pattern, resume_lower))
            return "strong", min(0.78 + n * 0.04, 0.98)

    # b) Implied skill check
    if skill_canonical in implied_scores:
        conf = implied_scores[skill_canonical]
        if conf >= 0.50:
            return "implied", round(conf, 2)

    # c) Semantic similarity
    cache_key = (skill_canonical, hash(resume_text[:600]))
    if cache_key not in _SEM_CACHE:
        skill_emb = embedder.encode([skill_canonical], convert_to_numpy=True)
        sim = float(cosine_similarity(resume_embedding, skill_emb)[0][0])
        _SEM_CACHE[cache_key] = sim
    sim = _SEM_CACHE[cache_key]

    if sim >= sem_threshold:
        conf = round(float(np.clip((sim - sem_threshold) / (1 - sem_threshold), 0.30, 0.65)), 2)
        return "partial", conf

    # d) Low implied (below threshold)
    if skill_canonical in implied_scores:
        conf = implied_scores[skill_canonical]
        return "partial", round(conf * 0.5, 2)

    return "missing", round(float(np.clip(sim * 0.3, 0.0, 0.25)), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  5.  IMPLIED SKILL COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_implied_skills(resume_text: str) -> dict[str, float]:
    """
    Scan resume for trigger skills. For each trigger found, grant implied
    confidence to its dependent skills.
    Returns {canonical_skill: max_confidence_across_all_triggers}
    """
    resume_lower = resume_text.lower()
    implied: dict[str, float] = {}

    for trigger_canonical, dependents in IMPLIED_SKILLS.items():
        trigger_aliases = SKILL_ALIASES.get(trigger_canonical, [])
        terms = [trigger_canonical] + [a.lower() for a in trigger_aliases]

        trigger_found = any(
            re.search(r"\b" + re.escape(t) + r"\b", resume_lower)
            for t in terms
        )
        if trigger_found:
            for dep_canonical, conf in dependents:
                # Take maximum confidence if multiple triggers grant the same skill
                implied[dep_canonical] = max(implied.get(dep_canonical, 0.0), conf)

    return implied


# ══════════════════════════════════════════════════════════════════════════════
#  6.  EXPERIENCE SIGNAL SCORER
# ══════════════════════════════════════════════════════════════════════════════

def compute_experience_signals(resume_text: str, seniority: str) -> dict:
    """
    For intern/junior roles: assess learning potential signals.
    Returns a dict with signal scores and a composite bonus (0-100).
    """
    rl = resume_text.lower()

    has_github     = any(s in rl for s in _GITHUB_SIGNALS)
    has_hackathon  = any(s in rl for s in _HACKATHON_SIGNALS)
    has_internship = any(s in rl for s in _INTERNSHIP_SIGNALS)
    has_deploy     = any(s in rl for s in _DEPLOY_SIGNALS)
    has_leadership = any(s in rl for s in _LEADERSHIP_SIGNALS)

    # Count projects
    project_count = len(re.findall(
        r"\b(project|built|developed|created|implemented|designed)\b", rl
    ))
    has_projects = project_count >= 2

    # CGPA detection
    cgpa_match = re.search(r"cgpa[:\s]+(\d+\.?\d*)", rl)
    high_cgpa  = cgpa_match and float(cgpa_match.group(1)) >= 8.5

    # Certifications
    cert_count = len(re.findall(
        r"\b(certificate|certification|certified|coursera|udemy|"
        r"harvard|google|aws certified|microsoft certified)\b", rl
    ))
    has_certs = cert_count >= 1

    signals = {
        "github":     has_github,
        "hackathon":  has_hackathon,
        "internship": has_internship,
        "deployment": has_deploy,
        "leadership": has_leadership,
        "projects":   has_projects,
        "high_cgpa":  high_cgpa,
        "certs":      has_certs,
    }

    if seniority == "intern":
        # Each signal contributes to a 0-100 bonus score
        weights = {
            "github": 15, "hackathon": 18, "internship": 20,
            "deployment": 12, "leadership": 8, "projects": 15,
            "high_cgpa": 7, "certs": 10,
        }
    elif seniority == "junior":
        weights = {
            "github": 12, "hackathon": 10, "internship": 18,
            "deployment": 18, "leadership": 10, "projects": 12,
            "high_cgpa": 5, "certs": 8,
        }
    else:  # senior
        weights = {
            "github": 8, "hackathon": 5, "internship": 5,
            "deployment": 25, "leadership": 22, "projects": 8,
            "high_cgpa": 2, "certs": 8,
        }

    bonus = sum(w for k, w in weights.items() if signals[k])
    bonus = min(bonus, 100)

    return {"signals": signals, "bonus": bonus, "seniority": seniority}


# ══════════════════════════════════════════════════════════════════════════════
#  7.  JD SKILL EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_jd_skills_with_importance(
    job_description: str,
) -> list[tuple[str, list[str], str]]:
    jd_lower = job_description.lower()
    found: list[tuple[str, list[str], str]] = []
    seen: set[str] = set()

    for canonical, aliases in SKILL_ALIASES.items():
        if canonical in seen:
            continue
        terms = [canonical] + [a.lower() for a in aliases]
        for term in terms:
            if re.search(r"\b" + re.escape(term) + r"\b", jd_lower):
                importance = classify_jd_skill_importance(term, job_description)
                found.append((canonical, aliases, importance))
                seen.add(canonical)
                break

    if not found:
        found = [
            ("python",           SKILL_ALIASES["python"],           "required"),
            ("machine learning", SKILL_ALIASES["machine learning"], "required"),
            ("git",              SKILL_ALIASES["git"],              "required"),
        ]
    return found


# ══════════════════════════════════════════════════════════════════════════════
#  8.  SMART CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def _smart_chunk(text: str, max_words: int = 180) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, wc = [], [], 0
    for sent in sentences:
        sw = len(sent.split())
        if wc + sw > max_words and current:
            chunks.append(" ".join(current))
            half = len(current) // 2
            current = current[half:]
            wc = sum(len(s.split()) for s in current)
        current.append(sent)
        wc += sw
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


# ══════════════════════════════════════════════════════════════════════════════
#  9.  MAIN SCORING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_match_score(resume_text: str, job_description: str) -> float:
    return compute_detailed_score(resume_text, job_description)["final_score"]


def compute_detailed_score(resume_text: str, job_description: str) -> dict:
    """
    Full ATS-realistic score with per-component breakdown and explainability.
    """
    embedder  = model_manager.embedder
    seniority = detect_seniority(job_description)

    # ── Weights by seniority ──────────────────────────────────────────────────
    W = {
        "intern":  {"sem": 0.40, "req": 0.25, "exp": 0.20, "opt": 0.15},
        "junior":  {"sem": 0.42, "req": 0.30, "exp": 0.15, "opt": 0.13},
        "senior":  {"sem": 0.40, "req": 0.35, "exp": 0.15, "opt": 0.10},
    }[seniority]

    # ── Semantic similarity ───────────────────────────────────────────────────
    resume_chunks = _smart_chunk(resume_text)
    jd_chunks     = _smart_chunk(job_description)
    resume_embs   = embedder.encode(resume_chunks, convert_to_numpy=True)
    jd_embs       = embedder.encode(jd_chunks,     convert_to_numpy=True)

    sims    = cosine_similarity(jd_embs, resume_embs)
    raw_sim = float(sims.max(axis=1).mean())

    # Calibration: intern JD ↔ student resume sits ~0.28–0.72 → map to 0–100
    sem_score = float(np.clip((raw_sim - 0.22) / 0.55 * 100, 0, 100))

    # ── Implied skills ────────────────────────────────────────────────────────
    resume_embedding = np.mean(resume_embs, axis=0, keepdims=True)
    implied_scores   = compute_implied_skills(resume_text)

    # ── JD skill extraction ───────────────────────────────────────────────────
    jd_skills    = _extract_jd_skills_with_importance(job_description)
    required_sk  = [(c, a) for c, a, i in jd_skills if i == "required"]
    optional_sk  = [(c, a) for c, a, i in jd_skills if i == "optional"]

    # ── Skill scoring ─────────────────────────────────────────────────────────
    skill_results = []
    req_scores, opt_scores = [], []
    strengths, gaps = [], []

    for canonical, aliases, importance in jd_skills:
        status, conf = _skill_present_score(
            canonical, aliases, resume_text, resume_embedding,
            embedder, implied_scores,
        )
        # Weight by status
        if status == "strong":
            weighted = conf
            strengths.append(canonical)
        elif status == "implied":
            weighted = conf * 0.75     # implied counts, but not as much as explicit
            strengths.append(f"{canonical} (implied)")
        elif status == "partial":
            weighted = conf * 0.55
        else:
            weighted = 0.0
            if importance == "required":
                gaps.append(canonical)

        (req_scores if importance == "required" else opt_scores).append(weighted)

        skill_results.append({
            "skill":      canonical,
            "aliases":    aliases[:3],
            "importance": importance,
            "status":     status,
            "confidence": conf,
        })

    req_component = (sum(req_scores) / len(req_scores) * 100) if req_scores else 65.0
    opt_component = (sum(opt_scores) / len(opt_scores) * 100) if opt_scores else 0.0

    # ── Experience / learning signal ─────────────────────────────────────────
    exp_info   = compute_experience_signals(resume_text, seniority)
    exp_score  = exp_info["bonus"]    # 0–100

    # ── Weighted final ────────────────────────────────────────────────────────
    final = (
        W["sem"] * sem_score
        + W["req"] * req_component
        + W["exp"] * exp_score
        + W["opt"] * opt_component
    )
    final = float(np.clip(final, 0, 100))

    # ── Explainability ────────────────────────────────────────────────────────
    reasoning = _build_reasoning(
        seniority, sem_score, req_component, exp_score,
        exp_info["signals"], strengths, gaps, final,
    )

    return {
        "final_score":     round(final, 1),
        "semantic_score":  round(sem_score, 1),
        "required_score":  round(req_component, 1),
        "optional_score":  round(opt_component, 1),
        "exp_score":       round(exp_score, 1),
        "seniority":       seniority,
        "weights":         W,
        "skill_breakdown": skill_results,
        "strengths":       strengths[:8],
        "gaps":            gaps[:6],
        "reasoning":       reasoning,
        "signals":         exp_info["signals"],
    }


def _build_reasoning(seniority, sem, req, exp, signals, strengths, gaps, final) -> str:
    parts = []

    if final >= 78:
        parts.append("Strong overall alignment with the job requirements.")
    elif final >= 62:
        parts.append("Good fit with some targeted gaps to address.")
    elif final >= 45:
        parts.append("Moderate fit — closing key skill gaps would significantly improve candidacy.")
    else:
        parts.append("Significant gaps vs. the job requirements.")

    if strengths:
        parts.append(f"Strong alignment in: {', '.join(strengths[:4])}.")
    if gaps:
        parts.append(f"Missing required: {', '.join(gaps[:3])}.")
    if seniority == "intern":
        if signals.get("hackathon"):
            parts.append("Hackathon participation signals initiative and real-world problem-solving.")
        if signals.get("internship"):
            parts.append("Prior internship experience is a strong differentiator for an intern role.")
        if signals.get("github"):
            parts.append("Active GitHub portfolio demonstrates project commitment.")
        if signals.get("high_cgpa"):
            parts.append("High academic performance (CGPA ≥ 8.5) signals strong fundamentals.")
    if not signals.get("deployment"):
        parts.append("Adding deployed projects would strengthen the application further.")

    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  10.  PROJECTED SCORE  (actually re-runs skill component)
# ══════════════════════════════════════════════════════════════════════════════

def compute_projected_score(
    resume_text: str,
    job_description: str,
    skills_to_add: list[str],
) -> float:
    """
    Virtually inject skills_to_add into the resume text and re-score.
    This gives a realistic projected score rather than a static +N estimate.
    """
    injected = resume_text + "\n\nAdditional Skills: " + ", ".join(skills_to_add)
    # Clear cache entries for this resume so new text is re-evaluated
    keys_to_clear = [k for k in _SEM_CACHE if k[1] == hash(resume_text[:600])]
    for k in keys_to_clear:
        del _SEM_CACHE[k]
    return compute_match_score(injected, job_description)


# ══════════════════════════════════════════════════════════════════════════════
#  11.  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_match_label(score: float) -> str:
    if score >= 78:
        return "Excellent"
    elif score >= 62:
        return "Good"
    elif score >= 45:
        return "Fair"
    else:
        return "Poor"


def get_potential_label(exp_score: float, seniority: str) -> str:
    """Secondary label used for intern/junior candidates."""
    if seniority not in ("intern", "junior"):
        return ""
    if exp_score >= 60:
        return "High Potential"
    elif exp_score >= 35:
        return "Moderate Potential"
    return "Developing"


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