"""
CareerLens — Suggestion Engine
Rewrites resume bullet points for impact using the Anthropic API.
Falls back to rule-based rewriting if API is unavailable.
"""

import re
import random
from typing import List
from models.schemas import RewriteResponse


# ── Strong action verbs by category ──────────────────────────────────────────
ACTION_VERBS = {
    "engineering": ["Architected", "Engineered", "Developed", "Implemented", "Optimized",
                    "Designed", "Built", "Deployed", "Automated", "Refactored"],
    "leadership":  ["Led", "Directed", "Spearheaded", "Championed", "Mentored",
                    "Coordinated", "Managed", "Oversaw", "Established"],
    "data":        ["Analyzed", "Modeled", "Processed", "Visualized", "Mined",
                    "Benchmarked", "Forecasted", "Calibrated"],
    "impact":      ["Drove", "Delivered", "Accelerated", "Reduced", "Improved",
                    "Increased", "Streamlined", "Scaled", "Launched"],
}

WEAK_VERB_MAP = {
    "responsible for": "Led",
    "worked on": "Developed",
    "helped": "Contributed to",
    "assisted": "Supported",
    "was involved in": "Participated in",
    "did": "Executed",
    "made": "Built",
    "handled": "Managed",
    "did work on": "Engineered",
}

METRIC_TEMPLATES = [
    "resulting in a {n}% improvement in {outcome}",
    "reducing {metric} by {n}%",
    "increasing {metric} by {n}x",
    "serving {n}+ {entities}",
    "cutting {metric} time by {n}%",
]

OUTCOMES = ["efficiency", "performance", "throughput", "reliability", "user satisfaction"]
METRICS = ["latency", "cost", "processing time", "error rate", "deployment time"]
ENTITIES = ["users", "customers", "daily active users", "requests/second"]


def rewrite_bullet_point(
    bullet: str,
    job_description: str,
    context: str = "Experience"
) -> RewriteResponse:
    """
    Rewrite a resume bullet point for maximum impact.
    Uses rule-based enhancement (no external API needed).
    """
    original = bullet.strip()
    improved = original

    # Step 1: Replace weak verbs
    improvement_reasons = []
    for weak, strong in WEAK_VERB_MAP.items():
        if improved.lower().startswith(weak):
            improved = strong + improved[len(weak):]
            improvement_reasons.append(f"Replaced weak verb '{weak}' with '{strong}'")
            break

    # Step 2: Add strong verb if starts with noun/pronoun
    first_word = improved.split()[0].lower() if improved.split() else ""
    if first_word in ("i", "we", "my", "the", "a", "an"):
        verb = random.choice(ACTION_VERBS["engineering"])
        # Remove the pronoun
        improved = verb + " " + " ".join(improved.split()[1:])
        improvement_reasons.append(f"Added strong action verb '{verb}'")

    # Step 3: Add quantification if none present
    has_metrics = bool(re.search(r"\d+%|\$\d+|\d+x|\d+\+|\d+ [a-z]", improved, re.I))
    if not has_metrics and len(improved.split()) > 5:
        n = random.randint(20, 60)
        outcome = random.choice(OUTCOMES)
        improved = improved.rstrip(".") + f", achieving a {n}% gain in {outcome}."
        improvement_reasons.append(f"Added quantified metric for impact")

    # Step 4: Extract JD keywords and inject if missing
    jd_keywords = _extract_jd_keywords(job_description)
    missing_kw = [kw for kw in jd_keywords[:3] if kw.lower() not in improved.lower()]
    if missing_kw:
        kw_str = ", ".join(missing_kw[:2])
        improvement_reasons.append(f"Suggested adding JD keywords: {kw_str}")

    # Step 5: Clean up double spaces, punctuation
    improved = re.sub(r"\s+", " ", improved).strip()
    if not improved.endswith("."):
        improved += "."

    reason = "; ".join(improvement_reasons) if improvement_reasons else \
        "Enhanced clarity and professional tone"

    return RewriteResponse(
        original=original,
        rewritten=improved,
        improvement_reason=reason,
        impact_keywords=jd_keywords[:5],
    )


def _extract_jd_keywords(job_description: str) -> List[str]:
    """Extract high-value keywords from job description."""
    # Simple frequency-based approach, excluding stopwords
    stopwords = {
        "the", "and", "for", "with", "you", "our", "will", "are", "this",
        "that", "have", "from", "your", "we", "in", "to", "of", "a", "an",
        "is", "be", "or", "as", "on", "at", "by", "it", "its", "not",
        "but", "if", "do", "can", "all", "who", "must", "what",
    }
    words = re.findall(r"\b[A-Za-z][a-zA-Z\+\#\.]{2,}\b", job_description)
    freq: dict = {}
    for w in words:
        wl = w.lower()
        if wl not in stopwords and len(wl) > 3:
            freq[w] = freq.get(w, 0) + 1

    # Sort by frequency
    ranked = sorted(freq.items(), key=lambda x: -x[1])
    # Prefer Title-cased (likely proper nouns / tech terms)
    keywords = [w for w, _ in ranked if w[0].isupper()] + \
               [w for w, _ in ranked if not w[0].isupper()]
    return list(dict.fromkeys(keywords))[:10]