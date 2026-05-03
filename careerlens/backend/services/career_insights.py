"""
CareerLens — Career Growth Intelligence
The unique "Career Growth Intelligence" feature.
Prioritizes skills to learn based on impact score.
"""

from typing import List, Dict
from models.schemas import LearningPath, InsightsResponse


# ── Learning path database ────────────────────────────────────────────────────
LEARNING_PATHS: Dict[str, dict] = {
    "Python": {
        "resources": ["Python.org official tutorial", "Real Python (realpython.com)", "CS50P (edx.org)"],
        "time": "4–6 weeks",
        "impact": 15.0,
    },
    "Machine Learning": {
        "resources": ["fast.ai (free)", "Andrew Ng ML Course (Coursera)", "Hands-On ML (O'Reilly)"],
        "time": "8–12 weeks",
        "impact": 20.0,
    },
    "Deep Learning": {
        "resources": ["fast.ai Part 1 & 2", "deeplearning.ai Specialization", "PyTorch tutorials"],
        "time": "10–16 weeks",
        "impact": 18.0,
    },
    "Docker": {
        "resources": ["Docker official docs", "Play with Docker (free lab)", "TechWorld with Nana (YouTube)"],
        "time": "1–2 weeks",
        "impact": 10.0,
    },
    "Kubernetes": {
        "resources": ["Kubernetes.io tutorials", "KodeKloud (free tier)", "CKA prep guide"],
        "time": "4–6 weeks",
        "impact": 12.0,
    },
    "AWS": {
        "resources": ["AWS Free Tier + hands-on labs", "AWS Skill Builder (free)", "A Cloud Guru"],
        "time": "6–10 weeks",
        "impact": 18.0,
    },
    "SQL": {
        "resources": ["SQLZoo (free)", "Mode SQL Tutorial", "LeetCode SQL problems"],
        "time": "2–3 weeks",
        "impact": 10.0,
    },
    "React": {
        "resources": ["React official docs (beta)", "Scrimba React Course (free)", "Fullstack Open"],
        "time": "4–6 weeks",
        "impact": 12.0,
    },
    "TypeScript": {
        "resources": ["TypeScript handbook (official)", "Execute Program", "Total TypeScript (free tier)"],
        "time": "2–3 weeks",
        "impact": 8.0,
    },
    "Data Analysis": {
        "resources": ["Kaggle free courses", "Pandas documentation", "StatQuest (YouTube)"],
        "time": "3–5 weeks",
        "impact": 14.0,
    },
    "Communication": {
        "resources": ["Toastmasters (free/low-cost)", "Coursera Communication Skills", "Writing Without Bullshit (book)"],
        "time": "Ongoing",
        "impact": 8.0,
    },
    "Leadership": {
        "resources": ["The Manager's Path (book)", "Harvard ManageMentor", "LinkedIn Learning Leadership"],
        "time": "Ongoing",
        "impact": 10.0,
    },
    "Kubernetes": {
        "resources": ["Kubernetes.io tutorials", "KodeKloud free tier", "Kubernetes in Action (book)"],
        "time": "4–6 weeks",
        "impact": 12.0,
    },
    "FastAPI": {
        "resources": ["FastAPI official docs (best docs ever)", "TestDriven.io FastAPI course", "Tiangolo GitHub examples"],
        "time": "1–2 weeks",
        "impact": 7.0,
    },
    "NLP": {
        "resources": ["Hugging Face NLP Course (free)", "Stanford CS224N (YouTube)", "spaCy 101"],
        "time": "6–10 weeks",
        "impact": 16.0,
    },
    "TensorFlow": {
        "resources": ["TensorFlow.org tutorials", "DeepLearning.AI TF Certificate", "Kaggle TF notebooks"],
        "time": "6–8 weeks",
        "impact": 14.0,
    },
    "PyTorch": {
        "resources": ["PyTorch official tutorials", "fast.ai (uses PyTorch)", "Deep Learning with PyTorch (book, free PDF)"],
        "time": "6–8 weeks",
        "impact": 15.0,
    },
}

DEFAULT_LEARNING_PATH = {
    "resources": ["Google & YouTube tutorials", "freeCodeCamp", "Official documentation"],
    "time": "2–4 weeks",
    "impact": 8.0,
}

CAREER_TIPS = {
    "default": [
        "Contribute to open-source projects to demonstrate real-world skills",
        "Build a project portfolio on GitHub with READMEs and live demos",
        "Write technical blog posts — they signal deep expertise to recruiters",
        "Network on LinkedIn: comment thoughtfully on posts in your target domain",
        "Tailor your resume for each application — use JD keywords strategically",
    ],
    "engineer": [
        "Build and deploy at least one end-to-end project with cloud infrastructure",
        "Practice system design problems (Grokking the System Design Interview)",
        "Contribute to widely-used open-source libraries in your tech stack",
        "Earn one cloud certification (AWS/GCP/Azure) — it's a strong signal",
    ],
    "data": [
        "Publish Kaggle notebooks demonstrating your EDA and modeling approach",
        "Build a data pipeline project with Airflow/dbt as a portfolio piece",
        "Learn SQL deeply — window functions, CTEs, and query optimization",
        "Create a dashboard (Tableau Public / Streamlit) with real datasets",
    ],
    "ml": [
        "Reproduce a recent ML paper and share it on GitHub",
        "Fine-tune a Hugging Face model on a custom dataset",
        "Build an ML API with FastAPI and deploy it on Render/Railway (free)",
        "Track your experiments with MLflow or Weights & Biases (free tier)",
    ],
}


def generate_insights(
    missing_skills: List[str],
    resume_score: float,
    job_title: str = "",
) -> InsightsResponse:
    """
    Generate Career Growth Intelligence insights.
    """
    # 1. Build prioritized learning paths for missing skills
    learning_paths = []
    for i, skill in enumerate(missing_skills[:6]):
        path_data = LEARNING_PATHS.get(skill, DEFAULT_LEARNING_PATH)
        learning_paths.append(LearningPath(
            skill=skill,
            priority=i + 1,
            resources=path_data["resources"],
            estimated_time=path_data["time"],
            impact_score=path_data["impact"],
        ))

    # Sort by impact descending
    learning_paths.sort(key=lambda x: -x.impact_score)
    for i, lp in enumerate(learning_paths):
        lp.priority = i + 1

    # 2. Estimate score after improvement
    total_impact = sum(lp.impact_score for lp in learning_paths[:3])
    estimated_score = min(resume_score + total_impact * 0.6, 96.0)

    # 3. Career tips — detect domain from job title
    jt_lower = job_title.lower()
    if any(k in jt_lower for k in ["data", "analyst", "bi", "warehouse"]):
        tips = CAREER_TIPS["data"][:3] + CAREER_TIPS["default"][:2]
    elif any(k in jt_lower for k in ["ml", "machine learning", "ai", "nlp", "deep"]):
        tips = CAREER_TIPS["ml"][:3] + CAREER_TIPS["default"][:2]
    elif any(k in jt_lower for k in ["engineer", "developer", "architect", "devops", "backend", "frontend"]):
        tips = CAREER_TIPS["engineer"][:3] + CAREER_TIPS["default"][:2]
    else:
        tips = CAREER_TIPS["default"]

    # 4. Market insight
    market_insight = _generate_market_insight(job_title, missing_skills)

    return InsightsResponse(
        top_skills_to_learn=learning_paths[:3],
        estimated_score_after_improvement=round(estimated_score, 1),
        career_growth_tips=tips[:4],
        job_market_insight=market_insight,
    )


def _generate_market_insight(job_title: str, missing_skills: List[str]) -> str:
    jt = job_title.lower()
    skill_str = ", ".join(missing_skills[:3]) if missing_skills else "core skills"

    if "machine learning" in jt or "ml engineer" in jt:
        return (
            f"ML Engineer roles are growing 40% YoY. Mastering {skill_str} "
            f"positions you in the top 15% of candidates. LLM fine-tuning and MLOps "
            f"are the hottest sub-skills in 2024–2025."
        )
    elif "data" in jt:
        return (
            f"Data roles remain the #1 fastest-growing category. Adding {skill_str} "
            f"could increase your interview rate by ~2x. Cloud data platforms "
            f"(Snowflake, BigQuery) dominate current JDs."
        )
    elif "backend" in jt or "software engineer" in jt:
        return (
            f"Backend engineering demand is strong, especially in distributed systems. "
            f"Proficiency in {skill_str} is mentioned in 70%+ of senior JDs. "
            f"System design skills are the primary filter at FAANG-adjacent companies."
        )
    else:
        return (
            f"Roles like '{job_title}' are in high demand. Closing your gap in "
            f"{skill_str} could increase your match score by 15–25 points, "
            f"significantly improving your callback rate."
        )