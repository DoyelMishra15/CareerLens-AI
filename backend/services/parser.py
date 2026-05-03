"""
CareerLens — PDF Parser Service
Uses PyMuPDF for fast, accurate text extraction.
"""

import fitz  # PyMuPDF
import re
import io
from typing import Tuple


def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, int]:
    """
    Extract clean text from PDF bytes.
    Returns (text, page_count).
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text("text")
        pages.append(text)
    doc.close()

    full_text = "\n".join(pages)
    clean_text = _clean_text(full_text)
    return clean_text, len(pages)


def _clean_text(text: str) -> str:
    """Remove excess whitespace, fix encoding artifacts."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines to one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove null bytes and weird chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Fix hyphenation artifacts (word-\nbreak → wordbreak)
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text.strip()


def detect_resume_sections(text: str) -> dict:
    """
    Heuristically detect resume sections.
    Returns a dict of {section_name: content}.
    """
    section_patterns = {
        "summary":     r"(summary|objective|profile|about)",
        "experience":  r"(experience|work history|employment|career)",
        "education":   r"(education|academic|qualification|degree)",
        "skills":      r"(skills|technical skills|competencies|expertise)",
        "projects":    r"(projects|portfolio|work samples)",
        "certifications": r"(certification|certificate|license|credential)",
        "achievements": r"(achievement|award|honor|recognition)",
    }

    lines = text.split("\n")
    sections = {}
    current_section = "header"
    current_lines = []

    for line in lines:
        stripped = line.strip().lower()
        matched = False
        for sec_name, pattern in section_patterns.items():
            if re.search(pattern, stripped) and len(stripped) < 40:
                if current_lines:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = sec_name
                current_lines = []
                matched = True
                break
        if not matched:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def extract_bullet_points(text: str) -> list:
    """Extract bullet point lines from resume text."""
    lines = text.split("\n")
    bullets = []
    bullet_markers = ("•", "–", "-", "▪", "◦", "○", "*", "→")

    for line in lines:
        stripped = line.strip()
        if stripped and (
            stripped[0] in bullet_markers or
            (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)")
        ):
            clean = re.sub(r"^[•–\-▪◦○\*→\d\.)\s]+", "", stripped).strip()
            if len(clean) > 20:
                bullets.append(clean)

    return bullets