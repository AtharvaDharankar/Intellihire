"""
extractor.py — ResumeIQ
Hybrid extraction pipeline:
  • Regex  → email, phone, date ranges, metrics
  • spaCy  → Named Entity Recognition (PERSON, ORG) for name & company
  • Keyword matching → skills from a curated ontology
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
from typing import Optional

import spacy

from models import (
    Achievement,
    Education,
    ExtractedResume,
    SeniorityLevel,
    WorkExperience,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & compiled patterns
# ─────────────────────────────────────────────────────────────────────────────

_EMAIL_RE   = re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b')
_PHONE_RE   = re.compile(r'(\+?\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}')
_YEAR_RANGE = re.compile(r'(\d{4})\s*[-–—]\s*(present|current|\d{4})', re.IGNORECASE)
_METRIC_RE  = re.compile(
    r'(\d+\s*%|\$[\d,]+|\d+\s*[xX]|\d[\d,]*\s*(million|billion|thousand|[kK]\b))',
    re.IGNORECASE,
)
_BULLET_RE  = re.compile(r'^[\s•\-\*\>\–]+')

SENIORITY_PATTERNS: dict[SeniorityLevel, str] = {
    SeniorityLevel.DIRECTOR:  r'\b(director|vp|vice president|head of|cto|cpo)\b',
    SeniorityLevel.PRINCIPAL: r'\b(principal|staff engineer|distinguished)\b',
    SeniorityLevel.LEAD:      r'\b(lead|tech lead|team lead|engineering lead)\b',
    SeniorityLevel.SENIOR:    r'\b(senior|sr\.?)\b',
    SeniorityLevel.MID:       r'\b(mid.?level|intermediate|software engineer)\b',
    SeniorityLevel.JUNIOR:    r'\b(junior|jr\.?|associate|entry.?level)\b',
    SeniorityLevel.INTERN:    r'\b(intern|trainee|apprentice)\b',
}

SECTION_HEADERS = {
    "experience":   re.compile(r'(?i)^(work\s+)?experience|employment|professional\s+history'),
    "education":    re.compile(r'(?i)^education|academic|qualifications?'),
    "skills":       re.compile(r'(?i)^(technical\s+)?skills?|technologies|competencies|tech\s+stack'),
    "achievements": re.compile(r'(?i)^achievements?|accomplishments?|awards?|projects?|highlights?'),
}

# Curated skill ontology (extend as needed)
SKILL_ONTOLOGY: frozenset[str] = frozenset({
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "scala", "r", "kotlin", "swift", "ruby", "php",
    # Web / API
    "react", "angular", "vue", "node.js", "fastapi", "django", "flask",
    "spring", "graphql", "rest api", "grpc",
    # ML / AI
    "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "lightgbm",
    "transformers", "bert", "gpt", "llm", "rag", "fine-tuning",
    "machine learning", "deep learning", "nlp", "computer vision", "mlops",
    "reinforcement learning", "hugging face",
    # Data
    "pandas", "numpy", "sql", "postgresql", "mysql", "mongodb", "redis",
    "elasticsearch", "kafka", "spark", "hadoop", "airflow", "dbt", "flink",
    # Cloud / DevOps
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ci/cd",
    "github actions", "jenkins", "helm", "prometheus", "grafana",
    # Vector / Search
    "faiss", "pinecone", "weaviate", "qdrant", "langchain",
    # Other
    "git", "linux", "agile", "scrum", "microservices",
})

_SKILL_PATTERNS: list[re.Pattern] = [
    re.compile(r'\b' + re.escape(s) + r'\b', re.IGNORECASE)
    for s in SKILL_ONTOLOGY
]

CURRENT_YEAR = datetime.datetime.now().year

# ─────────────────────────────────────────────────────────────────────────────
# spaCy loader (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _load_spacy() -> Optional[spacy.language.Language]:
    for model in ("en_core_web_sm", "en_core_web_md"):
        try:
            return spacy.load(model)
        except OSError:
            continue
    logger.warning(
        "spaCy model not found. Install via: python -m spacy download en_core_web_sm"
    )
    return None

_NLP = _load_spacy()


# ─────────────────────────────────────────────────────────────────────────────
# ResumeExtractor
# ─────────────────────────────────────────────────────────────────────────────

class ResumeExtractor:
    """
    Hybrid resume parser.

    Extraction strategy
    ───────────────────
    1. Regex     → email, phone, date ranges, quantified metrics
    2. spaCy NER → PERSON entity for candidate name, ORG for company names
    3. Keyword   → skill matching against SKILL_ONTOLOGY
    4. Heuristic → section splitting, seniority detection
    """

    # ── Regex extractors ──────────────────────────────────────────────────────

    @staticmethod
    def extract_email(text: str) -> Optional[str]:
        m = _EMAIL_RE.search(text)
        return m.group(0) if m else None

    @staticmethod
    def extract_phone(text: str) -> Optional[str]:
        m = _PHONE_RE.search(text)
        return m.group(0).strip() if m else None

    # ── NER / heuristic name extraction ───────────────────────────────────────

    @staticmethod
    def extract_name(text: str) -> str:
        """
        Try spaCy PERSON entity in the first 600 chars; fall back to
        the first non-empty, non-contact-info line.
        """
        if _NLP:
            doc = _NLP(text[:600])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text.strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Skip lines that look like contact info
        for line in lines[:5]:
            if not _EMAIL_RE.search(line) and not _PHONE_RE.search(line):
                return line
        return "Unknown Candidate"

    # ── Skill extraction ──────────────────────────────────────────────────────

    @staticmethod
    def extract_skills(text: str) -> list[str]:
        found: list[str] = []
        for skill, pattern in zip(SKILL_ONTOLOGY, _SKILL_PATTERNS):
            if pattern.search(text):
                found.append(skill)
        return sorted(set(found))

    # ── Experience / seniority ────────────────────────────────────────────────

    @staticmethod
    def calc_total_experience(text: str) -> float:
        """
        Sum durations from all YYYY–YYYY / YYYY–Present ranges.
        Caps at 40 years to avoid data artifacts.
        """
        total = 0.0
        for start_s, end_s in _YEAR_RANGE.findall(text):
            start = int(start_s)
            end   = CURRENT_YEAR if end_s.lower() in {"present", "current"} else int(end_s)
            total += max(0, end - start)
        return min(total, 40.0)

    @staticmethod
    def detect_seniority(text: str) -> SeniorityLevel:
        """
        Walk patterns from highest to lowest; first match wins.
        Defaults to MID if no pattern fires.
        """
        lower = text.lower()
        for level, pattern in SENIORITY_PATTERNS.items():
            if re.search(pattern, lower):
                return level
        return SeniorityLevel.MID

    # ── Section splitter ──────────────────────────────────────────────────────

    @staticmethod
    def split_sections(text: str) -> dict[str, str]:
        """
        Split resume into logical sections using header regex patterns.
        Returns a dict keyed by section name.
        """
        sections: dict[str, list[str]] = {k: [] for k in SECTION_HEADERS}
        sections["other"] = []
        current = "other"

        for line in text.splitlines():
            stripped = line.strip()
            matched  = False
            for sec, pattern in SECTION_HEADERS.items():
                if pattern.match(stripped) and len(stripped) < 60:
                    current = sec
                    matched = True
                    break
            if not matched:
                sections[current].append(line)

        return {k: "\n".join(v) for k, v in sections.items()}

    # ── Sub-entity parsers ────────────────────────────────────────────────────

    def parse_work_experience(self, text: str) -> list[WorkExperience]:
        """
        Split experience text on year-range anchors and parse each block.
        Uses spaCy ORG entity for company names when available.
        """
        if not text.strip():
            return []

        # Split blocks on lines that start with a year range
        blocks = re.split(r'\n(?=\d{4}\s*[-–—])', text)
        experiences: list[WorkExperience] = []

        for block in blocks:
            if len(block.strip()) < 20:
                continue

            # Duration
            m = _YEAR_RANGE.search(block)
            if m:
                start = int(m.group(1))
                end_s = m.group(2)
                end   = CURRENT_YEAR if end_s.lower() in {"present", "current"} else int(end_s)
                years = float(max(0, end - start))
            else:
                years = 1.0

            seniority = self.detect_seniority(block)
            skills_in_block = self.extract_skills(block)

            # Role / company from first two meaningful lines
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            role    = lines[0] if lines else "Unknown Role"
            company = "Unknown Company"
            if _NLP and len(lines) > 1:
                doc = _NLP(lines[1])
                orgs = [e.text for e in doc.ents if e.label_ == "ORG"]
                company = orgs[0] if orgs else lines[1]
            elif len(lines) > 1:
                company = lines[1]

            experiences.append(
                WorkExperience(
                    company=company,
                    role=role,
                    duration_years=years,
                    seniority=seniority,
                    description=block[:400],
                    technologies=skills_in_block,
                )
            )

        return experiences[:8]  # Cap to avoid noise

    @staticmethod
    def parse_education(text: str) -> list[Education]:
        """Detect degree keywords and extract adjacent institution/year."""
        if not text.strip():
            return []

        degree_re = re.compile(
            r'(Ph\.?D|Doctor|M\.?S\.?|M\.?Eng|Master|B\.?S\.?|B\.?E\.?|B\.?Tech|Bachelor|Associate|Diploma|Certificate)',
            re.IGNORECASE,
        )
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        educations: list[Education] = []

        for i, line in enumerate(lines):
            dm = degree_re.search(line)
            if not dm:
                continue
            year_m = re.search(r'\b(19|20)\d{2}\b', line)
            inst   = lines[i + 1] if i + 1 < len(lines) else "Unknown"
            educations.append(
                Education(
                    degree=dm.group(0),
                    field=line,
                    institution=inst,
                    year=int(year_m.group(0)) if year_m else None,
                )
            )

        return educations

    @staticmethod
    def parse_achievements(text: str) -> list[Achievement]:
        """Extract bullet-point achievements; flag ones with quantified metrics."""
        if not text.strip():
            return []

        achievements: list[Achievement] = []
        for line in text.splitlines():
            clean = _BULLET_RE.sub("", line).strip()
            if len(clean) < 20:
                continue
            m = _METRIC_RE.search(clean)
            achievements.append(
                Achievement(
                    description=clean[:250],
                    metric=m.group(0) if m else None,
                    impact="quantified" if m else "qualitative",
                )
            )
            if len(achievements) >= 10:
                break

        return achievements

    # ── Summary builder ───────────────────────────────────────────────────────

    @staticmethod
    def build_summary(resume: ExtractedResume) -> str:
        """
        Compact, information-dense string for the embedding model.
        Format: '<Name>. <N> yrs exp. Skills: <top 12>. Education: <highest>.'
        """
        skills_str = ", ".join(resume.skills[:12])
        exp_str    = f"{resume.total_years_experience:.1f} years experience"
        edu_str    = resume.education[0].field if resume.education else ""
        roles_str  = (
            "; ".join(e.role for e in resume.work_experience[:2])
            if resume.work_experience else ""
        )
        return (
            f"{resume.candidate_name}. {exp_str}. "
            f"Skills: {skills_str}. {edu_str}. {roles_str}."
        )

    # ── Main async entry point ────────────────────────────────────────────────

    async def extract(self, raw_text: str) -> ExtractedResume:
        """
        Full extraction pipeline — async to integrate with the batch processor.
        CPU-bound spaCy calls are offloaded to the default ThreadPoolExecutor
        so the event loop stays unblocked.
        """
        loop = asyncio.get_event_loop()

        def _sync_extract() -> ExtractedResume:
            sections = self.split_sections(raw_text)

            skills    = self.extract_skills(raw_text)
            total_exp = self.calc_total_experience(raw_text)
            name      = self.extract_name(raw_text)
            email     = self.extract_email(raw_text)
            phone     = self.extract_phone(raw_text)
            work_exp  = self.parse_work_experience(sections.get("experience", raw_text))
            education = self.parse_education(sections.get("education", raw_text))
            achievements = self.parse_achievements(
                sections.get("achievements", sections.get("other", ""))
            )

            resume = ExtractedResume(
                candidate_name=name,
                email=email,
                phone=phone,
                skills=skills,
                work_experience=work_exp,
                education=education,
                achievements=achievements,
                total_years_experience=total_exp,
                raw_text=raw_text,
            )
            resume.summary = self.build_summary(resume)
            return resume

        return await loop.run_in_executor(None, _sync_extract)
