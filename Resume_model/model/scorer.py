"""
scorer.py — ResumeIQ
Weighted scoring engine:
  • Technical Skills  (50%) — ontology-based skill overlap
  • Experience Depth  (25%) — years + seniority ordinal
  • Semantic Alignment(25%) — cosine similarity via sentence-transformers / FAISS
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from models import (
    ExtractedResume,
    JobDescription,
    RankingResponse,
    ScoredResume,
    SeniorityLevel,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

WEIGHT_SKILLS     = 0.50
WEIGHT_EXPERIENCE = 0.25
WEIGHT_SEMANTIC   = 0.25

# Seniority ordinal mapping (higher = more senior)
SENIORITY_ORDINAL: dict[SeniorityLevel, int] = {
    SeniorityLevel.INTERN:    0,
    SeniorityLevel.JUNIOR:    1,
    SeniorityLevel.MID:       2,
    SeniorityLevel.SENIOR:    3,
    SeniorityLevel.LEAD:      4,
    SeniorityLevel.PRINCIPAL: 5,
    SeniorityLevel.DIRECTOR:  6,
}

# Sentence-transformers model — local, no OpenAI key needed.
# Swap to "text-embedding-3-small" via openai client if you have a key.
_EMBED_MODEL_NAME = "./fine_tuned_resume_model_v2"
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model '%s' …", _EMBED_MODEL_NAME)
        _embedder = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedder


# ─────────────────────────────────────────────────────────────────────────────
# Sub-scorers
# ─────────────────────────────────────────────────────────────────────────────

def score_technical_skills(
    resume: ExtractedResume,
    jd: JobDescription,
) -> Tuple[float, List[str], List[str]]:
    """
    Score = weighted Jaccard-style overlap.
    Required skills have 2× weight vs preferred skills.

    Returns (score_0_100, matched_skills, missing_skills).
    """
    resume_skills_lower = {s.lower() for s in resume.skills}
    required_lower  = {s.lower() for s in jd.required_skills}
    preferred_lower = {s.lower() for s in jd.preferred_skills}

    req_matches  = required_lower  & resume_skills_lower
    pref_matches = preferred_lower & resume_skills_lower
    missing      = list(required_lower - resume_skills_lower)

    # Weighted numerator / denominator
    numerator   = 2 * len(req_matches) + len(pref_matches)
    denominator = 2 * len(required_lower) + len(preferred_lower) if (required_lower or preferred_lower) else 1

    score = (numerator / denominator) * 100.0
    matched = sorted(req_matches | pref_matches)

    return round(min(score, 100.0), 2), matched, missing


def score_experience_depth(
    resume: ExtractedResume,
    jd: JobDescription,
) -> float:
    """
    Composite experience score (0-100):
      • 60% — years of experience vs. JD minimum (soft cap at 2× min)
      • 40% — seniority gap between detected level and target level
    """
    # ── Years sub-score ──────────────────────────────────────────────────────
    min_yrs   = max(jd.min_years_experience, 1.0)
    candidate_yrs = resume.total_years_experience
    # Normalise: full marks at 1× min, diminishing gain up to 2× min
    yrs_score = min(candidate_yrs / min_yrs, 2.0) / 2.0 * 100.0

    # ── Seniority sub-score ──────────────────────────────────────────────────
    target_ord    = SENIORITY_ORDINAL[jd.seniority_target]
    max_seniority = max(
        (SENIORITY_ORDINAL.get(exp.seniority, 0) for exp in resume.work_experience),
        default=SENIORITY_ORDINAL[SeniorityLevel.JUNIOR],
    )
    # Full marks if candidate meets or exceeds target; linear penalty below
    if target_ord == 0:
        seniority_score = 100.0
    else:
        seniority_score = min(max_seniority / target_ord, 1.0) * 100.0

    return round(0.60 * yrs_score + 0.40 * seniority_score, 2)


# ─────────────────────────────────────────────────────────────────────────────
# FAISS-backed semantic scorer
# ─────────────────────────────────────────────────────────────────────────────

class SemanticIndex:
    """
    Builds a FAISS flat-L2 index over resume embeddings and scores them
    against the JD embedding using cosine similarity (via inner product on
    L2-normalised vectors).
    """

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._dim: int = 0

    def build(self, summaries: List[str]) -> None:
        embedder = _get_embedder()
        vectors  = embedder.encode(summaries, convert_to_numpy=True, show_progress_bar=False)
        vectors  = self._normalise(vectors)
        self._dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)   # Inner-product on unit vectors = cosine
        self._index.add(vectors.astype(np.float32))
        logger.info("FAISS index built: %d vectors of dim %d", len(summaries), self._dim)

    def query(self, jd_text: str, k: int) -> np.ndarray:
        """Return cosine similarity scores (0-1) for all k resumes."""
        if self._index is None:
            raise RuntimeError("Call build() before query().")
        embedder = _get_embedder()
        jd_vec   = embedder.encode([jd_text], convert_to_numpy=True, show_progress_bar=False)
        jd_vec   = self._normalise(jd_vec).astype(np.float32)
        scores, indices = self._index.search(jd_vec, k)
        # Re-order to original index order
        result = np.zeros(k, dtype=np.float32)
        result[indices[0]] = scores[0]
        return result

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return v / norms


# ─────────────────────────────────────────────────────────────────────────────
# Main Ranker
# ─────────────────────────────────────────────────────────────────────────────

class ResumeRanker:
    """
    Orchestrates extraction → scoring → ranking for a batch of resumes.

    Usage
    -----
    ranker = ResumeRanker()
    response = await ranker.rank(jd, extracted_resumes)
    """

    def __init__(self) -> None:
        self._semantic = SemanticIndex()

    async def rank(
        self,
        jd: JobDescription,
        resumes: List[ExtractedResume],
        start_time: float | None = None,
    ) -> RankingResponse:
        if not resumes:
            raise ValueError("No resumes provided.")

        t0 = start_time or time.perf_counter()

        # ── Build FAISS index asynchronously (offload heavy encode) ──────────
        summaries = [r.summary or r.raw_text[:512] for r in resumes]
        jd_text   = f"{jd.title}. {jd.description}. Skills: {', '.join(jd.required_skills)}"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._semantic.build, summaries)
        raw_semantic: np.ndarray = await loop.run_in_executor(
            None, self._semantic.query, jd_text, len(resumes)
        )

        # ── Score each resume ────────────────────────────────────────────────
        scored: List[ScoredResume] = []
        for idx, resume in enumerate(resumes):
            skill_score, matched, missing = score_technical_skills(resume, jd)
            exp_score    = score_experience_depth(resume, jd)
            sem_raw      = float(raw_semantic[idx])                # already in [0,1]
            sem_score    = round(max(sem_raw, 0.0) * 100.0, 2)    # scale to 0-100

            fit = round(
                WEIGHT_SKILLS     * skill_score
                + WEIGHT_EXPERIENCE * exp_score
                + WEIGHT_SEMANTIC   * sem_score,
                2,
            )

            scored.append(
                ScoredResume(
                    rank=0,             # assigned below after sort
                    candidate_name=resume.candidate_name,
                    email=resume.email,
                    fit_score=fit,
                    technical_skills_score=skill_score,
                    experience_depth_score=exp_score,
                    semantic_alignment_score=sem_score,
                    matched_skills=matched,
                    missing_skills=missing,
                    summary="",         # filled by summariser
                    filename=resume.filename,
                )
            )

        # ── Sort & assign ranks ──────────────────────────────────────────────
        scored.sort(key=lambda x: x.fit_score, reverse=True)
        for rank, s in enumerate(scored, start=1):
            s.rank = rank

        # ── Generate 2-sentence summaries ────────────────────────────────────
        for s, resume in zip(scored, {r.candidate_name: r for r in resumes}.values()):
            s.summary = _build_candidate_summary(s, jd)

        elapsed = round(time.perf_counter() - t0, 3)
        return RankingResponse(
            job_title=jd.title,
            total_resumes_processed=len(resumes),
            ranked_candidates=scored,
            processing_time_seconds=elapsed,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Summary builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_candidate_summary(s: ScoredResume, jd: JobDescription) -> str:
    """
    Rule-based 2-sentence summary (no extra LLM call — cheap & deterministic).
    Sentence 1: strengths.  Sentence 2: gaps or confirmation.
    """
    skill_pct  = round(s.technical_skills_score)
    exp_pct    = round(s.experience_depth_score)
    sem_pct    = round(s.semantic_alignment_score)

    # Sentence 1 — strongest dimension
    if skill_pct >= exp_pct and skill_pct >= sem_pct:
        matched_str = ", ".join(s.matched_skills[:4]) or "none"
        sent1 = (
            f"{s.candidate_name} shows strong technical alignment ({skill_pct}/100) "
            f"with key matched skills: {matched_str}."
        )
    elif exp_pct >= sem_pct:
        sent1 = (
            f"{s.candidate_name} demonstrates solid experience depth ({exp_pct}/100) "
            f"relative to the {jd.seniority_target} target level."
        )
    else:
        sent1 = (
            f"{s.candidate_name} has high contextual alignment ({sem_pct}/100) "
            f"with the {jd.title} role description."
        )

    # Sentence 2 — gap or praise
    if s.missing_skills:
        gap_str = ", ".join(s.missing_skills[:3])
        sent2 = f"Key gaps to address: {gap_str}."
    else:
        sent2 = (
            f"All required skills are covered, making them a strong fit "
            f"for {jd.company}'s {jd.title} opening."
        )

    return f"{sent1} {sent2}"
