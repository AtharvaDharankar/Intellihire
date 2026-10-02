"""
models.py — ResumeIQ
All Pydantic data models for the extraction, scoring, and API layers.
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class SeniorityLevel(str, Enum):
    INTERN    = "Intern"
    JUNIOR    = "Junior"
    MID       = "Mid-Level"
    SENIOR    = "Senior"
    LEAD      = "Lead"
    PRINCIPAL = "Principal"
    DIRECTOR  = "Director"


# ─────────────────────────────────────────────
# Extraction layer models
# ─────────────────────────────────────────────

class WorkExperience(BaseModel):
    company:        str
    role:           str
    duration_years: float = 0.0
    seniority:      SeniorityLevel = SeniorityLevel.MID
    description:    str = ""
    technologies:   List[str] = Field(default_factory=list)


class Education(BaseModel):
    degree:      str
    field:       str
    institution: str
    year:        Optional[int]  = None
    gpa:         Optional[float] = None


class Achievement(BaseModel):
    description: str
    metric:      Optional[str] = None   # e.g. "40%", "$200K"
    impact:      Optional[str] = None   # "quantified" | "qualitative"


class ExtractedResume(BaseModel):
    candidate_name:         str
    email:                  Optional[str] = None
    phone:                  Optional[str] = None
    skills:                 List[str]        = Field(default_factory=list)
    work_experience:        List[WorkExperience] = Field(default_factory=list)
    education:              List[Education]      = Field(default_factory=list)
    achievements:           List[Achievement]    = Field(default_factory=list)
    total_years_experience: float = 0.0
    raw_text:               str   = ""
    summary:                Optional[str] = None   # compact string used for embedding
    filename:               Optional[str] = None


# ─────────────────────────────────────────────
# Job Description model
# ─────────────────────────────────────────────

class JobDescription(BaseModel):
    title:                str
    company:              str
    required_skills:      List[str]
    preferred_skills:     List[str] = Field(default_factory=list)
    min_years_experience: float     = 3.0
    seniority_target:     SeniorityLevel = SeniorityLevel.SENIOR
    description:          str


# ─────────────────────────────────────────────
# Scoring / output models
# ─────────────────────────────────────────────

class ScoredResume(BaseModel):
    rank:                    int
    candidate_name:          str
    email:                   Optional[str] = None
    fit_score:               float = Field(..., ge=0, le=100, description="Composite 0-100 score")
    technical_skills_score:  float = Field(..., ge=0, le=100)
    experience_depth_score:  float = Field(..., ge=0, le=100)
    semantic_alignment_score: float = Field(..., ge=0, le=100)
    matched_skills:          List[str] = Field(default_factory=list)
    missing_skills:          List[str] = Field(default_factory=list)
    summary:                 str       = ""   # 2-sentence human-readable explanation
    filename:                Optional[str] = None


class RankingResponse(BaseModel):
    job_title:                 str
    total_resumes_processed:   int
    ranked_candidates:         List[ScoredResume]
    processing_time_seconds:   float


# ─────────────────────────────────────────────
# API request model
# ─────────────────────────────────────────────

class ResumeRankRequest(BaseModel):
    job_description: JobDescription
    resume_texts:    List[str] = Field(..., min_length=1, description="Raw resume text strings")
