"""
test_pipeline.py — ResumeIQ
Self-contained integration test.

Run:
    python test_pipeline.py

No server needed — exercises the full pipeline in-process.
"""

from __future__ import annotations

import asyncio
import json
import textwrap

from extractor import ResumeExtractor
from models import JobDescription, ResumeRankRequest, SeniorityLevel
from scorer import ResumeRanker

# ─────────────────────────────────────────────────────────────────────────────
# Mock Job Description
# ─────────────────────────────────────────────────────────────────────────────

MOCK_JD = JobDescription(
    title="Senior ML Engineer",
    company="Acme AI Labs",
    required_skills=[
        "python", "pytorch", "transformers", "mlops",
        "kubernetes", "aws", "sql", "docker",
    ],
    preferred_skills=["fastapi", "kafka", "spark", "faiss", "langchain"],
    min_years_experience=5.0,
    seniority_target=SeniorityLevel.SENIOR,
    description=(
        "We are looking for a Senior ML Engineer to design, build, and deploy "
        "large-scale machine-learning systems. You will work on NLP pipelines, "
        "LLM fine-tuning, and real-time inference infrastructure. Strong Python, "
        "deep learning frameworks (PyTorch), and cloud-native MLOps skills are essential."
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# Mock Resumes  (5 candidates with varied profiles)
# ─────────────────────────────────────────────────────────────────────────────

MOCK_RESUMES = [
    # ── Candidate 1: strong ML senior ────────────────────────────────────────
    textwrap.dedent("""\
        Alice Chen
        alice.chen@email.com | +1-415-555-0101

        EXPERIENCE
        2018 – Present
        Senior ML Engineer
        DeepMind Corp
        Led NLP pipeline development using PyTorch, Transformers, and Hugging Face.
        Deployed models to AWS EKS via Kubernetes. Built MLOps workflows with Airflow.
        Technologies: python, pytorch, transformers, kubernetes, aws, docker, mlops, sql, fastapi

        2016 – 2018
        ML Engineer
        StartupAI Inc
        Trained BERT-based classifiers; built REST APIs with FastAPI.

        EDUCATION
        M.S. Computer Science, Stanford University, 2016

        ACHIEVEMENTS
        • Reduced model inference latency by 40% via TorchScript optimisation.
        • Shipped RAG pipeline serving 2M queries/day on AWS.
    """),

    # ── Candidate 2: strong DevOps, weaker ML ────────────────────────────────
    textwrap.dedent("""\
        Bob Martinez
        bob.martinez@email.com

        EXPERIENCE
        2017 – Present
        Senior DevOps / Platform Engineer
        CloudScale Ltd
        Managed Kubernetes clusters on GCP and AWS. Designed CI/CD pipelines
        with GitHub Actions. Deployed Docker containers at scale.
        Technologies: kubernetes, docker, aws, gcp, terraform, prometheus, grafana, python

        2015 – 2017
        DevOps Engineer
        FinTech Corp
        Automated infrastructure with Terraform and Helm.

        EDUCATION
        B.S. Information Technology, UCLA, 2015

        ACHIEVEMENTS
        • Reduced deployment time by 60% through automated pipelines.
        • Managed fleet of 200+ microservices across 3 regions.
    """),

    # ── Candidate 3: junior data scientist ───────────────────────────────────
    textwrap.dedent("""\
        Carol Nguyen
        carol.nguyen@gmail.com | +1-312-555-0303

        EXPERIENCE
        2022 – Present
        Junior Data Scientist
        Analytics House
        Built scikit-learn classifiers and pandas ETL pipelines.
        Deployed simple Flask APIs on AWS Lambda.
        Technologies: python, scikit-learn, pandas, sql, aws, flask

        EDUCATION
        B.S. Statistics, University of Chicago, 2022

        ACHIEVEMENTS
        • Automated weekly reporting saving 5 hours/week.
    """),

    # ── Candidate 4: principal NLP researcher ────────────────────────────────
    textwrap.dedent("""\
        David Kim
        d.kim@researchlab.ai

        EXPERIENCE
        2013 – Present
        Principal NLP Researcher
        OpenResearch Institute
        12 years in NLP. Led transformer-based language model research.
        Fine-tuned GPT and BERT models at scale using PyTorch and Hugging Face.
        Built FAISS vector indexes for billion-scale semantic search.
        Technologies: python, pytorch, transformers, bert, gpt, faiss, langchain,
                      nlp, deep learning, machine learning, kubernetes, docker, aws

        2010 – 2013
        ML Engineer
        TechGiant Corp
        Spark-based feature pipelines and XGBoost models.
        Technologies: spark, python, sql, kafka, xgboost

        EDUCATION
        Ph.D. Computational Linguistics, MIT, 2013

        ACHIEVEMENTS
        • Published 8 peer-reviewed NLP papers with 1,200+ citations.
        • Built semantic search engine serving 50M users/day with 99.9% uptime.
        • Reduced LLM inference cost by $200K/year through quantisation.
    """),

    # ── Candidate 5: full-stack with some ML ─────────────────────────────────
    textwrap.dedent("""\
        Eva Johnson
        eva.johnson@dev.io | +44-20-7946-0505

        EXPERIENCE
        2019 – Present
        Senior Software Engineer
        WebCo International
        Built full-stack applications with React, Node.js, and PostgreSQL.
        Integrated basic ML models via REST APIs (scikit-learn, FastAPI).
        Technologies: python, javascript, react, node.js, fastapi, sql, docker, aws

        2016 – 2019
        Software Engineer
        AgencyX
        Developed microservices in Python and Go.
        Technologies: python, go, kubernetes, docker, redis

        EDUCATION
        B.S. Computer Science, Imperial College London, 2016

        ACHIEVEMENTS
        • Shipped SaaS product used by 10K+ businesses.
        • Reduced API latency by 30% via query optimisation.
    """),
]

# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_test():
    print("\n" + "=" * 70)
    print("  ResumeIQ — Integration Test")
    print("=" * 70)

    extractor = ResumeExtractor()
    ranker    = ResumeRanker()

    print(f"\n📋  Job:        {MOCK_JD.title} @ {MOCK_JD.company}")
    print(f"🔑  Required:   {', '.join(MOCK_JD.required_skills)}")
    print(f"⭐  Target:     {MOCK_JD.seniority_target} | {MOCK_JD.min_years_experience}+ yrs")
    print(f"\n🔄  Extracting {len(MOCK_RESUMES)} resumes …")

    # Async batch extraction
    extracted = await asyncio.gather(*[extractor.extract(t) for t in MOCK_RESUMES])

    for r in extracted:
        print(f"   ✔ {r.candidate_name:<20} | {r.total_years_experience:.0f} yrs | skills: {len(r.skills)}")

    print("\n⚙️  Scoring & ranking …")
    response = await ranker.rank(MOCK_JD, list(extracted))

    print(f"\n✅  Done in {response.processing_time_seconds:.2f}s")
    print(f"\n{'Rank':<5} {'Candidate':<22} {'Fit':>5} {'Skills':>7} {'Exp':>6} {'Sem':>6}")
    print("─" * 60)
    for c in response.ranked_candidates:
        print(
            f"#{c.rank:<4} {c.candidate_name:<22} "
            f"{c.fit_score:>5.1f} "
            f"{c.technical_skills_score:>7.1f} "
            f"{c.experience_depth_score:>6.1f} "
            f"{c.semantic_alignment_score:>6.1f}"
        )

    print("\n─── Summaries ───────────────────────────────────────────────────────")
    for c in response.ranked_candidates:
        print(f"\n#{c.rank} {c.candidate_name}")
        print(f"   {c.summary}")
        if c.matched_skills:
            print(f"   ✅ Matched: {', '.join(c.matched_skills[:5])}")
        if c.missing_skills:
            print(f"   ❌ Missing: {', '.join(c.missing_skills[:3])}")

    # Dump full JSON to file
    out_path = "ranking_result.json"
    with open(out_path, "w") as f:
        json.dump(response.model_dump(), f, indent=2)
    print(f"\n💾  Full JSON saved to {out_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(run_test())
