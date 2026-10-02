"""
cli.py — ResumeIQ
Rich terminal runner for the full pipeline.

Replaces test_pipeline.py with:
  • Live progress bar during batch extraction
  • Colour-coded ranked results table
  • Per-candidate summary cards
  • Cache stats at the end
  • Optional JSON dump

Usage
─────
    python cli.py                          # uses built-in mock data
    python cli.py --resumes path/to/*.txt  # real resume files
    python cli.py --out results.json       # also save JSON
    python cli.py --no-rich               # plain text (CI-friendly)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from pathlib import Path
import glob

# ─────────────────────────────────────────────────────────────────────────────
# Optional Rich dependency (graceful fallback to plain print)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich import box
    _RICH = True
except ImportError:
    _RICH = False

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline imports
# ─────────────────────────────────────────────────────────────────────────────

from batch_processor import BatchProgress, make_batch_extractor
from cache import get_cache
from models import JobDescription, SeniorityLevel
from scorer import ResumeRanker

# ─────────────────────────────────────────────────────────────────────────────
# Mock data (used when no --resumes are provided)
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

MOCK_RESUMES = [
    textwrap.dedent("""\
        Alice Chen
        alice.chen@email.com | +1-415-555-0101
        EXPERIENCE
        2018 – Present  Senior ML Engineer  DeepMind Corp
        Led NLP pipeline development using PyTorch, Transformers, Hugging Face.
        Deployed to AWS EKS via Kubernetes. Built MLOps workflows with Airflow.
        Technologies: python, pytorch, transformers, kubernetes, aws, docker, mlops, sql, fastapi
        2016 – 2018  ML Engineer  StartupAI Inc
        Trained BERT-based classifiers; REST APIs with FastAPI.
        EDUCATION
        M.S. Computer Science, Stanford University, 2016
        ACHIEVEMENTS
        • Reduced model inference latency by 40% via TorchScript optimisation.
        • Shipped RAG pipeline serving 2M queries/day on AWS.
    """),
    textwrap.dedent("""\
        Bob Martinez
        bob.martinez@email.com
        EXPERIENCE
        2017 – Present  Senior DevOps / Platform Engineer  CloudScale Ltd
        Managed Kubernetes clusters on GCP and AWS. CI/CD with GitHub Actions.
        Technologies: kubernetes, docker, aws, gcp, terraform, python
        2015 – 2017  DevOps Engineer  FinTech Corp
        Automated infrastructure with Terraform and Helm.
        EDUCATION
        B.S. Information Technology, UCLA, 2015
        ACHIEVEMENTS
        • Reduced deployment time by 60% through automated pipelines.
        • Managed 200+ microservices across 3 regions.
    """),
    textwrap.dedent("""\
        Carol Nguyen
        carol.nguyen@gmail.com | +1-312-555-0303
        EXPERIENCE
        2022 – Present  Junior Data Scientist  Analytics House
        Built scikit-learn classifiers and pandas ETL pipelines.
        Technologies: python, scikit-learn, pandas, sql, aws, flask
        EDUCATION
        B.S. Statistics, University of Chicago, 2022
        ACHIEVEMENTS
        • Automated weekly reporting saving 5 hours/week.
    """),
    textwrap.dedent("""\
        David Kim
        d.kim@researchlab.ai
        EXPERIENCE
        2013 – Present  Principal NLP Researcher  OpenResearch Institute
        12 years NLP. Fine-tuned GPT and BERT at scale using PyTorch, Hugging Face.
        Built FAISS indexes for billion-scale semantic search.
        Technologies: python, pytorch, transformers, bert, gpt, faiss, langchain,
                      nlp, deep learning, machine learning, kubernetes, docker, aws
        2010 – 2013  ML Engineer  TechGiant Corp
        Spark-based feature pipelines and XGBoost models.
        Technologies: spark, python, sql, kafka, xgboost
        EDUCATION
        Ph.D. Computational Linguistics, MIT, 2013
        ACHIEVEMENTS
        • Published 8 peer-reviewed NLP papers with 1,200+ citations.
        • Built semantic search serving 50M users/day with 99.9% uptime.
        • Reduced LLM inference cost by $200K/year through quantisation.
    """),
    textwrap.dedent("""\
        Eva Johnson
        eva.johnson@dev.io | +44-20-7946-0505
        EXPERIENCE
        2019 – Present  Senior Software Engineer  WebCo International
        Full-stack React, Node.js, PostgreSQL. Integrated ML models via FastAPI.
        Technologies: python, javascript, react, node.js, fastapi, sql, docker, aws
        2016 – 2019  Software Engineer  AgencyX
        Microservices in Python and Go.
        Technologies: python, go, kubernetes, docker, redis
        EDUCATION
        B.S. Computer Science, Imperial College London, 2016
        ACHIEVEMENTS
        • Shipped SaaS product used by 10K+ businesses.
        • Reduced API latency by 30% via query optimisation.
    """),
]


# ─────────────────────────────────────────────────────────────────────────────
# Plain-text fallback helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plain_print_results(response, cache_stats: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  Ranking complete in {response.processing_time_seconds:.2f}s")
    print(f"  Job: {response.job_title}   Candidates: {response.total_resumes_processed}")
    print(f"{'='*70}")
    print(f"\n{'Rank':<5} {'Candidate':<24} {'Fit':>5} {'Skills':>7} {'Exp':>6} {'Sem':>6}")
    print("─" * 58)
    for c in response.ranked_candidates:
        print(
            f"#{c.rank:<4} {c.candidate_name:<24} "
            f"{c.fit_score:>5.1f} "
            f"{c.technical_skills_score:>7.1f} "
            f"{c.experience_depth_score:>6.1f} "
            f"{c.semantic_alignment_score:>6.1f}"
        )
    print("\n─── Summaries " + "─" * 56)
    for c in response.ranked_candidates:
        print(f"\n#{c.rank} {c.candidate_name}  [{c.fit_score:.1f}/100]")
        print(f"  {c.summary}")
        if c.matched_skills:
            print(f"  ✅ Matched: {', '.join(c.matched_skills[:5])}")
        if c.missing_skills:
            print(f"  ❌ Missing: {', '.join(c.missing_skills[:3])}")
    print(f"\nCache: {cache_stats}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Rich-enhanced output
# ─────────────────────────────────────────────────────────────────────────────

def _rich_print_results(response, cache_stats: dict, console: "Console") -> None:
    # ── Rankings table ───────────────────────────────────────────────────────
    table = Table(
        title=f"[bold cyan]ResumeIQ Rankings[/] — {response.job_title}",
        box=box.ROUNDED,
        show_lines=True,
        highlight=True,
    )
    table.add_column("Rank",      style="bold yellow",  justify="center", width=6)
    table.add_column("Candidate", style="bold white",   min_width=20)
    table.add_column("Email",     style="dim",          min_width=22)
    table.add_column("Fit (/100)",style="bold green",   justify="right", width=10)
    table.add_column("Skills",    style="cyan",         justify="right", width=8)
    table.add_column("Exp",       style="magenta",      justify="right", width=6)
    table.add_column("Semantic",  style="blue",         justify="right", width=9)

    colors = {1: "bold green", 2: "green", 3: "yellow"}

    for c in response.ranked_candidates:
        color = colors.get(c.rank, "white")
        table.add_row(
            f"[{color}]#{c.rank}[/]",
            f"[{color}]{c.candidate_name}[/]",
            c.email or "—",
            f"[{color}]{c.fit_score:.1f}[/]",
            f"{c.technical_skills_score:.1f}",
            f"{c.experience_depth_score:.1f}",
            f"{c.semantic_alignment_score:.1f}",
        )

    console.print()
    console.print(table)
    console.print(
        f"[dim]Processed {response.total_resumes_processed} resumes "
        f"in {response.processing_time_seconds:.2f}s[/dim]\n"
    )

    # ── Per-candidate summary cards ──────────────────────────────────────────
    for c in response.ranked_candidates:
        matched = ", ".join(c.matched_skills[:6]) or "none"
        missing = ", ".join(c.missing_skills[:3]) or "none"
        body = (
            f"{c.summary}\n\n"
            f"[green]✅ Matched:[/green] {matched}\n"
            f"[red]❌ Missing:[/red] {missing}"
        )
        console.print(Panel(
            body,
            title=f"[bold]#{c.rank} {c.candidate_name}[/bold]  "
                  f"[bold yellow]{c.fit_score:.1f}/100[/bold yellow]",
            border_style="blue" if c.rank == 1 else "dim",
            padding=(0, 1),
        ))

    # ── Cache stats ──────────────────────────────────────────────────────────
    console.print(Panel(
        f"Entries: [cyan]{cache_stats.get('entries', '?')}[/cyan]  |  "
        f"Hits: [green]{cache_stats.get('hits', 0)}[/green]  |  "
        f"Misses: [red]{cache_stats.get('misses', 0)}[/red]  |  "
        f"Hit-rate: [yellow]{cache_stats.get('hit_rate', 0):.1%}[/yellow]",
        title="[dim]Cache Stats[/dim]",
        border_style="dim",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Main async runner
# ─────────────────────────────────────────────────────────────────────────────

async def run(resume_texts: list[str], out_path: str | None, use_rich: bool) -> None:
    console = Console() if (use_rich and _RICH) else None

    if console:
        console.rule("[bold cyan]ResumeIQ Pipeline[/bold cyan]")
        console.print(
            f"[bold]Job:[/bold] {MOCK_JD.title} @ {MOCK_JD.company}\n"
            f"[bold]Required:[/bold] {', '.join(MOCK_JD.required_skills)}\n"
            f"[bold]Target:[/bold] {MOCK_JD.seniority_target} | "
            f"{MOCK_JD.min_years_experience}+ yrs\n"
            f"[bold]Resumes:[/bold] {len(resume_texts)}"
        )
    else:
        print(f"\nResumeIQ — {MOCK_JD.title} @ {MOCK_JD.company}")
        print(f"Resumes: {len(resume_texts)}")

    extractor = make_batch_extractor()
    ranker    = ResumeRanker()

    # ── Extraction with live progress bar ────────────────────────────────────
    if console:
        progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
            TimeElapsedColumn(),
            console=console,
        )
        task_id = progress_bar.add_task("Extracting resumes …", total=len(resume_texts))

        def on_progress(p: BatchProgress):
            progress_bar.update(task_id, completed=p.completed)

        with progress_bar:
            extracted = await extractor.extract_batch(resume_texts, on_progress=on_progress)
    else:
        completed = [0]

        def on_progress(p: BatchProgress):
            print(f"\r  Extracting … {p.percent:.0f}%", end="", flush=True)

        extracted = await extractor.extract_batch(resume_texts, on_progress=on_progress)
        print()

    # ── Scoring ──────────────────────────────────────────────────────────────
    if console:
        with console.status("[bold green]Scoring & ranking …"):
            response = await ranker.rank(MOCK_JD, extracted)
    else:
        print("  Scoring & ranking …")
        response = await ranker.rank(MOCK_JD, extracted)

    # ── Results ──────────────────────────────────────────────────────────────
    cache_stats = await get_cache().stats()
    if console:
        _rich_print_results(response, cache_stats, console)
    else:
        _plain_print_results(response, cache_stats)

    # ── Optional JSON dump ───────────────────────────────────────────────────
    if out_path:
        with open(out_path, "w") as f:
            json.dump(response.model_dump(), f, indent=2)
        msg = f"Full JSON saved to {out_path}"
        if console:
            console.print(f"[dim]💾 {msg}[/dim]")
        else:
            print(f"\n{msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="ResumeIQ CLI")
    parser.add_argument(
        "--resumes", nargs="+", metavar="FILE",
        help="Path(s) or wildcards (e.g., folder/*.txt) to resume files.",
    )
    parser.add_argument(
        "--out", metavar="FILE",
        help="Save full JSON results to this path.",
    )
    parser.add_argument(
        "--no-rich", action="store_true",
        help="Disable Rich formatting.",
    )
    args = parser.parse_args()

    if args.resumes:
        # Handle Windows wildcard expansion manually 
        expanded_paths = []
        for pattern in args.resumes:
            matches = glob.glob(pattern)
            if matches:
                expanded_paths.extend(matches)
            else:
                expanded_paths.append(pattern) # Fallback for direct filenames
        
        # Read the files from the expanded list 
        texts = [Path(p).read_text(encoding="utf-8") for p in expanded_paths]
    else:
        texts = MOCK_RESUMES

    asyncio.run(run(texts, out_path=args.out, use_rich=not args.no_rich))


if __name__ == "__main__":
    main()
