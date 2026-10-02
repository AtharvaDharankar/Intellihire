"""
main.py — ResumeIQ  (v2)
FastAPI application exposing:
  POST /rank            — full batch ranking (up to 200 resumes)
  POST /rank/stream     — Server-Sent Events streaming with per-resume progress
  GET  /cache/stats     — cache hit-rate and entry count
  GET  /health          — liveness probe
  GET  /docs            — automatic OpenAPI docs (built-in)

Run locally:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import os
import uuid
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from file_parser import extract_text_from_file

from batch_processor import BatchExtractor, BatchProgress, make_batch_extractor
from cache import get_cache
from middleware import add_middleware
from models import RankingResponse, ResumeRankRequest
from scorer import ResumeRanker

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("resumeiq.api")

# ─────────────────────────────────────────────────────────────────────────────
# Singletons
# ─────────────────────────────────────────────────────────────────────────────

_batch_extractor: BatchExtractor = make_batch_extractor()
_ranker: ResumeRanker = ResumeRanker()

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (warm-up embedding model on startup)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ResumeIQ v2 starting — warming embedding model …")
    from scorer import _get_embedder
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _get_embedder)
    logger.info("Embedding model ready. Serving requests.")
    yield
    logger.info("ResumeIQ shutting down.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ResumeIQ",
    description=(
        "ML-powered resume ranking pipeline. "
        "Submit a Job Description and raw resume texts; "
        "receive a ranked list with Fit-Scores and per-candidate summaries."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
add_middleware(app)   # RequestID, StructuredLog, Timing

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "service": "ResumeIQ", "version": "2.0.0"}


@app.get("/cache/stats", tags=["ops"])
async def cache_stats():
    """Return hit-rate and entry count for the in-process resume cache."""
    return await get_cache().stats()


# ── POST /rank/files ────────────────────────────────────────────────────────
@app.post(
    "/rank/files",
    response_model=RankingResponse,
    summary="Upload resumes and rank them against a Job Description",
    tags=["pipeline"],
)
async def rank_files(
    title: str = Form(...),
    company: str = Form(...),
    description: str = Form(...),
    required_skills: str = Form(""),
    preferred_skills: str = Form(""),
    min_years_experience: float = Form(3.0),
    seniority_target: str = Form("Senior"),
    files: list[UploadFile] = File(...),
):
    from models import JobDescription, SeniorityLevel
    
    jd = JobDescription(
        title=title,
        company=company,
        description=description,
        required_skills=[s.strip() for s in required_skills.split(",") if s.strip()],
        preferred_skills=[s.strip() for s in preferred_skills.split(",") if s.strip()],
        min_years_experience=min_years_experience,
        seniority_target=SeniorityLevel(seniority_target)
    )

    t0 = time.perf_counter()
    n = len(files)
    if n > 200:
        raise HTTPException(422, "Batch limit is 200 resumes per request.")

    logger.info(
        "POST /rank/files — %d resumes for '%s' @ %s",
        n, jd.title, jd.company,
    )

    resume_texts = []
    saved_filenames = []

    for file in files:
        # Save file to uploads/ directory
        ext = file.filename.split('.')[-1] if '.' in file.filename else ''
        unique_name = f"{uuid.uuid4().hex}_{file.filename}"
        filepath = os.path.join("uploads", unique_name)
        
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
            
        saved_filenames.append(unique_name)
        
        # Extract text from the bytes
        text = extract_text_from_file(file.filename, content)
        resume_texts.append(text)

    # ── Extraction ───────────────────────────────────────────────────────────
    try:
        extracted = await _batch_extractor.extract_batch(resume_texts)
        # Link filenames back to extracted resumes
        for i, ext_res in enumerate(extracted):
            ext_res.filename = saved_filenames[i]
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(500, f"Extraction error: {exc}") from exc

    # ── Scoring + ranking ────────────────────────────────────────────────────
    try:
        response = await _ranker.rank(
            jd=jd,
            resumes=extracted,
            start_time=t0,
        )
    except Exception as exc:
        logger.exception("Ranking failed")
        raise HTTPException(500, f"Ranking error: {exc}") from exc

    return response


# ── POST /rank ──────────────────────────────────────────────────────────────

@app.post(
    "/rank",
    response_model=RankingResponse,
    summary="Rank resumes against a Job Description",
    tags=["pipeline"],
)
async def rank_resumes(request: ResumeRankRequest) -> RankingResponse:
    """
    **Pipeline**

    1. Batch-extract all resume texts via ``BatchExtractor`` (async, cached,
       semaphore-throttled at 16 concurrent parses).
    2. Embed resume summaries into a FAISS inner-product index.
    3. Score each candidate on three dimensions (Skills / Experience / Semantic).
    4. Return a ranked list with Fit-Scores and 2-sentence summaries.

    Accepts up to **200 resumes** per call.
    For progress streaming, use ``POST /rank/stream`` instead.
    """
    t0 = time.perf_counter()

    n = len(request.resume_texts)
    if n > 200:
        raise HTTPException(422, "Batch limit is 200 resumes per request.")

    logger.info(
        "POST /rank — %d resumes for '%s' @ %s",
        n, request.job_description.title, request.job_description.company,
    )

    # ── Extraction ───────────────────────────────────────────────────────────
    try:
        extracted = await _batch_extractor.extract_batch(request.resume_texts)
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(500, f"Extraction error: {exc}") from exc

    logger.info("Extraction complete: %d resumes parsed.", len(extracted))

    # ── Scoring + ranking ────────────────────────────────────────────────────
    try:
        response = await _ranker.rank(
            jd=request.job_description,
            resumes=extracted,
            start_time=t0,
        )
    except Exception as exc:
        logger.exception("Ranking failed")
        raise HTTPException(500, f"Ranking error: {exc}") from exc

    top = response.ranked_candidates[0] if response.ranked_candidates else None
    logger.info(
        "Ranking done in %.2fs. Top: %s (%.1f/100)",
        response.processing_time_seconds,
        top.candidate_name if top else "N/A",
        top.fit_score if top else 0,
    )
    return response


# ── POST /rank/stream ────────────────────────────────────────────────────────

@app.post(
    "/rank/stream",
    summary="Stream ranking progress via Server-Sent Events",
    tags=["pipeline"],
    response_class=StreamingResponse,
)
async def rank_resumes_stream(request: ResumeRankRequest):
    """
    Identical pipeline to ``POST /rank`` but streams JSON progress events via
    **Server-Sent Events** (SSE).

    Event types
    -----------
    ``progress``   — emitted after each resume is parsed.
    ``result``     — final ranked list (same shape as ``/rank``).
    ``error``      — on failure.

    Example client (JavaScript)
    ---------------------------
    ```js
    const evtSource = new EventSource('/rank/stream');
    evtSource.addEventListener('progress', e => console.log(JSON.parse(e.data)));
    evtSource.addEventListener('result',   e => renderResults(JSON.parse(e.data)));
    ```
    """
    n = len(request.resume_texts)
    if n > 200:
        raise HTTPException(422, "Batch limit is 200 resumes per request.")

    async def event_generator():
        t0 = time.perf_counter()

        # ── SSE helper ───────────────────────────────────────────────────────
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        # ── Progress callback — runs in the asyncio thread ───────────────────
        # We push SSE events via a queue to avoid blocking the extractor thread.
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def on_progress(p: BatchProgress):
            asyncio.get_event_loop().call_soon_threadsafe(
                queue.put_nowait, p.as_dict()
            )

        # ── Launch extraction in background ──────────────────────────────────
        extract_task = asyncio.create_task(
            _batch_extractor.extract_batch(request.resume_texts, on_progress=on_progress)
        )

        # Stream progress events while extraction runs
        while not extract_task.done():
            try:
                progress_data = await asyncio.wait_for(queue.get(), timeout=0.2)
                yield sse("progress", progress_data)
            except asyncio.TimeoutError:
                pass   # keep loop spinning; check task.done() next iteration

        # Drain remaining progress events
        while not queue.empty():
            yield sse("progress", queue.get_nowait())

        # ── Retrieve extraction results ───────────────────────────────────────
        try:
            extracted = await extract_task
        except Exception as exc:
            yield sse("error", {"message": str(exc)})
            return

        # ── Rank ─────────────────────────────────────────────────────────────
        try:
            response = await _ranker.rank(
                jd=request.job_description,
                resumes=extracted,
                start_time=t0,
            )
            yield sse("result", response.model_dump())
        except Exception as exc:
            yield sse("error", {"message": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )
