"""
batch_processor.py — ResumeIQ
Scalable async batch extraction for 100+ resumes.

Design goals
────────────
• Concurrency cap    — asyncio.Semaphore prevents thundering-herd on
                       the spaCy / embedding thread pool.
• Cache integration  — identical resume texts are parsed exactly once.
• Progress callbacks — callers can stream progress to SSE or a CLI bar.
• Chunked submission — large batches are split into CHUNK_SIZE windows
                       so memory stays bounded.
• Partial failures   — a bad resume yields a sentinel record rather than
                       crashing the whole batch.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from cache import ResumeCache, get_cache
from extractor import ResumeExtractor
from models import ExtractedResume

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────────────

MAX_CONCURRENCY = 16   # simultaneous in-flight extractions
CHUNK_SIZE      = 25   # resumes processed per asyncio.gather chunk


# ─────────────────────────────────────────────────────────────────────────────
# Progress tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchProgress:
    total:     int
    completed: int = 0
    failed:    int = 0
    start_ts:  float = field(default_factory=time.monotonic)

    @property
    def percent(self) -> float:
        return round(100.0 * self.completed / self.total, 1) if self.total else 0.0

    @property
    def elapsed(self) -> float:
        return round(time.monotonic() - self.start_ts, 2)

    @property
    def eta_seconds(self) -> Optional[float]:
        if self.completed == 0:
            return None
        rate = self.completed / self.elapsed
        remaining = self.total - self.completed
        return round(remaining / rate, 1)

    def as_dict(self) -> dict:
        return {
            "total":     self.total,
            "completed": self.completed,
            "failed":    self.failed,
            "percent":   self.percent,
            "elapsed_s": self.elapsed,
            "eta_s":     self.eta_seconds,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel resume (returned on parse failure)
# ─────────────────────────────────────────────────────────────────────────────

def _error_resume(raw_text: str, error: str) -> ExtractedResume:
    return ExtractedResume(
        candidate_name="[PARSE ERROR]",
        raw_text=raw_text[:200],
        summary=f"Extraction failed: {error}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core batch extractor
# ─────────────────────────────────────────────────────────────────────────────

class BatchExtractor:
    """
    Extracts structured data from a list of raw resume texts with:
    - Semaphore-bounded concurrency
    - Per-item cache lookups
    - Chunked processing to bound memory
    - Optional progress callbacks

    Parameters
    ----------
    extractor : ResumeExtractor
        Underlying single-resume parser.
    cache : ResumeCache | None
        Cache backend; defaults to the global singleton.
    max_concurrency : int
        Max simultaneous extractions.
    chunk_size : int
        Number of resumes per asyncio.gather window.
    """

    def __init__(
        self,
        extractor: Optional[ResumeExtractor] = None,
        cache: Optional[ResumeCache] = None,
        max_concurrency: int = MAX_CONCURRENCY,
        chunk_size: int = CHUNK_SIZE,
    ) -> None:
        self._extractor       = extractor or ResumeExtractor()
        self._cache           = cache or get_cache()
        self._semaphore       = asyncio.Semaphore(max_concurrency)
        self._chunk_size      = chunk_size

    async def _extract_one(
        self,
        raw_text: str,
        progress: BatchProgress,
        on_progress: Optional[Callable[[BatchProgress], None]],
    ) -> ExtractedResume:
        """Extract a single resume with semaphore + cache guard."""
        async with self._semaphore:
            # ── Cache check ──────────────────────────────────────────────────
            cached = await self._cache.get(raw_text)
            if cached is not None:
                progress.completed += 1
                if on_progress:
                    on_progress(progress)
                return cached

            # ── Parse ────────────────────────────────────────────────────────
            try:
                resume = await self._extractor.extract(raw_text)
                await self._cache.set(raw_text, resume)
            except Exception as exc:
                logger.warning("Extraction error: %s", exc)
                resume = _error_resume(raw_text, str(exc))
                progress.failed += 1

            progress.completed += 1
            if on_progress:
                on_progress(progress)
            return resume

    async def extract_batch(
        self,
        texts: List[str],
        on_progress: Optional[Callable[[BatchProgress], None]] = None,
    ) -> List[ExtractedResume]:
        """
        Extract all resumes, returning results in the same order as ``texts``.

        Parameters
        ----------
        texts : list[str]
            Raw resume texts.
        on_progress : callable | None
            Called after every completed resume.  Receives a
            ``BatchProgress`` snapshot — useful for SSE or tqdm.
        """
        if not texts:
            return []

        progress = BatchProgress(total=len(texts))
        logger.info("BatchExtractor: starting %d resumes (chunk=%d, concurrency=%d)",
                    len(texts), self._chunk_size, self._semaphore._value)

        all_results: List[ExtractedResume] = []

        # Process in chunks to keep asyncio queue from exploding on huge batches
        for chunk_start in range(0, len(texts), self._chunk_size):
            chunk = texts[chunk_start : chunk_start + self._chunk_size]
            tasks = [
                self._extract_one(text, progress, on_progress)
                for text in chunk
            ]
            chunk_results = await asyncio.gather(*tasks)
            all_results.extend(chunk_results)
            logger.debug(
                "Chunk %d–%d done. Progress: %s%%",
                chunk_start, chunk_start + len(chunk), progress.percent,
            )

        logger.info(
            "BatchExtractor finished: %d ok, %d failed, %.2fs elapsed.",
            progress.completed - progress.failed,
            progress.failed,
            progress.elapsed,
        )
        return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────────────────────

def make_batch_extractor(
    max_concurrency: int = MAX_CONCURRENCY,
    chunk_size: int = CHUNK_SIZE,
) -> BatchExtractor:
    return BatchExtractor(
        extractor=ResumeExtractor(),
        cache=get_cache(),
        max_concurrency=max_concurrency,
        chunk_size=chunk_size,
    )
