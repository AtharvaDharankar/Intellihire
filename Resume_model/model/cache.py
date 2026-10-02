"""
cache.py — ResumeIQ
Lightweight async in-process cache for extracted resumes.

Keyed by SHA-256 of the raw resume text so identical submissions
(e.g. the same PDF uploaded twice in a 200-resume batch) are parsed
only once.  Drop-in replacement for a Redis client: swap
``InMemoryResumeCache`` with ``RedisResumeCache`` (stub below) when
you move to a multi-worker deployment.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Optional

from models import ExtractedResume

logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────

class ResumeCache:
    async def get(self, raw_text: str) -> Optional[ExtractedResume]:
        raise NotImplementedError

    async def set(self, raw_text: str, resume: ExtractedResume) -> None:
        raise NotImplementedError

    async def stats(self) -> dict:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# In-process LRU-ish cache  (single-process; safe for asyncio)
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryResumeCache(ResumeCache):
    """
    Thread-safe (asyncio) in-memory cache with TTL and max-size eviction.

    Parameters
    ----------
    max_size : int
        Maximum number of entries before LRU eviction kicks in.
    ttl_seconds : int
        Entries older than this are treated as misses and evicted lazily.
    """

    def __init__(self, max_size: int = 1_000, ttl_seconds: int = 3600) -> None:
        self._store:  dict[str, tuple[ExtractedResume, float]] = {}
        self._max    = max_size
        self._ttl    = ttl_seconds
        self._hits   = 0
        self._misses = 0
        self._lock   = asyncio.Lock()

    async def get(self, raw_text: str) -> Optional[ExtractedResume]:
        key = _sha256(raw_text)
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            resume, ts = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            logger.debug("Cache HIT for key %s…", key[:8])
            return resume

    async def set(self, raw_text: str, resume: ExtractedResume) -> None:
        key = _sha256(raw_text)
        async with self._lock:
            if len(self._store) >= self._max:
                # Evict oldest entry (insertion-order dict, Python 3.7+)
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
                logger.debug("Cache evicted oldest entry.")
            self._store[key] = (resume, time.monotonic())

    async def stats(self) -> dict:
        async with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._store),
                "hits":    self._hits,
                "misses":  self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
                "max_size": self._max,
                "ttl_seconds": self._ttl,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Redis stub — swap in when running multiple uvicorn workers
# ─────────────────────────────────────────────────────────────────────────────

class RedisResumeCache(ResumeCache):  # pragma: no cover
    """
    Example Redis-backed cache using ``redis.asyncio``.

    Install:  pip install redis[asyncio]
    Usage:    cache = RedisResumeCache(url="redis://localhost:6379/0")
    """

    def __init__(self, url: str = "redis://localhost:6379/0", ttl: int = 3600):
        try:
            import redis.asyncio as aioredis  # type: ignore
        except ImportError:
            raise RuntimeError("Install redis[asyncio]: pip install redis[asyncio]")
        self._redis = aioredis.from_url(url, decode_responses=False)
        self._ttl   = ttl
        self._hits  = 0
        self._misses = 0

    async def get(self, raw_text: str) -> Optional[ExtractedResume]:
        import pickle
        key  = f"resumeiq:{_sha256(raw_text)}"
        data = await self._redis.get(key)
        if data is None:
            self._misses += 1
            return None
        self._hits += 1
        return pickle.loads(data)  # noqa: S301

    async def set(self, raw_text: str, resume: ExtractedResume) -> None:
        import pickle
        key  = f"resumeiq:{_sha256(raw_text)}"
        data = pickle.dumps(resume)
        await self._redis.setex(key, self._ttl, data)

    async def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "backend": "redis",
            "hits":    self._hits,
            "misses":  self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────────────────────

_cache: ResumeCache = InMemoryResumeCache()


def get_cache() -> ResumeCache:
    return _cache


def configure_cache(cache: ResumeCache) -> None:
    global _cache
    _cache = cache
