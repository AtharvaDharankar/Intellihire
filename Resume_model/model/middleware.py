"""
middleware.py — ResumeIQ
Production-grade FastAPI middleware stack:

  1. RequestIDMiddleware  — injects X-Request-ID into every request/response.
  2. StructuredLogMiddleware — emits one JSON log line per request with
                               method, path, status, latency, and request_id.
  3. TimingMiddleware     — adds X-Process-Time-Ms response header.

Usage (in main.py)
──────────────────
    from middleware import add_middleware
    add_middleware(app)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger("resumeiq.access")

_REQUEST_ID_HEADER = "X-Request-ID"
_TIMING_HEADER     = "X-Process-Time-Ms"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Request-ID middleware
# ─────────────────────────────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Accepts an incoming X-Request-ID (e.g. from a load-balancer) or generates
    a fresh UUID4.  The ID is attached to the request state so downstream
    handlers can log it, and echoed back in the response headers.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = request.headers.get(_REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = req_id

        response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = req_id
        return response


# ─────────────────────────────────────────────────────────────────────────────
# 2. Structured JSON access-log middleware
# ─────────────────────────────────────────────────────────────────────────────

class StructuredLogMiddleware(BaseHTTPMiddleware):
    """
    Emits one JSON log line per completed request.  The log record includes:

        {
          "request_id": "...",
          "method":     "POST",
          "path":       "/rank",
          "status":     200,
          "latency_ms": 42.7,
          "client_ip":  "127.0.0.1"
        }

    Compatible with log-aggregators (Datadog, CloudWatch, Loki, etc.)
    when you configure a JSONFormatter on the root logger.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t0  = time.perf_counter()
        req_id = getattr(request.state, "request_id", "-")

        response = await call_next(request)

        latency_ms = round((time.perf_counter() - t0) * 1_000, 1)
        record = {
            "request_id": req_id,
            "method":     request.method,
            "path":       request.url.path,
            "status":     response.status_code,
            "latency_ms": latency_ms,
            "client_ip":  request.client.host if request.client else "unknown",
        }

        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(level, json.dumps(record))

        return response


# ─────────────────────────────────────────────────────────────────────────────
# 3. Timing header middleware
# ─────────────────────────────────────────────────────────────────────────────

class TimingMiddleware(BaseHTTPMiddleware):
    """Adds X-Process-Time-Ms to every response so clients can measure E2E latency."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t0       = time.perf_counter()
        response = await call_next(request)
        elapsed  = round((time.perf_counter() - t0) * 1_000, 1)
        response.headers[_TIMING_HEADER] = str(elapsed)
        return response


# ─────────────────────────────────────────────────────────────────────────────
# Convenience installer
# ─────────────────────────────────────────────────────────────────────────────

def add_middleware(app: FastAPI) -> None:
    """
    Add all ResumeIQ middleware to the FastAPI app.
    Order matters — Starlette applies middleware inside-out, so the *last*
    added runs *first* in the request direction.
    """
    # Outermost → innermost request direction:
    # RequestID → StructuredLog → Timing → route handler
    app.add_middleware(TimingMiddleware)
    app.add_middleware(StructuredLogMiddleware)
    app.add_middleware(RequestIDMiddleware)
