"""Simple in-memory sliding-window rate limit (per IP)."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        s = get_settings()
        limit = int(getattr(s, "rate_limit_per_minute", 120) or 0)
        if limit <= 0:
            return await call_next(request)
        path = request.url.path
        # Strict on auth
        if path.startswith("/api/auth/") or path.startswith("/api/billing/checkout"):
            limit = min(limit, 30)
        if not path.startswith("/api"):
            return await call_next(request)
        if path in ("/api/health", "/api/sla", "/api/plans") or path.startswith(
            "/api/billing/webhook"
        ) or path.startswith("/api/godseye/"):
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._hits[ip]
        while window and window[0] < now - 60:
            window.popleft()
        if len(window) >= limit:
            return JSONResponse(
                {"detail": f"Rate limit exceeded ({limit}/min). Retry shortly."},
                status_code=429,
            )
        window.append(now)
        return await call_next(request)
