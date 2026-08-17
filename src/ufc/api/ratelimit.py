"""In-memory per-IP sliding-window rate limiter.

Single-process design: the HF Space runs one uvicorn worker (Dockerfile CMD),
so a process-local dict is a correct global limit. Not suitable for multi-worker.
"""
from __future__ import annotations

import time
from collections import deque

from starlette.requests import Request
from starlette.responses import JSONResponse

# (max requests, window seconds)
GENERAL_LIMIT = (60, 60.0)   # every /api/* route
STRICT_LIMIT = (10, 60.0)    # expensive POST routes
STRICT_PATHS = frozenset({"/api/predict", "/api/portfolio", "/api/cards/refresh"})

# Absolute ceiling across ALL keys combined, independent of per-key buckets.
# Per-key limiting alone is bypassable by rotating the (spoofable) bucket key —
# each rotation gets a fresh per-key bucket — and at _MAX_TRACKED_KEYS the
# per-key dict itself gets cleared, wiping every legitimate user's counter.
# This global bucket can't be reset by key rotation, so it bounds total
# throughput even under a key-rotation stampede. Generous for this app's
# real traffic (a handful of users polling cards) but bounds abuse.
GLOBAL_LIMIT = (600, 60.0)

_MAX_TRACKED_KEYS = 10_000   # memory cap

_buckets: dict[tuple[str, str], deque] = {}
_global_bucket: deque = deque()
_last_prune: float = 0.0


def _client_ip(request: Request) -> str:
    # HF Spaces sits behind exactly one trusted reverse proxy, which APPENDS
    # the real client IP as the last hop of X-Forwarded-For. Leftmost entries
    # are fully client-controlled — a client can send its own XFF header, and
    # reading the leftmost entry let rotating a spoofed value yield a fresh
    # bucket per request (total per-key-limit bypass). Read the rightmost
    # entry instead: the hop the trusted proxy itself appended, which the
    # client cannot forge.
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "unknown"


def _prune(now: float) -> None:
    """Drop buckets idle > 2 windows; runs at most once per 60 s."""
    global _last_prune
    if now - _last_prune < 60.0:
        return
    _last_prune = now
    stale = [k for k, dq in _buckets.items() if not dq or now - dq[-1] > 120.0]
    for k in stale:
        del _buckets[k]


def check_rate_limit(request: Request) -> JSONResponse | None:
    """Return a 429 response if over budget, else None (allow)."""
    path = request.url.path
    if not path.startswith("/api/"):
        return None
    now = time.monotonic()
    _prune(now)

    # Global ceiling first — can't be bypassed by key rotation, and doesn't
    # depend on (or get reset by) the per-key bucket dict below.
    g_max, g_window = GLOBAL_LIMIT
    while _global_bucket and now - _global_bucket[0] > g_window:
        _global_bucket.popleft()
    if len(_global_bucket) >= g_max:
        retry_after = max(1, int(g_window - (now - _global_bucket[0])) + 1)
        return JSONResponse(
            status_code=429,
            content={"detail": "Server busy; slow down."},
            headers={"Retry-After": str(retry_after)},
        )

    tier = "strict" if path in STRICT_PATHS else "general"
    max_req, window = STRICT_LIMIT if tier == "strict" else GENERAL_LIMIT
    key = (_client_ip(request), tier)
    dq = _buckets.get(key)
    if dq is None:
        if len(_buckets) >= _MAX_TRACKED_KEYS:
            _buckets.clear()  # bounded memory beats perfect fairness here
        dq = _buckets[key] = deque()
    while dq and now - dq[0] > window:
        dq.popleft()
    if len(dq) >= max_req:
        retry_after = max(1, int(window - (now - dq[0])) + 1)
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded; slow down."},
            headers={"Retry-After": str(retry_after)},
        )
    dq.append(now)
    _global_bucket.append(now)
    return None
