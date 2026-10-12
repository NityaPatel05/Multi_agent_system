"""API authentication + rate limiting.

`API_KEY` (one key, or several comma-separated) is required via `X-API-Key: <key>` or
`Authorization: Bearer <key>`. If `API_KEY` is unset the API stays open (local dev) and
logs a loud warning at startup; docker-compose makes it mandatory. Rate limiting is an
in-process sliding window per client (API key if present, else IP) on run-starting routes,
since every run spends paid LLM/search quota ("denial of wallet").
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "10"))


def _configured_keys() -> list[str]:
    return [k.strip() for k in os.environ.get("API_KEY", "").split(",") if k.strip()]


def auth_enabled() -> bool:
    return bool(_configured_keys())


def warn_if_open() -> None:
    if not auth_enabled():
        logger.warning("API_KEY is not set: the API is UNAUTHENTICATED. Set API_KEY before "
                       "exposing this service beyond localhost.")


def _presented_key(request: Request) -> str:
    key = request.headers.get("x-api-key")
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth[7:].strip()
    return key or ""


def require_api_key(request: Request) -> None:
    keys = _configured_keys()
    if not keys:
        return
    presented = _presented_key(request)
    if not any(hmac.compare_digest(presented.encode(), k.encode()) for k in keys):
        raise HTTPException(status_code=401, detail="invalid or missing API key",
                            headers={"WWW-Authenticate": "Bearer"})


_hits: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def rate_limit(request: Request) -> None:
    client = _presented_key(request) or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    with _lock:
        q = _hits[client]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            retry = max(1, int(60 - (now - q[0])))
            raise HTTPException(status_code=429, detail="rate limit exceeded",
                                headers={"Retry-After": str(retry)})
        q.append(now)


def reset_rate_limits() -> None:  # for tests
    with _lock:
        _hits.clear()
