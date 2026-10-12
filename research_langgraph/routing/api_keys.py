"""Multi-key failover for provider API keys.

`GROQ_API_KEY` / `GOOGLE_API_KEY` may hold several comma-separated keys. A call tries them
in turn and moves on ONLY when the key itself is the problem (rate-limited / exhausted,
revoked, forbidden) — never on errors that would fail identically on every key (bad model
id, malformed request, safety block). A key that just failed is put on a cooldown so the
next calls skip it instead of burning a request to rediscover it's exhausted.

Caveat: rate limits are per organisation (Groq) / per project (Gemini), so extra keys only
add capacity if they belong to separate orgs/projects — check the provider's terms before
using keys to widen a quota. They are always useful as a backup if one key is revoked.

Key values are never logged or put in exception text; only their position (`key#N`).
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from typing import Callable, List, Optional, Sequence, Tuple

RATE_LIMIT_COOLDOWN_S = float(os.environ.get("KEY_RATE_LIMIT_COOLDOWN_S", "60"))
DEAD_KEY_COOLDOWN_S = float(os.environ.get("KEY_DEAD_COOLDOWN_S", "3600"))
TRANSIENT_RETRIES = int(os.environ.get("PROVIDER_TRANSIENT_RETRIES", "3"))
TRANSIENT_BACKOFF_S = float(os.environ.get("PROVIDER_TRANSIENT_BACKOFF_S", "2"))


def parse_keys(*env_names: str) -> List[str]:
    """First non-empty env var among `env_names`, split on commas, blanks/dupes removed."""
    for name in env_names:
        raw = os.environ.get(name)
        if raw and raw.strip():
            seen: List[str] = []
            for k in raw.split(","):
                k = k.strip()
                if k and k not in seen:
                    seen.append(k)
            return seen
    return []


class KeyPool:
    """Round-robin start (spreads load) + per-key cooldown, thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rr = itertools.count()
        self._until: dict[str, float] = {}

    def ordered(self, keys: Sequence[str]) -> List[Tuple[int, str]]:
        """(index, key) pairs: healthy keys first (rotated), cooling keys last as a last resort."""
        now = time.monotonic()
        with self._lock:
            start = next(self._rr) % len(keys)
            rotated = [(i, keys[i]) for i in list(range(start, len(keys))) + list(range(start))]
            healthy = [p for p in rotated if self._until.get(p[1], 0.0) <= now]
            cooling = [p for p in rotated if self._until.get(p[1], 0.0) > now]
        return healthy + cooling

    def cool(self, key: str, seconds: float) -> None:
        with self._lock:
            self._until[key] = time.monotonic() + seconds


def call_with_failover(
    pool: KeyPool,
    keys: Sequence[str],
    attempt: Callable[[str], object],
    classify: Callable[[Exception], Optional[float]],
    provider: str,
    is_transient: Callable[[Exception], bool] = lambda e: False,
):
    """Run `attempt(key)` over the pool. `classify(exc)` returns a cooldown in seconds if the
    error is key-specific (=> try the next key) or None if it should propagate as-is."""
    from research_langgraph.routing.model_router import ProviderError  # avoid import cycle

    if not keys:
        raise ProviderError(f"{provider} API key variable contains no keys")
    failures: List[str] = []
    for idx, key in pool.ordered(keys):
        tries = 0
        while True:
            try:
                return attempt(key)
            except Exception as e:  # noqa: BLE001 - classified below
                cooldown = classify(e)
                if cooldown is not None:
                    pool.cool(key, cooldown)
                    failures.append(f"key#{idx}: {type(e).__name__}")
                    break  # this key is the problem -> next key
                if is_transient(e) and tries < TRANSIENT_RETRIES:
                    # 5xx / overloaded / timeout: the MODEL is the problem, not the key,
                    # so retry the same key after a short backoff instead of failing over.
                    time.sleep(TRANSIENT_BACKOFF_S * (2 ** tries))
                    tries += 1
                    continue
                raise ProviderError(f"{provider} {type(e).__name__}: {e}") from e
    raise ProviderError(f"all {len(keys)} {provider} key(s) exhausted/rejected ({'; '.join(failures)})")
