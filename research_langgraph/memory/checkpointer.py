"""Checkpointer (guardrails layer, CLAUDE.md hard rule: PostgresSaver).

A crashed run resumes from the last completed node instead of restarting. Postgres is the
production checkpointer; when `DATABASE_URL` isn't set (local/dev/smoke-testing without a
Postgres instance running) this falls back to LangGraph's in-memory saver so the graph
stays runnable — same graceful-degradation pattern used throughout tools/ and routing/.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, Optional, Tuple

# One pool per DATABASE_URL, shared by every run_research()/API call in the process
# (previously each call opened a fresh pool of up to 10 connections and never closed it).
_lock = threading.Lock()
_savers: Dict[str, Tuple[object, object]] = {}


def close_checkpointers() -> None:
    """Close all cached Postgres pools (call from app shutdown / end of an eval run)."""
    with _lock:
        for pool, _ in _savers.values():
            pool.close()
        _savers.clear()


def get_checkpointer(database_url: Optional[str] = None):
    """Return a LangGraph checkpointer.

    Postgres-backed if a connection string is available (creates the required tables on
    first use via `.setup()`); otherwise an in-memory saver, which is fine for local runs
    and the smoke test but does NOT survive a process restart.
    """
    db_url = database_url or os.environ.get("DATABASE_URL")
    if db_url:
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg_pool import ConnectionPool

        from psycopg.rows import dict_row

        with _lock:
            cached = _savers.get(db_url)
            if cached is not None:
                return cached[1]
            pool = ConnectionPool(
                conninfo=db_url, max_size=10, open=True,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            )
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            _savers[db_url] = (pool, checkpointer)
            return checkpointer

    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()
