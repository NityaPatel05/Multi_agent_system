"""Checkpointer (guardrails layer, CLAUDE.md hard rule: PostgresSaver).

A crashed run resumes from the last completed node instead of restarting. Postgres is the
production checkpointer; when `DATABASE_URL` isn't set (local/dev/smoke-testing without a
Postgres instance running) this falls back to LangGraph's in-memory saver so the graph
stays runnable — same graceful-degradation pattern used throughout tools/ and routing/.
"""

from __future__ import annotations

import os
from typing import Optional


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

        pool = ConnectionPool(conninfo=db_url, max_size=10, kwargs={"autocommit": True})
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        return checkpointer

    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()
