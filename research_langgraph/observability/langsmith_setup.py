"""LangSmith tracing setup (observability layer, architecture.md §6).

Gives node-level traces (inputs/outputs, latency, retries) natively once
LANGSMITH_API_KEY is set — no separate OpenTelemetry collector needed given LangGraph's
built-in integration, which is the simpler choice at this project's size.
"""

from __future__ import annotations

import os


def configure_langsmith(project: str = "research-langgraph") -> bool:
    """Turn on LangSmith tracing if an API key is configured. Returns whether tracing is
    active, so callers (main.py / api/main.py) can log observability status at startup
    instead of silently no-oping when the key is absent."""
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    os.environ["LANGCHAIN_API_KEY"] = api_key
    return True
