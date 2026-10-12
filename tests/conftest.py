import os

import pytest

_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "EXA_API_KEY", "TAVILY_API_KEY",
         "DATABASE_URL", "QDRANT_URL", "MCP_SEARCH_URL", "MCP_EXTRACT_URL", "LANGSMITH_API_KEY"]


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    """Importing the package no longer loads .env, but clear anything inherited from the
    shell so every test runs on the deterministic offline mocks."""
    for k in _KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EPISODIC_STORE_PATH", os.devnull)
