"""Standalone HTTP wrapper around mcp_extract_adapter — the Dockerized MCP extract
server (architecture.md §7). Same "thin wrapper" principle as mcp_search_server.py: all
extraction/fallback logic lives in mcp_extract_adapter.py, not here.

Run standalone: `uvicorn research_langgraph.tools.mcp_extract_server:app --port 8002`
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from research_langgraph.tools import mcp_extract_adapter

app = FastAPI(title="MCP Extract Server")


class ExtractRequest(BaseModel):
    url: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "adapter": mcp_extract_adapter.ADAPTER_NAME}


@app.post("/extract")
def extract(req: ExtractRequest) -> dict:
    text, extractor, fell_back, failed, error = mcp_extract_adapter.extract(req.url)
    return {
        "text": text,
        "extractor": extractor,
        "fell_back": fell_back,
        "failed": failed,
        "error": error,
    }
