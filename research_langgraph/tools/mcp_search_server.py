"""Standalone HTTP wrapper around mcp_search_adapter — the Dockerized MCP search server
(architecture.md §7). Exposes exactly the adapter's `search()` function over HTTP so it
can be deployed and redeployed independently of the rest of the pipeline. All actual
search/retry/fallback logic still lives in mcp_search_adapter.py — this file adds no
routing logic of its own, same "thin" principle as mcp_orchestrator.py.

Run standalone: `uvicorn research_langgraph.tools.mcp_search_server:app --port 8001`
"""

from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from research_langgraph.tools import mcp_search_adapter

app = FastAPI(title="MCP Search Server")


class SearchRequest(BaseModel):
    queries: List[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "adapter": mcp_search_adapter.ADAPTER_NAME}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    sources, events = mcp_search_adapter.search(req.queries)
    return {"sources": sources, "events": events}
