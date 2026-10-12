"""Standalone HTTP wrapper around mcp_search_adapter — the Dockerized MCP search server
(architecture.md §7). Exposes exactly the adapter's `search()` function over HTTP so it
can be deployed and redeployed independently of the rest of the pipeline. All actual
search/retry/fallback logic still lives in mcp_search_adapter.py — this file adds no
routing logic of its own, same "thin" principle as mcp_orchestrator.py.

Run standalone: `uvicorn research_langgraph.tools.mcp_search_server:app --port 8001`
"""

from __future__ import annotations

from typing import List

import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from research_langgraph.tools import mcp_search_adapter

app = FastAPI(title="MCP Search Server")


def _require_service_token(x_service_token: str = Header(default="")) -> None:
    """Shared secret between the API and this tool service. Unset = open (local dev);
    docker-compose makes TOOL_SERVICE_TOKEN mandatory."""
    expected = os.environ.get("TOOL_SERVICE_TOKEN", "")
    if expected and not hmac.compare_digest(x_service_token.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid service token")


class SearchRequest(BaseModel):
    queries: List[str] = Field(max_length=50)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "adapter": mcp_search_adapter.ADAPTER_NAME}


@app.post("/search", dependencies=[Depends(_require_service_token)])
def search(req: SearchRequest) -> dict:
    sources, events = mcp_search_adapter.search(req.queries)
    return {"sources": sources, "events": events}
