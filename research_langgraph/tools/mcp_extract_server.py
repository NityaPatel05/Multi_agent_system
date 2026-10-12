"""Standalone HTTP wrapper around mcp_extract_adapter — the Dockerized MCP extract
server (architecture.md §7). Same "thin wrapper" principle as mcp_search_server.py: all
extraction/fallback logic lives in mcp_extract_adapter.py, not here.

Run standalone: `uvicorn research_langgraph.tools.mcp_extract_server:app --port 8002`
"""

from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from research_langgraph.tools import mcp_extract_adapter

app = FastAPI(title="MCP Extract Server")


def _require_service_token(x_service_token: str = Header(default="")) -> None:
    """Shared secret between the API and this tool service. Unset = open (local dev);
    docker-compose makes TOOL_SERVICE_TOKEN mandatory."""
    expected = os.environ.get("TOOL_SERVICE_TOKEN", "")
    if expected and not hmac.compare_digest(x_service_token.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid service token")


class ExtractRequest(BaseModel):
    url: str = Field(max_length=2048)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "adapter": mcp_extract_adapter.ADAPTER_NAME}


@app.post("/extract", dependencies=[Depends(_require_service_token)])
def extract(req: ExtractRequest) -> dict:
    text, extractor, fell_back, failed, error = mcp_extract_adapter.extract(req.url)
    return {
        "text": text,
        "extractor": extractor,
        "fell_back": fell_back,
        "failed": failed,
        "error": error,
    }
