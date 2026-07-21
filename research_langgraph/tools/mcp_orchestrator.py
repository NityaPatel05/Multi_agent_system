"""Thin MCP tool orchestrator (CLAUDE.md §1, hard rule).

Deliberately dumb: auth passthrough + request logging only, NO routing logic. Which
provider to use within a capability (Exa vs. Tavily, PyMuPDF vs. raw-text) is decided by
the adapters themselves; which node calls what, when is decided by the LangGraph
Supervisor. This file exists only so agents have one call surface and every invocation
produces a uniform audit entry — it must never grow into a "god orchestrator" that holds
routing logic, state, or error resolution (architecture.md §1).

Deployment (architecture.md §7, "Dockerized MCP servers"): when `MCP_SEARCH_URL` /
`MCP_EXTRACT_URL` are set — docker-compose wires these to the standalone
`tools/mcp_search_server.py` / `mcp_extract_server.py` containers — dispatch goes over
HTTP, so a broken search integration can be redeployed independently of the rest of the
pipeline. Locally, with neither set, dispatch calls the adapters in-process (exactly the
prior behavior). Same self-healing philosophy as everywhere else in this project: if the
networked MCP server is unreachable, this falls back to running the adapter in-process
rather than failing the run.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from research_langgraph.tools import mcp_extract_adapter, mcp_search_adapter

_AUTH_ENV_VARS = ("EXA_API_KEY", "TAVILY_API_KEY")
_HTTP_TIMEOUT = float(os.environ.get("MCP_HTTP_TIMEOUT", "20"))


def auth_context() -> dict:
    """Auth passthrough: surface which provider credentials are present. No decisions
    made here about which to use — that's the adapter's job."""
    return {var: bool(os.environ.get(var)) for var in _AUTH_ENV_VARS}


def _post(url: str, payload: dict) -> dict:
    import httpx  # lazy import: only needed when a Dockerized MCP server is configured

    resp = httpx.post(url, json=payload, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def dispatch_search(node_name: str, queries: List[str]) -> Tuple[List[dict], List[dict]]:
    """Log + relay a search request — over HTTP to a Dockerized MCP search server if
    MCP_SEARCH_URL is set, otherwise in-process. Returns (sources, audit_log)."""
    search_url = os.environ.get("MCP_SEARCH_URL")
    recovery_entry: Optional[dict] = None

    if search_url:
        try:
            body = _post(f"{search_url.rstrip('/')}/search", {"queries": queries})
            sources, events = body["sources"], body["events"]
        except Exception as e:
            sources, events = mcp_search_adapter.search(queries)
            recovery_entry = {
                "node": node_name, "kind": "recovery", "name": "mcp_search_server",
                "provider": "in_process", "params": {"url": search_url}, "ok": True,
                "detail": f"remote MCP search server unreachable ({e}); ran adapter in-process",
            }
    else:
        sources, events = mcp_search_adapter.search(queries)

    audit = [{**e, "node": node_name} for e in events]
    if recovery_entry:
        audit.append(recovery_entry)
    return sources, audit


def dispatch_extract(node_name: str, source_id: str, url: str) -> Tuple[dict, List[dict]]:
    """Log + relay an extraction request — over HTTP to a Dockerized MCP extract server if
    MCP_EXTRACT_URL is set, otherwise in-process. Returns (doc, audit_log)."""
    extract_url = os.environ.get("MCP_EXTRACT_URL")
    recovery_entry: Optional[dict] = None

    if extract_url:
        try:
            body = _post(f"{extract_url.rstrip('/')}/extract", {"url": url})
            text, extractor = body["text"], body["extractor"]
            fell_back, failed, error = body["fell_back"], body["failed"], body["error"]
        except Exception as e:
            text, extractor, fell_back, failed, error = mcp_extract_adapter.extract(url)
            recovery_entry = {
                "node": node_name, "kind": "recovery", "name": "mcp_extract_server",
                "provider": "in_process", "params": {"url": extract_url}, "ok": True,
                "detail": f"remote MCP extract server unreachable ({e}); ran adapter in-process",
            }
    else:
        text, extractor, fell_back, failed, error = mcp_extract_adapter.extract(url)

    doc = {
        "source_id": source_id,
        "url": url,
        "text": text,
        "char_count": len(text),
        "extractor": extractor,
        "fell_back": fell_back,
        "failed": failed,
        "error": error,
    }
    if failed:
        detail = error
    elif extractor == "mock":
        detail = "primary and raw-text extraction both failed; used offline mock"
    elif fell_back:
        detail = "fell back to raw-text extraction"
    else:
        detail = ""
    entry = {
        "node": node_name,
        "kind": "recovery" if fell_back else "tool_call",
        "name": f"{extractor}_extract",
        "provider": extractor,
        "params": {"url": url},
        "ok": not failed,
        "detail": detail,
    }
    entries = [entry] + ([recovery_entry] if recovery_entry else [])
    return doc, entries
