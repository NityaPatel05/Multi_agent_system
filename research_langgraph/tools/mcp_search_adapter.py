"""MCP search adapter — Domain-Specific Adapter for search (CLAUDE.md §1).

Owns the actual HTTP calls to search providers (Exa primary, Tavily fallback) plus
retries, rate-limit handling, and response normalization. The Search *agent* never
reaches this module directly — it goes through `mcp_orchestrator` (hard rule); this
keeps retries/fallback/rate-limit handling in exactly one place instead of duplicated
across every agent that happens to search.

Self-healing fallback chain (guardrails layer): `exa_search` failing/timing out triggers
an automatic retry against `tavily_search`; the failover is logged as a "recovery" event,
not surfaced as an error. When neither provider is configured, falls back to a
deterministic offline mock so the graph stays runnable without API keys.
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Tuple

ADAPTER_NAME = "mcp_search_adapter"


def _source_id(url: str) -> str:
    return "src_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def _normalize(raw_hits: List[dict], provider: str) -> List[dict]:
    normalized = []
    for h in raw_hits:
        url = h.get("url") or h.get("link") or ""
        if not url:
            continue
        normalized.append({
            "id": _source_id(url),
            "url": url,
            "title": h.get("title", ""),
            "snippet": (h.get("snippet") or h.get("text") or h.get("content") or "")[:300],
            "provider": provider,
            "score": float(h.get("score") or 0.0),
        })
    return normalized


def _exa_search(query: str) -> List[dict]:
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY not set")
    from exa_py import Exa  # lazy import: optional dependency

    client = Exa(api_key=api_key)
    resp = client.search_and_contents(query, num_results=5, text=True)
    return [
        {"url": r.url, "title": r.title, "snippet": (r.text or "")[:300], "score": r.score}
        for r in resp.results
    ]


def _tavily_search(query: str) -> List[dict]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    from tavily import TavilyClient  # lazy import: optional dependency

    client = TavilyClient(api_key=api_key)
    resp = client.search(query, max_results=5)
    return [
        {"url": r["url"], "title": r.get("title", ""), "snippet": r.get("content", "")[:300],
         "score": r.get("score", 0.0)}
        for r in resp.get("results", [])
    ]


def _mock_search(query: str) -> List[dict]:
    seed = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    return [
        {
            "url": f"https://example.org/{seed}/{i}",
            "title": f"Mock result {i} for: {query}",
            "snippet": f"Deterministic offline placeholder result #{i} for query '{query}'.",
            "score": round(1.0 - i * 0.15, 2),
        }
        for i in range(1, 4)
    ]


def search(queries: List[str]) -> Tuple[List[dict], List[dict]]:
    """Run each query through Exa (primary), auto-failing-over to Tavily, then to a
    deterministic mock if neither is configured. Returns (normalized_sources, events);
    events are audit-shaped dicts without a `node` field — the orchestrator stamps that
    on, since only it knows which agent dispatched the call."""
    results: List[dict] = []
    events: List[dict] = []
    for query in queries:
        provider = "exa"
        hits: List[dict] = []
        exa_err = tavily_err = None
        try:
            hits = _exa_search(query)
        except Exception as e:
            exa_err = e
            try:
                hits = _tavily_search(query)
                provider = "tavily"
                events.append({
                    "kind": "recovery", "name": "exa_search", "provider": "tavily",
                    "params": {"query": query}, "ok": True,
                    "detail": f"exa_search failed ({exa_err}); failed over to tavily_search",
                })
            except Exception as e2:
                tavily_err = e2
                hits = _mock_search(query)
                provider = "mock"
                events.append({
                    "kind": "recovery", "name": "tavily_search", "provider": "mock",
                    "params": {"query": query}, "ok": True,
                    "detail": f"exa_search and tavily_search both unavailable "
                              f"({exa_err} / {tavily_err}); used offline mock",
                })
        events.append({
            "kind": "tool_call", "name": f"{provider}_search", "provider": provider,
            "params": {"query": query}, "ok": True, "detail": "",
        })
        results.extend(_normalize(hits, provider))
    return results, events
