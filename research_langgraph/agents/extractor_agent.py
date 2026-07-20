"""Extractor agent — dispatches document extraction for ONE URL per invocation.

Hard rule: never calls PyMuPDF/BeautifulSoup directly — the actual parsing happens in
mcp_extract_adapter, reached only through mcp_orchestrator (same separation as Search).
Invoked once per validated URL via LangGraph's Send() fan-out (architecture.md §3).
"""

from __future__ import annotations

from research_langgraph.tools import mcp_orchestrator


def extractor_agent(state: dict) -> dict:
    source_id = state["source_id"]
    url = state["url"]
    dimension = state.get("dimension", "")
    doc, audit = mcp_orchestrator.dispatch_extract("extractor_agent", source_id, url)
    doc["dimension"] = dimension
    # Only a full mock counts as a fallback worth flagging — a "raw" fallback is still
    # real fetched content, just without proper parsing (see extraction_fallbacks).
    fallback_history = [
        {"node": "Extractor Agent", "reason": e["detail"] or "extraction fell back to mock", "fallback": "Mock"}
        for e in audit
        if e.get("provider") == "mock"
    ]
    return {"extracted_docs": [doc], "audit_log": audit, "fallback_history": fallback_history}
