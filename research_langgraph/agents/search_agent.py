"""Search agent — plans/dispatches queries only.

Hard rule: this agent NEVER calls Exa/Tavily directly. The actual HTTP call happens in
mcp_search_adapter, reached only through mcp_orchestrator. This node is invoked once per
dimension via LangGraph's Send() fan-out (architecture.md §3), so multiple invocations
run concurrently — each is stateless and independent.
"""

from __future__ import annotations

from research_langgraph.tools import mcp_orchestrator


def search_agent(state: dict) -> dict:
    dimension = state["dimension"]
    queries = state["queries"]
    sources, audit = mcp_orchestrator.dispatch_search("search_agent", queries)
    for s in sources:
        s["dimension"] = dimension
    fallback_history = [
        {"node": "Search Agent", "reason": e["detail"], "fallback": "Mock"}
        for e in audit
        if e.get("kind") == "recovery" and e.get("provider") == "mock"
    ]
    return {"raw_search_results": sources, "audit_log": audit, "fallback_history": fallback_history}
