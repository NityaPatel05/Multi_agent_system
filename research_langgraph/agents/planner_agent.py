"""Planner agent — decomposes the topic into dimensions, generates queries per dimension,
and estimates complexity. Structured output validated against schemas.PlannerOutput.

`complexity` is read once by the Supervisor (main.py) to set the run's round/re-run
budget (architecture.md §3) — the Planner does not set its own budget. The graph also
gates on a human-approval interrupt right after this node (main.py), per CLAUDE.md's
plan-approval checkpoint feature.
"""

from __future__ import annotations

import logging

from research_langgraph.routing.model_router import call_structured, fallback_history_entries
from research_langgraph.schemas.validated_outputs import Complexity, PlannerOutput

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Planner agent in a multi-agent research system.
Decompose the given topic into the DIMENSIONS a complete answer needs (e.g. Definition,
Architecture, Applications, Limitations, Recent Research). For each dimension, generate
2-4 concrete search queries. Then estimate overall `complexity`:
- "simple": a narrow, well-established factual topic; one search round should suffice.
- "medium": most topics; some room for a targeted re-run.
- "complex": fast-moving, contested, or many-dimension topics likely to need multiple
  search rounds and re-runs.
Respond with dimensions, queries_by_dimension, complexity, and a one-sentence rationale.
"""


def _mock_plan(topic: str) -> PlannerOutput:
    dims = ["Definition", "Key Details", "Applications", "Limitations"]
    return PlannerOutput(
        dimensions=dims,
        queries_by_dimension={
            d: [f"{topic} {d.lower()}", f"{topic} {d.lower()} overview"] for d in dims
        },
        complexity=Complexity.medium,
        rationale="Offline mock plan: no model provider was configured.",
    )


def _episodic_hint(topic: str) -> str:
    """Episodic memory (our OWN past runs only - never live-web RAG): show the planner the
    dimensions earlier, similar runs used. Best-effort; any failure just means no hint."""
    try:
        from research_langgraph.memory.episodic_store import retrieve_similar_runs

        past = retrieve_similar_runs(topic, top_k=2)
    except Exception:
        logger.warning("episodic memory lookup failed; planning without it", exc_info=True)
        return ""
    lines = []
    for r in past:
        cov = (r.get("evidence_map") or {}).get("evidence_report", {}).get("coverage") or {}
        names = list(cov) if isinstance(cov, dict) else []
        if r.get("topic") and names:
            lines.append(f"- \"{str(r['topic'])[:120]}\": {', '.join(str(n)[:60] for n in names[:8])}")
    if not lines:
        return ""
    return ("\n\nFor reference, earlier runs on similar topics covered these dimensions "
            "(adapt, don't copy blindly):\n" + "\n".join(lines))


def planner_agent(state: dict) -> dict:
    topic = state["topic"]
    budget_mode = state.get("budget_mode", "balanced")

    hint = ""
    if state.get("is_broaden"):
        hint = (
            f"\n\nA previous search round returned too few credible sources "
            f"({state.get('search_broaden_hint', 'the current plan')}). Broaden or rephrase "
            f"the queries for the affected dimensions."
        )
    elif state.get("plan_revision_hint"):
        hint = f"\n\nThe user requested changes to the previous plan: {state['plan_revision_hint']}"

    if not state.get("is_broaden") and not state.get("plan_revision_hint"):
        hint += _episodic_hint(topic)

    parsed, result = call_structured(
        node_name="planner",
        system=SYSTEM_PROMPT,
        user=f"Topic: {topic}{hint}",
        schema=PlannerOutput,
        mock_fn=lambda: _mock_plan(topic),
        budget_mode=budget_mode,
    )
    return {
        "dimensions": parsed.dimensions,
        "queries_by_dimension": parsed.queries_by_dimension,
        "complexity": parsed.complexity.value,
        "plan_rationale": parsed.rationale,
        "status": "awaiting_approval" if not state.get("is_broaden") else "running",
        "audit_log": result.audit_entries(),
        "token_usage": [result.token_usage_entry()],
        "fallback_history": fallback_history_entries(result),
    }
