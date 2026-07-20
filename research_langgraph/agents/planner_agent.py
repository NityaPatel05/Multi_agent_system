"""Planner agent — decomposes the topic into dimensions, generates queries per dimension,
and estimates complexity. Structured output validated against schemas.PlannerOutput.

`complexity` is read once by the Supervisor (main.py) to set the run's round/re-run
budget (architecture.md §3) — the Planner does not set its own budget. The graph also
gates on a human-approval interrupt right after this node (main.py), per CLAUDE.md's
plan-approval checkpoint feature.
"""

from __future__ import annotations

from research_langgraph.routing.model_router import call_structured, fallback_history_entries
from research_langgraph.schemas.validated_outputs import Complexity, PlannerOutput

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
