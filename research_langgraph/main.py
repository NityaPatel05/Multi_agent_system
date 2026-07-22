"""state.py + main.py: the core orchestration graph (CLAUDE.md build order step 1).

Wires the Supervisor -> Planner (+ human-approval interrupt) -> parallel Search/Extract
Send() fan-out with explicit Merge nodes -> Synthesizer -> Evidence Verifier -> Critic ->
targeted re-run loop, exactly as diagrammed in docs/architecture.md §3. The Supervisor's
round/re-run budget is derived once from the Planner's `complexity` estimate (the table in
architecture.md §3) and enforced by conditional edges, not left to open-ended model
judgment (hard rule).
"""

from __future__ import annotations

import uuid
from typing import List, Union

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from research_langgraph.agents.critic_agent import critic_agent
from research_langgraph.agents.evidence_verifier_agent import evidence_verifier_agent
from research_langgraph.agents.extraction_merge import extraction_merge
from research_langgraph.agents.extractor_agent import extractor_agent
from research_langgraph.agents.planner_agent import planner_agent
from research_langgraph.agents.search_agent import search_agent
from research_langgraph.agents.search_merge import search_merge
from research_langgraph.agents.synthesizer_agent import synthesizer_agent
from research_langgraph.agents.validator_agent import validator_agent
from research_langgraph.memory.checkpointer import get_checkpointer
from research_langgraph.observability.execution_quality import format_execution_quality_caveat
from research_langgraph.state import ResearchState

# --------------------------------------------------------------------------- #
# Supervisor budget table (architecture.md §3) — derived ONCE from Planner's
# `complexity` estimate, not left to open-ended model judgment (hard rule).
# --------------------------------------------------------------------------- #
_ROUND_BUDGET = {
    "simple": {"search_rounds_allowed": 1, "rerun_budget": 0},
    "medium": {"search_rounds_allowed": 2, "rerun_budget": 1},
    "complex": {"search_rounds_allowed": 2, "rerun_budget": 3},
}


# --------------------------------------------------------------------------- #
# Extra nodes not big enough to warrant their own agents/ file
# --------------------------------------------------------------------------- #
def human_approval_node(state: dict) -> dict:
    """Plan-approval checkpoint (CLAUDE.md's #3 differentiating feature): a single
    LangGraph `interrupt()` call pauses the run until a human approves or requests
    changes to the Planner's coverage map before any search spend happens."""
    decision = interrupt({
        "type": "plan_approval",
        "dimensions": state.get("dimensions", []),
        "queries_by_dimension": state.get("queries_by_dimension", {}),
        "complexity": state.get("complexity", "medium"),
        "rationale": state.get("plan_rationale", ""),
    })
    if isinstance(decision, dict):
        approved = bool(decision.get("approved", True))
        revision_hint = decision.get("feedback", "")
    else:
        approved = bool(decision)
        revision_hint = ""
    return {"plan_approved": approved, "plan_revision_hint": revision_hint, "status": "running"}


def supervisor_init_budget_node(state: dict) -> dict:
    """Supervisor reads the Planner's complexity once and sets the round/re-run budget
    for the whole run (architecture.md §3 table)."""
    complexity = state.get("complexity", "medium")
    budget = _ROUND_BUDGET.get(complexity, _ROUND_BUDGET["medium"])
    return {
        "search_rounds_allowed": budget["search_rounds_allowed"],
        "search_rounds_used": 1,  # the dispatch that follows this node IS round 1
        "rerun_budget": budget["rerun_budget"],
        "rerun_used": 0,
        "in_targeted_rerun": False,
        "is_broaden": False,
        "status": "running",
    }


def consume_search_round_node(state: dict) -> dict:
    """`Validator -> Planner` broaden loop: fewer than 3 credible sources cleared the
    bar, and the round budget isn't exhausted, so spend one more search round."""
    return {
        "search_rounds_used": state.get("search_rounds_used", 0) + 1,
        "is_broaden": True,
        "search_broaden_hint": f"only {state.get('credible_count', 0)} credible source(s) found",
    }


def prepare_rerun_node(state: dict) -> dict:
    """Turns the Critic's directives into a narrowed, TARGETED search dispatch (hard
    rule: never a blanket restart). Coverage gaps re-search just that dimension;
    contradictions and poor-diversity flags become one-off pseudo-dimension searches;
    uncited claims need no new search at all — just a Synthesizer revision pass.

    A `retry_degraded` directive (execution_quality was DEGRADED) is handled separately:
    the underlying extracted documents are still valid — only the LLM judgment layer was
    compromised — so this skips search entirely and goes straight back to the Synthesizer
    for a fresh attempt, hopefully against a now-recovered provider.
    """
    directives = state.get("rerun_directives", [])

    if any(d.get("action") == "retry_degraded" for d in directives):
        return {
            "rerun_used": state.get("rerun_used", 0) + 1,
            "active_queries_by_dimension": {},
            "rerun_search_needed": False,
            "recite_targets": [],
            "in_targeted_rerun": True,
        }

    qbd = state.get("queries_by_dimension", {})
    topic = state.get("topic", "")

    active: dict[str, list[str]] = {}
    for dim in state.get("rerun_dimensions", []):
        active[dim] = qbd.get(dim, [f"{topic} {dim}"])
    for subtopic in state.get("rerun_subtopics", []):
        active[f"resolve:{subtopic}"] = [subtopic]
    for d in directives:
        if d.get("action") == "improve_diversity":
            target = d.get("target", "")
            active[f"diversity:{target}"] = [f"{topic} {target}"]

    recite_targets = [d.get("target", "") for d in directives if d.get("action") == "recite"]

    return {
        "rerun_used": state.get("rerun_used", 0) + 1,
        "active_queries_by_dimension": active,
        "rerun_search_needed": bool(active),
        "recite_targets": recite_targets,
        "in_targeted_rerun": True,
    }


def finalize_node(state: dict) -> dict:
    """Assemble the final markdown report + evidence map (CLAUDE.md's #2 differentiating
    feature: structured claim -> evidence span -> source -> confidence JSON alongside the
    report). Surfaces execution_quality prominently — a DEGRADED report (mock content
    and/or mock judgment) gets a loud warning, not a quiet caveat, since its coverage
    metrics cannot be trusted (see critic_agent.py)."""
    execution_quality = state.get("execution_quality", {"status": "FULL", "reason": []})
    budget_exhausted = not state.get("approved") and state.get("rerun_used", 0) >= state.get(
        "rerun_budget", 0
    )

    caveats = [format_execution_quality_caveat(execution_quality)]
    if budget_exhausted:
        caveats.append(
            f"\n\n---\n*Note: re-run budget ({state.get('rerun_budget')}) was exhausted before "
            f"all Critic-flagged issues were resolved. Remaining issues: "
            f"{state.get('critic_summary', '')}*"
        )

    return {
        "final_report": state.get("report_markdown", "") + "".join(caveats),
        "evidence_map": {
            "claims": state.get("claims", []),
            "evidence_report": state.get("evidence_report", {}),
            "sources": state.get("sources", []),
            "execution_quality": execution_quality,
            "fallback_history": state.get("fallback_history", []),
            "rejection_reasons": state.get("rejection_reasons", {}),
        },
        "execution_quality": execution_quality,
        "status": "done",
    }


# --------------------------------------------------------------------------- #
# Send()-fan-out route functions (architecture.md §3 route_to_search example)
# --------------------------------------------------------------------------- #
def route_to_search(state: dict) -> List[Send]:
    return [
        Send("search_agent", {"dimension": d, "queries": qs})
        for d, qs in state.get("queries_by_dimension", {}).items()
    ]


def route_to_targeted_search(state: dict) -> List[Send]:
    return [
        Send("search_agent", {"dimension": d, "queries": qs})
        for d, qs in state.get("active_queries_by_dimension", {}).items()
    ]


def route_to_extract(state: dict) -> Union[str, List[Send]]:
    already = {d["source_id"] for d in state.get("extracted_docs", [])}
    to_extract = [s for s in state.get("validated_sources", []) if s["id"] not in already]
    if not to_extract:
        return "extraction_merge"
    return [
        Send("extractor_agent", {"source_id": s["id"], "url": s["url"], "dimension": s.get("dimension", "")})
        for s in to_extract
    ]


# --------------------------------------------------------------------------- #
# Conditional-edge decision functions
# --------------------------------------------------------------------------- #
def after_planner(state: dict) -> Union[str, List[Send]]:
    if state.get("is_broaden"):
        return route_to_search(state)  # skip re-approval; budget already set
    return "human_approval"


def after_human_approval(state: dict) -> str:
    return "supervisor_init_budget" if state.get("plan_approved") else "planner"


def after_validator(state: dict) -> Union[str, List[Send]]:
    if (
        state.get("needs_broader_search")
        and not state.get("in_targeted_rerun")
        and state.get("search_rounds_used", 0) < state.get("search_rounds_allowed", 1)
    ):
        return "consume_search_round"
    return route_to_extract(state)


def after_critic(state: dict) -> str:
    if state.get("approved"):
        return "finalize"
    if state.get("rerun_used", 0) >= state.get("rerun_budget", 0):
        return "finalize"
    return "prepare_rerun"


def after_prepare_rerun(state: dict) -> Union[str, List[Send]]:
    if state.get("rerun_search_needed"):
        return route_to_targeted_search(state)
    return "synthesizer"


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def build_graph(checkpointer=None):
    builder = StateGraph(ResearchState)

    builder.add_node("planner", planner_agent)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("supervisor_init_budget", supervisor_init_budget_node)
    builder.add_node("consume_search_round", consume_search_round_node)
    builder.add_node("search_agent", search_agent)
    builder.add_node("search_merge", search_merge)
    builder.add_node("validator", validator_agent)
    builder.add_node("extractor_agent", extractor_agent)
    builder.add_node("extraction_merge", extraction_merge)
    builder.add_node("synthesizer", synthesizer_agent)
    builder.add_node("evidence_verifier", evidence_verifier_agent)
    builder.add_node("critic", critic_agent)
    builder.add_node("prepare_rerun", prepare_rerun_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", after_planner, ["human_approval", "search_agent"])
    builder.add_conditional_edges(
        "human_approval", after_human_approval, ["supervisor_init_budget", "planner"]
    )
    builder.add_conditional_edges("supervisor_init_budget", route_to_search, ["search_agent"])
    builder.add_edge("search_agent", "search_merge")
    builder.add_edge("search_merge", "validator")
    builder.add_conditional_edges(
        "validator", after_validator, ["consume_search_round", "extractor_agent", "extraction_merge"]
    )
    builder.add_edge("consume_search_round", "planner")
    builder.add_edge("extractor_agent", "extraction_merge")
    builder.add_edge("extraction_merge", "synthesizer")
    builder.add_edge("synthesizer", "evidence_verifier")
    builder.add_edge("evidence_verifier", "critic")
    builder.add_conditional_edges("critic", after_critic, ["finalize", "prepare_rerun"])
    builder.add_conditional_edges(
        "prepare_rerun", after_prepare_rerun, ["search_agent", "synthesizer"]
    )
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


def mark_run_failed(graph, config: dict) -> None:
    """Best-effort: persist status='failed' into the checkpoint so a caller polling
    get_state() later (e.g. a FastAPI client hitting GET /research/{run_id}) sees a
    terminal status instead of a run stuck at 'running' forever. A crashed node leaves
    the checkpoint at whatever state existed before it — nothing marks the run itself
    failed unless a caller explicitly does so, which is what this closes. Never raises:
    a failure here (e.g. no checkpoint ever got written before the crash) must not mask
    the original exception the caller is already handling.
    """
    try:
        graph.update_state(config, {"status": "failed"})
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Convenience runner (used by the smoke test and the CLI entrypoint below)
# --------------------------------------------------------------------------- #
def run_research(
    topic: str,
    thread_id: str | None = None,
    auto_approve: bool = True,
    budget_mode: str = "balanced",
) -> dict:
    checkpointer = get_checkpointer()
    graph = build_graph(checkpointer=checkpointer)
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(
            {"topic": topic, "run_id": thread_id, "budget_mode": budget_mode},
            config=config,
        )
        if auto_approve and result.get("__interrupt__"):
            result = graph.invoke(Command(resume={"approved": True}), config=config)
    except Exception:
        mark_run_failed(graph, config)
        raise
    return result


if __name__ == "__main__":
    import json
    import sys

    topic_arg = " ".join(sys.argv[1:]) or "small language models"
    final_state = run_research(topic_arg)
    print(final_state.get("final_report", "(no report produced)"))
    print("\n--- evidence_map ---")
    print(json.dumps(final_state.get("evidence_map", {}), indent=2, default=str))
