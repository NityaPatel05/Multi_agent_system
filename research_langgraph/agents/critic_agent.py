"""Critic agent — reads the Evidence Verifier's single output object and either approves
the report or returns TARGETED re-run directives.

Hard rule: never a blanket restart — every directive names a specific dimension,
sub-topic, source-type instruction, or claim to fix. The Supervisor's remaining re-run
budget (set from the Planner's complexity estimate) is enforced by main.py's conditional
edge, not here; this agent only says what's wrong and how narrowly to fix it.

Execution-quality gate: before doing any content judgment, the Critic checks
`execution_quality` (computed fresh from the current round's Synthesizer/Evidence
Verifier outcome — see observability/execution_quality.py). If it's DEGRADED — meaning
the report's content and/or the judgment of that content came from a mock fallback, not a
real model — the Critic short-circuits to an automatic rejection with NO LLM call: a mock
verdict on mock content proves nothing, so there's nothing to discuss. This is also why
`execution_status` is a richer, separate signal from the plain `approved` bool: a
DEGRADED rejection means "never genuinely evaluated," not "evaluated and found wanting."
"""

from __future__ import annotations

from research_langgraph.observability.execution_quality import compute_execution_quality
from research_langgraph.routing.model_router import call_structured, fallback_history_entries
from research_langgraph.schemas.validated_outputs import (
    CriticOutput,
    ExecutionStatus,
    ReRunAction,
    ReRunDirective,
)

SYSTEM_PROMPT = """You are the Critic. You read the Evidence Verifier's report (coverage,
contradictions, source_diversity, citation_completeness) and the draft. Approve the
report ONLY if coverage is strong across all dimensions, there are no unresolved
contradictions, source diversity is acceptable, and citation completeness is high.
Otherwise, return a list of directives, each one TARGETED at a specific weakness:
- coverage gap in a dimension -> action=research_dimension, target=<dimension>
- unresolved contradiction -> action=resolve_contradiction, target=<topic>
- poor diversity -> action=improve_diversity, target=<source-type instruction>
- uncited claim -> action=recite, target=<claim text>
NEVER recommend redoing the whole report — only the specific weak parts.
"""

_COVERAGE_OK = 0.7
_DIVERSITY_OK_HINT = "acceptable"
_DEGRADED_REASON = "Verification performed using fallback models."


def _mock_critic(evidence_report: dict) -> CriticOutput:
    directives = []
    for dim, score in evidence_report.get("coverage", {}).items():
        if score < _COVERAGE_OK:
            directives.append(ReRunDirective(
                action=ReRunAction.research_dimension, target=dim,
                reason=f"coverage {score} below threshold",
            ))
    for c in evidence_report.get("contradictions", []):
        if c.get("status") != "Resolved":
            directives.append(ReRunDirective(
                action=ReRunAction.resolve_contradiction, target=c.get("topic", ""),
                reason=c.get("reason", "unresolved contradiction"),
            ))
    if _DIVERSITY_OK_HINT not in evidence_report.get("diversity_flag", _DIVERSITY_OK_HINT):
        directives.append(ReRunDirective(
            action=ReRunAction.improve_diversity,
            target="prioritize Academic/Government sources",
            reason=evidence_report.get("diversity_flag", ""),
        ))
    for claim in evidence_report.get("citation_completeness", {}).get("uncited_claims", []):
        directives.append(ReRunDirective(action=ReRunAction.recite, target=claim, reason="uncited claim"))

    approved = not directives
    return CriticOutput(
        approved=approved,
        directives=directives,
        summary="Approved." if approved else f"{len(directives)} targeted issue(s) found.",
    )


def critic_agent(state: dict) -> dict:
    execution_quality = compute_execution_quality(state)

    if execution_quality["status"] == "DEGRADED":
        directive = ReRunDirective(action=ReRunAction.retry_degraded, target="", reason=_DEGRADED_REASON)
        return {
            "approved": False,
            "execution_status": ExecutionStatus.degraded.value,
            "retry_recommended": True,
            "execution_quality": execution_quality,
            "rerun_directives": [directive.model_dump(mode="json")],
            "critic_summary": _DEGRADED_REASON,
            "rerun_dimensions": [],
            "rerun_subtopics": [],
            "rerun_recite": False,
        }

    evidence_report = state.get("evidence_report", {})
    budget_mode = state.get("budget_mode", "balanced")
    parsed, result = call_structured(
        node_name="critic",
        system=SYSTEM_PROMPT,
        user=f"Evidence report: {evidence_report}\nDraft: {state.get('report_markdown', '')}",
        schema=CriticOutput,
        mock_fn=lambda: _mock_critic(evidence_report),
        budget_mode=budget_mode,
    )
    # mode="json" — action is a ReRunAction (str) Enum; without this, LangGraph's
    # checkpointer serializer stores a raw enum instance instead of a plain string, which
    # it now warns is deprecated and "will be blocked in a future version."
    directive_dicts = [d.model_dump(mode="json") for d in parsed.directives]
    return {
        "approved": parsed.approved,
        "execution_status": (ExecutionStatus.approved if parsed.approved else ExecutionStatus.rejected).value,
        "retry_recommended": not parsed.approved,
        "execution_quality": execution_quality,
        "rerun_directives": directive_dicts,
        "critic_summary": parsed.summary,
        "rerun_dimensions": [d["target"] for d in directive_dicts if d["action"] == "research_dimension"],
        "rerun_subtopics": [d["target"] for d in directive_dicts if d["action"] == "resolve_contradiction"],
        "rerun_recite": any(d["action"] == "recite" for d in directive_dicts),
        "audit_log": result.audit_entries(),
        "token_usage": [result.token_usage_entry()],
        "fallback_history": fallback_history_entries(result),
    }
