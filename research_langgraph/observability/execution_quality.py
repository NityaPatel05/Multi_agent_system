"""Execution-quality assessment (observability layer, extending architecture.md §6).

Distinguishes "the report was rejected on its merits" from "the report was never
genuinely evaluated because the pipeline itself fell back to mocks" — a three-state
signal (`FULL` / `PARTIAL` / `DEGRADED`) computed fresh on every Critic pass from the
current round's Synthesizer/Evidence Verifier outcome plus the run's accumulated
audit_log. Consumed by critic_agent.py (DEGRADED forces an automatic rejection — see its
module docstring) and main.py's finalize_node (surfaces the caveat in the final report).
"""

from __future__ import annotations

from typing import Dict, List

from research_langgraph.schemas.validated_outputs import ExecutionQuality, ExecutionQualityStatus

# Friendly labels for nodes whose mock usage only degrades quality to PARTIAL — anything
# NOT the Synthesizer or Evidence Verifier, since those two are handled precisely via the
# per-round `synthesizer_used_mock` / `evidence_verifier_used_mock` state flags below.
_PARTIAL_NODE_LABELS = {
    "planner": "Mock Planner",
    "search_agent": "Mock Search",
    "validator": "Mock Validator",
    "extractor_agent": "Mock Extraction",
    "critic": "Mock Critic",
}


def compute_execution_quality(state: Dict) -> Dict:
    """Read the current round's Synthesizer/Evidence Verifier outcome (last-write-wins
    flags, so a mock hiccup in an EARLIER round that later succeeded for real doesn't
    falsely flag the current result) plus any other mock usage anywhere in the run's
    accumulated audit_log, and classify FULL / PARTIAL / DEGRADED.
    """
    reasons: List[str] = []
    degraded = False

    if state.get("synthesizer_used_mock"):
        reasons.append("Mock Synthesizer")
        degraded = True
    if state.get("evidence_verifier_used_mock"):
        reasons.append("Mock Evidence Verifier")
        degraded = True

    other_mock_nodes = set()
    for entry in state.get("audit_log", []):
        if entry.get("provider") != "mock":
            continue
        node = entry.get("node", "")
        if node in ("synthesizer", "evidence_verifier"):
            continue  # already accounted for precisely above
        other_mock_nodes.add(node)

    for node in sorted(other_mock_nodes):
        reasons.append(_PARTIAL_NODE_LABELS.get(node, f"Mock {node.replace('_', ' ').title()}"))

    if degraded:
        status = ExecutionQualityStatus.degraded
    elif reasons:
        status = ExecutionQualityStatus.partial
    else:
        status = ExecutionQualityStatus.full

    return ExecutionQuality(status=status, reason=reasons).model_dump(mode="json")


def format_execution_quality_caveat(execution_quality: Dict) -> str:
    """Markdown caveat block for the final report, per execution_quality's status."""
    status = execution_quality.get("status", "FULL")
    reasons = execution_quality.get("reason", [])

    if status == "DEGRADED":
        reason_lines = "\n".join(f"- {r} fallback used" for r in reasons)
        return (
            "\n\n---\n**⚠ Report generated in degraded execution mode.**\n\n"
            f"Reason:\n{reason_lines}\n\n"
            "Coverage metrics may be optimistic.\n\n"
            "Retry recommended."
        )
    if status == "PARTIAL":
        return f"\n\n---\n*Note: some pipeline stages used fallback (mock) execution: {', '.join(reasons)}.*"
    return ""
