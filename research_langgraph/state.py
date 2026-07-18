"""ResearchState — the graph's working memory (layer 5 in CLAUDE.md).

One TypedDict, threaded through every node. Fields written by parallel Send() branches
(raw_search_results, extracted_docs, audit_log, token_usage) use `operator.add` reducers
so LangGraph concatenates branch outputs instead of clobbering them; everything else is
last-write-wins (a single node owns it).
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


def _merge_dicts(a: Optional[Dict[str, str]], b: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Reducer for dict-valued state fields (operator.add doesn't support dict + dict).
    Later rounds win on a key collision, but nothing from an earlier round is dropped —
    used for rejection_reasons so a source rejected in an early broaden round is still
    explained even if a later round never re-touches that id."""
    merged = dict(a or {})
    merged.update(b or {})
    return merged


class AuditEntry(TypedDict, total=False):
    """One per-tool-call / per-model-call log line (guardrails layer §5)."""

    node: str
    kind: str  # "tool_call" | "model_call" | "recovery"
    name: str  # tool name or model tier
    provider: str
    params: Dict[str, Any]
    ok: bool
    detail: str


class TokenUsage(TypedDict, total=False):
    node: str
    tier: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class FallbackEvent(TypedDict, total=False):
    """One self-healing event — see schemas.validated_outputs.FallbackEvent."""

    node: str
    reason: str
    fallback: str


class ExecutionQuality(TypedDict, total=False):
    """See schemas.validated_outputs.ExecutionQuality / ExecutionQualityStatus."""

    status: str  # "FULL" | "PARTIAL" | "DEGRADED"
    reason: List[str]


class ResearchState(TypedDict, total=False):
    # ---- input ----
    topic: str
    run_id: str
    budget_mode: str  # "fast" | "balanced" | "thorough"

    # ---- Planner output ----
    dimensions: List[str]
    queries_by_dimension: Dict[str, List[str]]
    complexity: str  # simple | medium | complex
    plan_rationale: str
    plan_approved: Optional[bool]  # set by the human-approval interrupt
    is_broaden: bool  # True when re-entering Planner from the Validator broaden loop
    search_broaden_hint: str
    plan_revision_hint: str  # set when the human rejects the plan at the approval gate

    # ---- Supervisor-set budget (derived once from complexity) ----
    search_rounds_allowed: int
    search_rounds_used: int
    rerun_budget: int
    rerun_used: int

    # ---- Search fan-out / merge ----
    raw_search_results: Annotated[List[dict], operator.add]  # one list-append per Send() branch
    sources: List[dict]  # Search Merge output (deduped/normalized Source dicts)

    # ---- Validator ----
    validated_sources: List[dict]
    rejected_urls: List[str]
    credible_count: int
    needs_broader_search: bool
    rejection_reasons: Annotated[Dict[str, str], _merge_dicts]  # source id -> why rejected

    # ---- Extraction fan-out / merge ----
    extracted_docs: Annotated[List[dict], operator.add]
    documents: List[dict]  # Extraction Merge output
    extraction_fallbacks: List[str]
    extraction_dropped: List[str]

    # ---- Synthesizer ----
    report_markdown: str
    claims: List[dict]
    synthesizer_used_mock: bool  # last-write-wins: reflects the MOST RECENT synthesis pass

    # ---- Evidence Verifier (single holistic-judgment object) ----
    evidence_report: dict  # EvidenceVerifierOutput.model_dump()
    evidence_verifier_used_mock: bool  # last-write-wins: reflects the MOST RECENT verify pass

    # ---- Critic ----
    approved: bool
    execution_status: str  # "APPROVED" | "REJECTED" | "DEGRADED" — richer than `approved`
    retry_recommended: bool
    rerun_directives: List[dict]
    critic_summary: str
    execution_quality: dict  # ExecutionQuality.model_dump() — computed fresh each Critic pass

    # ---- targeted re-run routing (set by Critic's conditional edge) ----
    rerun_dimensions: List[str]  # dimensions to re-search+extract
    rerun_subtopics: List[str]  # contradictions to resolve
    rerun_recite: bool  # send back to Synthesizer only
    recite_targets: List[str]  # the specific uncited claim texts to fix
    active_queries_by_dimension: Dict[str, List[str]]  # narrowed query set for a targeted re-run
    rerun_search_needed: bool  # False when the only directive is `recite` (no new search)

    # ---- guardrails / observability ----
    audit_log: Annotated[List[AuditEntry], operator.add]
    token_usage: Annotated[List[TokenUsage], operator.add]
    fallback_history: Annotated[List[FallbackEvent], operator.add]

    # ---- final ----
    final_report: str
    evidence_map: dict
    status: str  # running | awaiting_approval | done | failed
