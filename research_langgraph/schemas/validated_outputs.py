"""Pydantic models for every node's structured output.

Per CLAUDE.md convention: *every* node that produces structured output validates it
against a model defined here. Malformed LLM output triggers one error-corrective retry
(see routing.model_router.structured_call) rather than a silent pass-through.

These models are the contract between nodes. The graph state (state.py) stores plain
dicts (LangGraph serializes state through the checkpointer), so each node validates into
one of these models and stores `.model_dump()`.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Enums / controlled vocabularies
# --------------------------------------------------------------------------- #
class Complexity(str, Enum):
    simple = "simple"
    medium = "medium"
    complex = "complex"


class Confidence(str, Enum):
    high = "High"
    medium = "Medium"
    unverified = "Unverified"


class SourceType(str, Enum):
    academic = "Academic"
    government = "Government"
    company = "Company"
    documentation = "Documentation"
    github = "GitHub"
    news = "News"
    books = "Books"
    other = "Other"


class ExecutionQualityStatus(str, Enum):
    """Three-state signal for how much of the run used real providers vs. mock fallback.

    FULL: every model/tool call in the run used a real provider.
    PARTIAL: some non-critical node (search, extract, planner, validator, critic) fell
      back to mock, but the Synthesizer and Evidence Verifier — the nodes that actually
      produce and judge the report's content — both used real providers.
    DEGRADED: the Synthesizer and/or Evidence Verifier used mock. Either the report's
      content or the judgment of that content (or both) is synthetic, so any approval
      based on it cannot be trusted (see critic_agent.py's deterministic short-circuit).
    """

    full = "FULL"
    partial = "PARTIAL"
    degraded = "DEGRADED"


class FallbackEvent(BaseModel):
    """One entry in ResearchState.fallback_history — a clean, structured record of a
    self-healing event, independent of audit_log's free-text detail strings. Surfaces
    directly in LangSmith traces since it's part of the graph state."""

    node: str
    reason: str
    fallback: str  # "Mock" | "Gemini" | "Tavily" | ... — what actually served the call


class ExecutionQuality(BaseModel):
    status: ExecutionQualityStatus = ExecutionQualityStatus.full
    reason: List[str] = Field(default_factory=list)


class ExecutionStatus(str, Enum):
    """The Critic's tri-state verdict — richer than a bare approved/rejected bool."""

    approved = "APPROVED"
    rejected = "REJECTED"
    degraded = "DEGRADED"


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
class PlannerOutput(BaseModel):
    """Planner decomposes the topic into dimensions + queries and estimates complexity.

    `complexity` is what the Supervisor reads once, up front, to set the run's
    round/re-run budget (see architecture.md §3).
    """

    dimensions: List[str] = Field(..., min_length=1)
    queries_by_dimension: Dict[str, List[str]] = Field(..., min_length=1)
    complexity: Complexity
    rationale: str = ""

    @field_validator("queries_by_dimension")
    @classmethod
    def _every_dimension_has_queries(cls, v: Dict[str, List[str]]):
        empty = [d for d, qs in v.items() if not qs]
        if empty:
            raise ValueError(f"dimensions with no queries: {empty}")
        return v


# --------------------------------------------------------------------------- #
# Search / sources
# --------------------------------------------------------------------------- #
class Source(BaseModel):
    """A single, normalized search hit. Produced by the search adapter, deduped/normalized
    by Search Merge. `source_type` is intentionally NOT set here — classification is the
    Evidence Verifier's job (hard rule), so it stays None until that pass."""

    id: str
    url: str
    title: str = ""
    snippet: str = ""
    dimension: str = ""
    provider: str = ""  # which adapter returned it (exa / tavily / mock)
    score: float = 0.0
    source_type: Optional[SourceType] = None  # filled in ONLY by Evidence Verifier


class SearchAgentOutput(BaseModel):
    """One parallel Search branch (per dimension). Plans/dispatches queries; the actual
    provider HTTP call is made by the MCP search adapter, not here."""

    dimension: str
    raw_results: List[Source] = Field(default_factory=list)


class SearchMergeOutput(BaseModel):
    """Merge/normalize only — no classification, no diversity scoring (hard rule)."""

    sources: List[Source] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #
class ValidatorOutput(BaseModel):
    """Credibility gate only. Does NOT do holistic judgment.

    Returns ids only — never asks the model to round-trip full source objects. Earlier
    this schema had the LLM return complete `validated_sources: List[Source]` records,
    which risked the model silently paraphrasing or dropping fields it wasn't explicitly
    told to preserve (e.g. `dimension`), quietly corrupting downstream coverage/citation
    tracking. validator_agent.py reconstructs full records from the original merged
    source list using these ids, so the model only ever has to output a classification,
    not a copy of data it already has. `credible_count` and `needs_broader_search` are
    likewise computed in code from `len(accepted_ids)`, not self-reported by the model,
    for the same consistency reason.
    """

    accepted_ids: List[str] = Field(default_factory=list)
    rejected_ids: List[str] = Field(default_factory=list)
    reasons: Dict[str, str] = Field(default_factory=dict)  # rejected id -> why, e.g.
    # {"src_003": "Blog source", "src_007": "Duplicate", "src_009": "Low credibility"} —
    # surfaces in ResearchState.rejection_reasons, so LangSmith traces show exactly why
    # each source disappeared instead of just a final count.

    @field_validator("rejected_ids")
    @classmethod
    def _no_id_in_both_lists(cls, v, info):
        accepted = set(info.data.get("accepted_ids") or [])
        overlap = accepted & set(v)
        if overlap:
            raise ValueError(f"ids cannot be both accepted and rejected: {sorted(overlap)}")
        return v


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #
class ExtractedDoc(BaseModel):
    """One extracted document. `fell_back` records adapter-level self-healing (PyMuPDF ->
    raw text) so the audit trail can tell 'flaky but recovered' apart from 'down'."""

    source_id: str
    url: str
    dimension: str = ""
    text: str = ""
    char_count: int = 0
    extractor: str = ""  # pdf / html / raw / mock
    fell_back: bool = False
    failed: bool = False
    error: str = ""


class ExtractorOutput(BaseModel):
    """One parallel Extract branch (per URL)."""

    doc: ExtractedDoc


class ExtractionMergeOutput(BaseModel):
    """Re-associate extracted chunks by dimension, drop failed, log fallbacks."""

    documents: List[ExtractedDoc] = Field(default_factory=list)
    fallbacks: List[str] = Field(default_factory=list)  # source_ids that fell back
    dropped: List[str] = Field(default_factory=list)  # source_ids that failed entirely


# --------------------------------------------------------------------------- #
# Synthesizer
# --------------------------------------------------------------------------- #
class Claim(BaseModel):
    """A factual assertion in the draft, with the source(s) the Synthesizer cited for it.
    Uncited claims (empty source_ids) are exactly what the Evidence Verifier flags."""

    id: str
    text: str
    dimension: str = ""
    source_ids: List[str] = Field(default_factory=list)


class SynthesizerOutput(BaseModel):
    report_markdown: str
    claims: List[Claim] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Evidence Verifier — the single holistic-judgment object (all 5 checks)
# --------------------------------------------------------------------------- #
class Contradiction(BaseModel):
    topic: str
    sources: List[str] = Field(default_factory=list)
    status: str = "Needs clarification"
    reason: str = ""


class CitationCompleteness(BaseModel):
    uncited_claims: List[str] = Field(default_factory=list)
    completeness_score: float = 1.0


class EvidenceVerifierOutput(BaseModel):
    """All five holistic checks land in ONE object consumed by the Critic (hard rule:
    nothing else computes coverage / contradictions / diversity / citation completeness)."""

    claim_confidence: Dict[str, Confidence] = Field(default_factory=dict)
    coverage: Dict[str, float] = Field(default_factory=dict)
    contradictions: List[Contradiction] = Field(default_factory=list)
    source_types: Dict[str, SourceType] = Field(default_factory=dict)
    source_diversity: Dict[str, float] = Field(default_factory=dict)
    diversity_flag: str = ""
    citation_completeness: CitationCompleteness = Field(default_factory=CitationCompleteness)

    @field_validator("coverage", "source_diversity")
    @classmethod
    def _scores_in_range(cls, v: Dict[str, float]):
        bad = {k: s for k, s in v.items() if not (0.0 <= s <= 1.0)}
        if bad:
            raise ValueError(f"scores must be in [0,1]: {bad}")
        return v


# --------------------------------------------------------------------------- #
# Critic — decides on TARGETED re-runs only (hard rule: never a blanket restart)
# --------------------------------------------------------------------------- #
class ReRunAction(str, Enum):
    research_dimension = "research_dimension"       # coverage gap -> re-search+extract one dim
    resolve_contradiction = "resolve_contradiction"  # contradiction -> resolve one sub-topic
    improve_diversity = "improve_diversity"          # poor diversity -> re-search w/ source-type hint
    recite = "recite"                                # uncited claim -> back to Synthesizer only
    retry_degraded = "retry_degraded"                # DEGRADED execution -> redo Synthesizer +
                                                       # Evidence Verifier fresh, no new search needed


class ReRunDirective(BaseModel):
    action: ReRunAction
    target: str = ""    # dimension name / sub-topic / source-type instruction
    reason: str = ""


class CriticOutput(BaseModel):
    approved: bool
    directives: List[ReRunDirective] = Field(default_factory=list)
    summary: str = ""

    @field_validator("directives")
    @classmethod
    def _directives_match_approval(cls, v, info):
        approved = info.data.get("approved")
        if approved and v:
            raise ValueError("an approved report must not carry re-run directives")
        if approved is False and not v:
            # Without this, a rejected report with zero directives sends prepare_rerun_node
            # an empty targeted-search set, which blindly re-runs the Synthesizer with no
            # guidance at all — burning re-run budget without ever addressing what the
            # Critic actually disliked. Forces the built-in corrective retry in
            # model_router.call_structured to make the model say what's wrong.
            raise ValueError("a rejected report must specify at least one targeted directive")
        return v


__all__ = [
    "Complexity",
    "Confidence",
    "SourceType",
    "ExecutionQualityStatus",
    "FallbackEvent",
    "ExecutionQuality",
    "ExecutionStatus",
    "PlannerOutput",
    "Source",
    "SearchAgentOutput",
    "SearchMergeOutput",
    "ValidatorOutput",
    "ExtractedDoc",
    "ExtractorOutput",
    "ExtractionMergeOutput",
    "Claim",
    "SynthesizerOutput",
    "Contradiction",
    "CitationCompleteness",
    "EvidenceVerifierOutput",
    "ReRunAction",
    "ReRunDirective",
    "CriticOutput",
]
