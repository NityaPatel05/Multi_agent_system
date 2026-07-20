"""Evidence Verifier — the single holistic-judgment stage.

Hard rule: coverage, contradictions, source-type/diversity, and citation completeness are
computed HERE ONLY — no other node (Search Merge, Validator, Critic) reimplements any of
this. Five checks land in one structured object consumed directly by the Critic
(architecture.md §3).
"""

from __future__ import annotations

from collections import Counter
from typing import List

from research_langgraph.routing.model_router import call_structured, fallback_history_entries
from research_langgraph.schemas.validated_outputs import (
    CitationCompleteness,
    Confidence,
    EvidenceVerifierOutput,
    SourceType,
)

SYSTEM_PROMPT = """You are the Evidence Verifier. You perform ALL holistic judgment for
this run — no other agent does this. Given the draft report's claims, the extracted
documents backing them, the source list, and the Planner's dimension list, produce ONE
structured object with:
1. claim_confidence: per-claim High/Medium/Unverified confidence based on how well the
   extracted text actually supports it.
2. coverage: per-dimension score 0-1 against the Planner's dimensions.
3. contradictions: any sub-topics where sources disagree.
4. source_types: classify EVERY source (Academic/Government/Company/Documentation/GitHub/
   News/Books/Other) — this is the only place source classification happens.
5. source_diversity: proportion of sources per type, plus a diversity_flag describing any
   over-indexing.
6. citation_completeness: list any claim with no supporting source_ids as uncited, plus
   an overall completeness_score.
"""


def _classify(url: str) -> SourceType:
    u = url.lower()
    if "github.com" in u:
        return SourceType.github
    if u.endswith(".gov") or ".gov/" in u:
        return SourceType.government
    if u.endswith(".edu") or "arxiv.org" in u or ".ac." in u:
        return SourceType.academic
    if "docs." in u or "/docs" in u:
        return SourceType.documentation
    if "news" in u or "blog" in u:
        return SourceType.news
    return SourceType.company


def _mock_verify(dimensions: List[str], claims: List[dict], sources: List[dict]) -> EvidenceVerifierOutput:
    claim_confidence = {
        c["id"]: (Confidence.medium if c.get("source_ids") else Confidence.unverified)
        for c in claims
    }
    covered_dims = {c.get("dimension") for c in claims if c.get("source_ids")}
    coverage = {d: (1.0 if d in covered_dims else 0.0) for d in dimensions}

    source_types = {s["id"]: _classify(s["url"]) for s in sources}
    counts = Counter(source_types.values())
    total = max(1, len(source_types))
    diversity = {st.value: round(counts.get(st, 0) / total, 2) for st in SourceType}
    dominant = max(diversity, key=diversity.get) if diversity else "Other"
    academic_gov = diversity.get("Academic", 0) + diversity.get("Government", 0)
    diversity_flag = (
        f"poor — over-indexed on {dominant}, little Academic/Government representation"
        if academic_gov < 0.15 else "acceptable spread across source types"
    )

    uncited = [c["text"] for c in claims if not c.get("source_ids")]
    completeness_score = round(1 - (len(uncited) / max(1, len(claims))), 2)

    return EvidenceVerifierOutput(
        claim_confidence=claim_confidence,
        coverage=coverage,
        contradictions=[],
        source_types=source_types,
        source_diversity=diversity,
        diversity_flag=diversity_flag,
        citation_completeness=CitationCompleteness(
            uncited_claims=uncited, completeness_score=completeness_score
        ),
    )


def evidence_verifier_agent(state: dict) -> dict:
    dimensions = state.get("dimensions", [])
    claims = state.get("claims", [])
    sources = state.get("sources", [])
    documents = state.get("documents", [])
    budget_mode = state.get("budget_mode", "balanced")
    parsed, result = call_structured(
        node_name="evidence_verifier",
        system=SYSTEM_PROMPT,
        user=(
            f"Dimensions: {dimensions}\nClaims: {claims}\nSources: {sources}\n"
            f"Documents: {documents}"
        ),
        schema=EvidenceVerifierOutput,
        mock_fn=lambda: _mock_verify(dimensions, claims, sources),
        budget_mode=budget_mode,
    )
    return {
        "evidence_report": parsed.model_dump(mode="json"),
        "evidence_verifier_used_mock": result.used_mock,
        "audit_log": result.audit_entries(),
        "token_usage": [result.token_usage_entry()],
        "fallback_history": fallback_history_entries(result),
    }
