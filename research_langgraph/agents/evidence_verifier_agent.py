"""Evidence Verifier — the single holistic-judgment stage.

Hard rule: coverage, contradictions, source-type/diversity, and citation completeness are
computed HERE ONLY — no other node (Search Merge, Validator, Critic) reimplements any of
this. Five checks land in one structured object consumed directly by the Critic
(architecture.md §3).
"""

from __future__ import annotations

from collections import Counter
from typing import List

from research_langgraph.agents import grounding
from research_langgraph.agents.prompt_utils import UNTRUSTED_NOTE, render_documents
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
2. coverage: per-dimension score 0-1 against the Planner's dimensions. Use this anchored
   scale and be strict - do NOT give 1.0 unless you checked the documents yourself:
   1.0 = the report has a dedicated section for the dimension with at least 2 distinct
   claims, each clearly supported by the documents it cites; 0.5 = a section exists but is
   thin, vague, or only partly supported; 0.0 = no section, or no supported claims.
3. contradictions: any sub-topics where sources disagree.
4. source_types: classify EVERY source (Academic/Government/Company/Documentation/GitHub/
   News/Books/Other) — this is the only place source classification happens.
5. source_diversity: proportion of sources per type, plus a diversity_flag describing any
   over-indexing.
6. citation_completeness: list any claim with no supporting source_ids as uncited, plus
   an overall completeness_score.
""" + UNTRUSTED_NOTE


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


def _apply_grounding(parsed: EvidenceVerifierOutput, state: dict, dimensions: List[str],
                     claims: List[dict], documents: List[dict]) -> dict:
    """Run the deterministic checks and combine with the LLM's judgment, taking the STRICTER
    of the two (LLM judges inflate). Also recomputes citation completeness from the actual
    citations rather than trusting the model's self-report."""
    evals = grounding.evaluate_claims(claims, documents)
    det_cov = grounding.dimension_coverage(dimensions, claims, evals, state.get("report_markdown", ""))
    out = parsed.model_dump(mode="json")

    out["coverage"] = {d: round(min(out["coverage"].get(d, det_cov[d]), det_cov[d]), 2) for d in dimensions}

    conf = dict(out["claim_confidence"])
    for c in claims:
        e = evals.get(c["id"], {"cited": 0, "support": 0.0})
        if e["cited"] == 0 or e["support"] < grounding.LOW_SUPPORT:
            conf[c["id"]] = Confidence.unverified.value
        elif e["support"] < grounding.SUPPORT_THRESHOLD and conf.get(c["id"]) == Confidence.high.value:
            conf[c["id"]] = Confidence.medium.value
    out["claim_confidence"] = conf

    uncited = [c["text"] for c in claims if evals.get(c["id"], {}).get("cited", 0) == 0]
    out["citation_completeness"] = {
        "uncited_claims": uncited,
        "completeness_score": round(1 - len(uncited) / max(1, len(claims)), 2),
    }
    out["grounding"] = grounding.summary_metrics(claims, evals, int(state.get("citations_removed", 0)))
    return out


def evidence_verifier_agent(state: dict) -> dict:
    dimensions = state.get("dimensions", [])
    claims = state.get("claims", [])
    # only Validator-accepted sources: rejected/spam ones must not skew diversity/coverage
    sources = state.get("validated_sources", state.get("sources", []))
    documents = state.get("documents", [])
    budget_mode = state.get("budget_mode", "balanced")
    parsed, result = call_structured(
        node_name="evidence_verifier",
        system=SYSTEM_PROMPT,
        user=(
            f"Dimensions: {dimensions}\nClaims: {claims}\nSources: {sources}\n"
            f"Documents: {render_documents(documents)}"
        ),
        schema=EvidenceVerifierOutput,
        mock_fn=lambda: _mock_verify(dimensions, claims, sources),
        budget_mode=budget_mode,
    )
    report = _apply_grounding(parsed, state, dimensions, claims, documents)
    return {
        "evidence_report": report,
        "evidence_verifier_used_mock": result.used_mock,
        "audit_log": result.audit_entries(),
        "token_usage": [result.token_usage_entry()],
        "fallback_history": fallback_history_entries(result),
    }
