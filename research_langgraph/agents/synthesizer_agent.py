"""Synthesizer agent — drafts the report from extracted documents, citing claims back to
source ids where it can. Citation completeness is CHECKED later by the Evidence Verifier
(hard rule) — this agent just cites honestly as it writes rather than inventing sources.
"""

from __future__ import annotations

from research_langgraph.routing.model_router import call_structured, fallback_history_entries
from research_langgraph.schemas.validated_outputs import Claim, SynthesizerOutput

SYSTEM_PROMPT = """You are the Synthesizer agent. Given extracted documents (grouped by
dimension) and the topic, write a well-organized markdown report covering every
dimension, and list the individual factual claims you made along with the source_ids
backing each one (source_ids come from the documents you were given). Cite honestly — if
a claim isn't backed by any document, still include it as a claim but leave source_ids
empty rather than inventing a citation.
"""


def _mock_synthesize(topic: str, documents: list) -> SynthesizerOutput:
    by_dim: dict[str, list] = {}
    for d in documents:
        by_dim.setdefault(d.get("dimension") or "General", []).append(d)

    sections = []
    claims = []
    for i, (dim, docs) in enumerate(by_dim.items()):
        body = " ".join((d.get("text") or "")[:200] for d in docs).strip()
        sections.append(f"## {dim}\n\n{body or '(no extracted content)'}")
        claims.append(Claim(
            id=f"c{i}",
            text=f"Summary claim for {dim}",
            dimension=dim,
            source_ids=[d["source_id"] for d in docs[:1]],
        ))

    report = f"# {topic}\n\n" + ("\n\n".join(sections) if sections else "(no sources extracted)")
    return SynthesizerOutput(report_markdown=report, claims=claims)


def synthesizer_agent(state: dict) -> dict:
    topic = state["topic"]
    documents = state.get("documents", [])
    budget_mode = state.get("budget_mode", "balanced")

    recite_targets = state.get("recite_targets") or []
    revise_hint = ""
    if recite_targets:
        revise_hint = (
            f"\n\nThis is a targeted revision, not a fresh draft: start from the existing "
            f"report below and either add a citation (source_id) for each of these "
            f"previously-uncited claims or remove/rephrase them if no supporting document "
            f"exists. Keep the rest of the report unchanged.\n"
            f"Uncited claims to fix: {recite_targets}\n"
            f"Existing report:\n{state.get('report_markdown', '')}"
        )

    parsed, result = call_structured(
        node_name="synthesizer",
        system=SYSTEM_PROMPT,
        user=f"Topic: {topic}\nDocuments: {documents}{revise_hint}",
        schema=SynthesizerOutput,
        mock_fn=lambda: _mock_synthesize(topic, documents),
        budget_mode=budget_mode,
    )
    return {
        "report_markdown": parsed.report_markdown,
        "claims": [c.model_dump() for c in parsed.claims],
        "synthesizer_used_mock": result.used_mock,
        "audit_log": result.audit_entries(),
        "token_usage": [result.token_usage_entry()],
        "fallback_history": fallback_history_entries(result),
    }
