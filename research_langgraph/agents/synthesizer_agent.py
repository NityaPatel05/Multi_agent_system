"""Synthesizer agent — drafts the report from extracted documents, citing claims back to
source ids. Citation completeness/support is CHECKED later by the Evidence Verifier (hard
rule) — this agent just cites honestly as it writes rather than inventing sources.

Method (plan-and-execute, as in Open Deep Research / Anthropic's research system): each
planned dimension is written as its OWN section from its OWN documents, in parallel, then
the sections are assembled under a short executive summary. One giant prompt over every
document produced shallow reports where the first dimensions crowded out the rest; per-
section synthesis gives each dimension a full evidence budget and depth. The only
post-processing here is sanitation (drop citations to ids that don't exist), not judgment.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from research_langgraph.agents.grounding import drop_invalid_citations
from research_langgraph.agents.prompt_utils import UNTRUSTED_NOTE, render_documents
from research_langgraph.routing.model_router import RoutedCallResult, call_structured, fallback_history_entries
from research_langgraph.schemas.validated_outputs import Claim, SynthesizerOutput

SECTION_PROMPT = """You are the Synthesizer writing ONE section of a research report.
You are given the topic, the section's dimension, and the extracted documents for it.

Write the section body (do NOT include a heading) as substantive prose, roughly 250-450
words: concrete facts, figures, names, dates, mechanisms, trade-offs, and where sources
disagree say so. Use ONLY information found in the documents; never add outside knowledge
and never invent numbers. Prefer the most specific and most recent information.

Then list the individual factual claims you made (one distinct fact per claim, written as a
standalone sentence; aim for at least 3-5 claims) with the source_ids that support each (ids come from the documents'
source_id attributes). Every factual sentence in the body must be backed by a claim. If a
claim is not backed by any document, still list it but leave source_ids empty rather than
inventing a citation. Set each claim's dimension to the section's dimension.
""" + UNTRUSTED_NOTE

SUMMARY_PROMPT = """You write the executive summary of a research report. Given the topic
and the finished sections, write 3-5 sentences (about 90-150 words) capturing the most
important findings across sections. Add nothing that is not in the sections. Put the text
in report_markdown and return no claims."""

REVISE_PROMPT = """You are the Synthesizer making a targeted revision of an existing report.
Use ONLY the documents given. Keep everything not mentioned unchanged. Return the full
revised report_markdown and the full claims list with source_ids from the documents.
""" + UNTRUSTED_NOTE


def _docs_for(dim: str, documents: List[dict]) -> List[dict]:
    return [d for d in documents if dim == d.get("dimension") or dim in (d.get("dimensions") or [])]


def _mock_section(dim: str, docs: List[dict]) -> SynthesizerOutput:
    body = " ".join((d.get("text") or "")[:200] for d in docs).strip()
    claims = [Claim(id="m", text=f"Summary claim for {dim}", dimension=dim,
                    source_ids=[d["source_id"] for d in docs[:1]])] if docs else []
    return SynthesizerOutput(report_markdown=body or "(no extracted content)", claims=claims)


def _mock_revise(topic: str, documents: List[dict], dims: List[str]) -> SynthesizerOutput:
    parts, claims = [f"# {topic}"], []
    for i, dim in enumerate(dims or ["General"]):
        sec = _mock_section(dim, _docs_for(dim, documents) or documents)
        parts.append(f"## {dim}\n\n{sec.report_markdown}")
        claims += [c.model_copy(update={"id": f"c{i}"}) for c in sec.claims]
    return SynthesizerOutput(report_markdown="\n\n".join(parts), claims=claims)


_TYPOGRAPHIC = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u00a0": " ", "\u202f": " "})


def _tidy(text: str) -> str:
    """Models (Gemini especially) emit non-breaking hyphens/spaces that break search, diffing
    and plain-text consoles; normalise to ASCII equivalents."""
    return text.translate(_TYPOGRAPHIC)


def _strip_leading_heading(body: str, dim: str) -> str:
    body = _tidy(body)
    return re.sub(r"\A\s*#{1,4}[^\n]*\n+", "", body.strip(), count=1) if body.lstrip().startswith("#") else body.strip()


def _title(topic: str) -> str:
    t = topic.strip()
    return t[:1].upper() + t[1:] if t and t == t.lower() else t


def _write_section(topic: str, dim: str, docs: List[dict], budget_mode: str):
    parsed, result = call_structured(
        node_name="synthesizer",
        system=SECTION_PROMPT,
        user=f"Topic: {topic}\nSection dimension: {dim}\nDocuments: {render_documents(docs)}",
        schema=SynthesizerOutput,
        mock_fn=lambda: _mock_section(dim, docs),
        budget_mode=budget_mode,
    )
    return dim, parsed, result


def _collect(results: List[RoutedCallResult]) -> Tuple[list, list, list, bool]:
    audit, tokens, fallbacks = [], [], []
    for r in results:
        audit += r.audit_entries()
        tokens.append(r.token_usage_entry())
        fallbacks += fallback_history_entries(r)
    return audit, tokens, fallbacks, any(r.used_mock for r in results)


def synthesizer_agent(state: dict) -> dict:
    topic = state["topic"]
    documents = state.get("documents", [])
    dimensions = state.get("dimensions", []) or sorted({d.get("dimension", "General") for d in documents})
    budget_mode = state.get("budget_mode", "balanced")
    recite_targets = state.get("recite_targets") or []

    # --- targeted revision (only uncited claims to fix, no new evidence): edit in place ---
    if recite_targets and not state.get("rerun_search_needed") and state.get("report_markdown"):
        parsed, result = call_structured(
            node_name="synthesizer",
            system=REVISE_PROMPT,
            user=(f"Topic: {topic}\nUncited claims to fix (add a citation or remove/rephrase): "
                  f"{recite_targets}\nExisting report:\n{state['report_markdown']}\n"
                  f"Documents: {render_documents(documents)}"),
            schema=SynthesizerOutput,
            mock_fn=lambda: _mock_revise(topic, documents, dimensions),
            budget_mode=budget_mode,
        )
        claims, removed = drop_invalid_citations([c.model_dump() for c in parsed.claims], documents)
        audit, tokens, fallbacks, used_mock = _collect([result])
        return {
            "report_markdown": parsed.report_markdown, "claims": claims,
            "citations_removed": removed, "synthesizer_used_mock": used_mock,
            "audit_log": audit, "token_usage": tokens, "fallback_history": fallbacks,
        }

    # --- full draft: one section per dimension, written in parallel ---
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(dimensions)))) as pool:
        futures = [pool.submit(_write_section, topic, dim, _docs_for(dim, documents), budget_mode)
                   for dim in dimensions if _docs_for(dim, documents)]
        done = [f.result() for f in futures]
    by_dim = {dim: (parsed, res) for dim, parsed, res in done}

    sections, claims = [], []
    for i, dim in enumerate(dimensions):
        if dim not in by_dim:  # no evidence: say so (the Verifier scores it 0 and triggers a targeted re-search)
            sections.append(f"## {dim}\n\n*No sources were retrieved for this dimension.*")
            continue
        parsed, _ = by_dim[dim]
        sections.append(f"## {dim}\n\n{_tidy(_strip_leading_heading(parsed.report_markdown, dim))}")
        for n, c in enumerate(parsed.claims, 1):
            claims.append({**c.model_dump(), "id": f"c{i + 1}_{n}", "dimension": dim, "text": _tidy(c.text)})

    results = [res for _, res in by_dim.values()]
    body = "\n\n".join(sections)

    summary = ""
    if sections:
        sp, sres = call_structured(
            node_name="synthesizer", system=SUMMARY_PROMPT,
            user=f"Topic: {topic}\nSections:\n{body[:24000]}",
            schema=SynthesizerOutput, mock_fn=lambda: SynthesizerOutput(report_markdown=""),
            budget_mode=budget_mode,
        )
        summary = sp.report_markdown.strip()
        results.append(sres)

    report = f"# {_title(topic)}\n\n" + (f"{_tidy(summary)}\n\n" if summary else "") + body
    claims, removed = drop_invalid_citations(claims, documents)
    audit, tokens, fallbacks, _ = _collect(results)
    used_mock = any(r.used_mock for r, _ in zip(results, by_dim))  # section calls only, not the summary
    return {
        "report_markdown": report,
        "claims": claims,
        "citations_removed": removed,
        "synthesizer_used_mock": used_mock,
        "audit_log": audit,
        "token_usage": tokens,
        "fallback_history": fallbacks,
    }
