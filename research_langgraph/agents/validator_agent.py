"""Validator agent — credibility gate only.

Hard rule: does NOT classify source type and does NOT score diversity — all holistic
judgment, including source-type/diversity, lives in the Evidence Verifier only. This
node's one job is a basic credibility pass; `Validator -> Planner` (conditional edge in
main.py) triggers a broader search round when fewer than 3 sources clear the bar.

The model only ever returns ids + rejection reasons (schemas.ValidatorOutput) — never
full source records — so it can't silently drop/paraphrase a field like `dimension` when
asked to copy data back. Full records are reconstructed here from the original merged
source list by id; `credible_count`/`needs_broader_search` are computed in code from
`len(accepted_ids)` rather than trusted as separate self-reported fields, for the same
consistency reason.
"""

from __future__ import annotations

from typing import Dict, List

from research_langgraph.routing.model_router import call_structured, fallback_history_entries
from research_langgraph.schemas.validated_outputs import ValidatorOutput

SYSTEM_PROMPT = """You are the Validator agent. You receive a list of candidate sources,
each with an id, url, title, and snippet. Your ONLY job is a basic credibility gate:
reject sources that are obviously spam, broken, off-topic, low-credibility, or duplicates
of another source in the list. Do NOT classify source type (academic/news/etc.) and do
NOT score diversity — that is handled elsewhere.

Return:
- accepted_ids: ids of sources that pass the credibility bar
- rejected_ids: ids of sources that don't
- reasons: for EACH rejected id, a short reason (e.g. "Blog source", "Duplicate", "Low credibility")

Every source id must appear in exactly one of accepted_ids or rejected_ids — reference
sources ONLY by their id, never invent or alter it.
"""

_BLOCKED_DOMAIN_HINTS = ("spam.", "example-broken.")
_CREDIBLE_THRESHOLD = 3


def _heuristic_pass(s: dict) -> bool:
    url = s.get("url", "")
    if not url.startswith(("http://", "https://")):
        return False
    if any(hint in url for hint in _BLOCKED_DOMAIN_HINTS):
        return False
    return True


def _mock_validate(sources: List[dict]) -> ValidatorOutput:
    accepted_ids, rejected_ids, reasons = [], [], {}
    for s in sources:
        if _heuristic_pass(s):
            accepted_ids.append(s["id"])
        else:
            rejected_ids.append(s["id"])
            reasons[s["id"]] = "blocked domain or malformed URL"
    return ValidatorOutput(accepted_ids=accepted_ids, rejected_ids=rejected_ids, reasons=reasons)


def _brief(sources: List[dict]) -> List[Dict[str, str]]:
    return [
        {"id": s["id"], "url": s["url"], "title": s.get("title", ""), "snippet": s.get("snippet", "")}
        for s in sources
    ]


def validator_agent(state: dict) -> dict:
    sources = state.get("sources", [])
    by_id = {s["id"]: s for s in sources}
    budget_mode = state.get("budget_mode", "balanced")

    parsed, result = call_structured(
        node_name="validator",
        system=SYSTEM_PROMPT,
        user=f"Sources:\n{_brief(sources)}",
        schema=ValidatorOutput,
        mock_fn=lambda: _mock_validate(sources),
        budget_mode=budget_mode,
    )

    # Reconstruct full records in code from the original list — the model never had to
    # (and couldn't) mangle them, since it only ever dealt with ids.
    validated_sources = [by_id[i] for i in parsed.accepted_ids if i in by_id]
    rejected_urls = [by_id[i]["url"] for i in parsed.rejected_ids if i in by_id]
    credible_count = len(validated_sources)

    return {
        "validated_sources": validated_sources,
        "rejected_urls": rejected_urls,
        "credible_count": credible_count,
        "needs_broader_search": credible_count < _CREDIBLE_THRESHOLD,
        "rejection_reasons": parsed.reasons,
        "audit_log": result.audit_entries(),
        "token_usage": [result.token_usage_entry()],
        "fallback_history": fallback_history_entries(result),
    }
