"""Search Merge — dedup + normalize only.

Hard rule: source-type classification and diversity scoring do NOT happen here — that's
the Evidence Verifier's job exclusively (architecture.md §3). This node's job stays
narrow on purpose: gather the parallel Search branches' raw_search_results, dedup by URL,
and normalize the field shape before handing off to Validation.
"""

from __future__ import annotations

from typing import Dict, List


def _dedup_by_url(sources: List[dict]) -> List[dict]:
    seen: Dict[str, dict] = {}
    dims: Dict[str, List[str]] = {}
    for s in sources:
        url = s.get("url")
        if not url:
            continue
        d = s.get("dimension", "")
        if d and d not in dims.setdefault(url, []):
            dims[url].append(d)  # keep EVERY dimension a URL was found under
        if url not in seen or s.get("score", 0) > seen[url].get("score", 0):
            seen[url] = s
    return [{**s, "dimensions": dims.get(url, [])} for url, s in seen.items()]


def search_merge(state: dict) -> dict:
    deduped = _dedup_by_url(state.get("raw_search_results", []))
    normalized = [
        {
            "id": s.get("id", ""),
            "url": s["url"],
            "title": s.get("title", ""),
            "snippet": s.get("snippet", ""),
            "dimension": s.get("dimension", ""),  # best-scoring hit's dimension (extraction tag)
            "dimensions": s.get("dimensions", []),  # all dimensions this URL covers
            "provider": s.get("provider", ""),
            "score": s.get("score", 0.0),
            "source_type": None,  # classification happens only in the Evidence Verifier
        }
        for s in deduped
    ]
    # is_broaden is reset here: it only needs to survive planner -> after_planner routing
    return {"sources": normalized, "is_broaden": False}
