"""Extraction Merge — re-associate extracted chunks with their originating dimension,
drop failed extractions, and surface which sources fell back to a lower-fidelity
extraction path so the audit trail captures it (architecture.md §3).
"""

from __future__ import annotations


def extraction_merge(state: dict) -> dict:
    docs = state.get("extracted_docs", [])
    if "validated_sources" in state:  # drop documents of sources that are no longer validated
        valid_ids = {s["id"] for s in state["validated_sources"]}
        docs = [d for d in docs if d.get("source_id") in valid_ids]
    by_id: dict = {}
    for d in docs:  # one document per source (last extraction wins)
        by_id[d["source_id"]] = d
    docs = list(by_id.values())
    kept = [d for d in docs if not d.get("failed")]
    dropped = [d["source_id"] for d in docs if d.get("failed")]
    fallbacks = [d["source_id"] for d in docs if d.get("fell_back")]
    kept = sorted(kept, key=lambda d: d.get("dimension", ""))
    return {
        "documents": kept,
        "extraction_fallbacks": fallbacks,
        "extraction_dropped": dropped,
    }
