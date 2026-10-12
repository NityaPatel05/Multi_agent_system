"""Deterministic grounding checks (no LLM): do the citations exist, does the cited text
actually support the claim, and does each planned dimension have real supported content.

Why this exists: an LLM asked to score its own report's coverage/faithfulness tends to
inflate (judges favour fluent, long, same-family output). Best practice for research
systems is to run cheap deterministic checks FIRST - citation existence, support overlap
between a claim and the cited passages - and treat the LLM as an interpreter on top, never
as the sole gate. The Evidence Verifier takes the STRICTER of the LLM score and these.

Support is a lexical-overlap heuristic (fraction of the claim's content words, and ALL of
its numbers, present in the cited documents). It cannot prove a claim true, but it reliably
catches fabricated figures/names and citations that point at unrelated pages.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set, Tuple

MIN_SUPPORTED_CLAIMS_PER_DIM = 2
SUPPORT_THRESHOLD = 0.5  # claim counts as "supported" at/above this overlap
LOW_SUPPORT = 0.3  # below this the claim is treated as unverified

_STOP = frozenset("""
about above after again against also because been before being between both cannot could does doing down during each
from further have having here herself himself into itself just more most much must myself only other ours ourselves
over same should some such than that their theirs them then there these they this those through under until very
were what when where which while whom will with would your yours yourself yourselves often typically generally
""".split())
_WORD = re.compile(r"[a-z0-9][a-z0-9\-\.%]*[a-z0-9%]|[a-z0-9]")
_NUM = re.compile(r"\d[\d,\.]*")


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokens(text: str) -> Set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if len(w) >= 4 and w not in _STOP}


def numbers(text: str) -> Set[str]:
    return {n.strip(".,").replace(",", "") for n in _NUM.findall(text) if len(n.strip(".,")) >= 2}


def valid_source_ids(documents: Iterable[dict]) -> Set[str]:
    return {d["source_id"] for d in documents if d.get("source_id") and not d.get("failed")}


def drop_invalid_citations(claims: List[dict], documents: List[dict]) -> Tuple[List[dict], int]:
    """Remove source_ids that don't exist among the extracted documents (hallucinated
    citations). Returns (claims with cleaned source_ids, number of citations removed)."""
    ok = valid_source_ids(documents)
    removed = 0
    out = []
    for c in claims:
        ids = [s for s in c.get("source_ids", []) if s in ok]
        removed += len(c.get("source_ids", [])) - len(ids)
        out.append({**c, "source_ids": list(dict.fromkeys(ids))})
    return out, removed


def claim_support(claim_text: str, cited_docs: List[dict]) -> float:
    """0..1: how much of the claim is findable in the documents it cites."""
    if not cited_docs:
        return 0.0
    blob = " ".join((d.get("text") or "") for d in cited_docs)
    doc_tokens, doc_numbers = tokens(blob), numbers(blob)
    ctoks = tokens(claim_text)
    if not ctoks:
        return 1.0  # nothing checkable (very short claim); don't penalize
    score = len(ctoks & doc_tokens) / len(ctoks)
    cnums = numbers(claim_text)
    if cnums:  # every figure the claim asserts must appear in the sources
        missing = len(cnums - doc_numbers) / len(cnums)
        score *= 1.0 - 0.7 * missing
    return round(score, 3)


def evaluate_claims(claims: List[dict], documents: List[dict]) -> Dict[str, dict]:
    by_id = {d["source_id"]: d for d in documents if d.get("source_id")}
    result = {}
    for c in claims:
        cited = [by_id[s] for s in c.get("source_ids", []) if s in by_id]
        result[c["id"]] = {"cited": len(cited), "support": claim_support(c.get("text", ""), cited)}
    return result


def _heading_texts(report_markdown: str) -> List[str]:
    return [m.group(1).strip() for m in re.finditer(r"^#{1,4}\s+(.+)$", report_markdown, re.M)]


def has_section(dimension: str, report_markdown: str) -> bool:
    want = tokens(dimension) or {dimension.lower()}
    for h in _heading_texts(report_markdown):
        have = tokens(h) or {h.lower()}
        if len(want & have) / len(want) >= 0.5:
            return True
    return False


def dimension_coverage(dimensions: List[str], claims: List[dict], evals: Dict[str, dict],
                       report_markdown: str) -> Dict[str, float]:
    """1.0 needs a dedicated section AND >= MIN_SUPPORTED_CLAIMS_PER_DIM claims that are
    cited and actually supported by their sources; partial credit below that."""
    out = {}
    for dim in dimensions:
        supported = sum(
            1 for c in claims
            if c.get("dimension") == dim and evals.get(c["id"], {}).get("cited", 0) > 0
            and evals[c["id"]]["support"] >= SUPPORT_THRESHOLD
        )
        base = min(1.0, supported / MIN_SUPPORTED_CLAIMS_PER_DIM)
        out[dim] = round(base if has_section(dim, report_markdown) else base * 0.5, 2)
    return out


def summary_metrics(claims: List[dict], evals: Dict[str, dict], removed_citations: int = 0) -> Dict[str, float]:
    n = max(1, len(claims))
    cited = sum(1 for c in claims if evals.get(c["id"], {}).get("cited", 0) > 0)
    supported = sum(1 for c in claims if evals.get(c["id"], {}).get("support", 0) >= SUPPORT_THRESHOLD
                    and evals.get(c["id"], {}).get("cited", 0) > 0)
    return {
        "cited_claim_rate": round(cited / n, 3),
        "claim_support_rate": round(supported / n, 3),
        "invalid_citations_removed": float(removed_citations),
    }
