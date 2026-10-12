"""MCP extract adapter — Domain-Specific Adapter for extraction (CLAUDE.md §1).

Owns document fetch/parse (PyMuPDF for PDFs, BeautifulSoup for HTML). The Extractor
*agent* never reaches this module directly — it goes through `mcp_orchestrator` (same
separation as search, hard rule).

Fetching goes through `safe_fetch` (public IPs only, redirects re-validated per hop, size
and total-time caps). The URL is fetched ONCE; fallbacks reuse the same bytes.

Self-healing fallback chain: PyMuPDF/BS4 failing falls back to raw-text extraction (HTML
only). A placeholder mock document is produced ONLY in offline mode (see `_offline_mode`);
in live mode a failure is reported as `failed=True` so extraction_merge drops the source
instead of letting fake text be cited as evidence.
"""

from __future__ import annotations

import os
import re
from typing import Tuple
from urllib.parse import urlparse

from research_langgraph.tools import safe_fetch

ADAPTER_NAME = "mcp_extract_adapter"

_MAX_PDF_PAGES = 200
_MAX_TEXT_CHARS = 500_000  # bound parser output (PDF/HTML bombs); prompts truncate further


def _looks_like_pdf(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def _fetch(url: str) -> bytes:
    return safe_fetch.fetch(url)


def _pdf_text(raw: bytes) -> str:
    import fitz  # PyMuPDF, lazy import: optional dependency

    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        parts, total = [], 0
        for i, page in enumerate(doc):
            if i >= _MAX_PDF_PAGES or total >= _MAX_TEXT_CHARS:
                break
            t = page.get_text()
            parts.append(t)
            total += len(t)
        return "\n".join(parts)[:_MAX_TEXT_CHARS]
    finally:
        doc.close()


def _html_text(raw: bytes) -> str:
    from bs4 import BeautifulSoup  # lazy import: optional dependency

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()[:_MAX_TEXT_CHARS]


def _raw_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", text)  # strip any markup, keep content
    return re.sub(r"\s+", " ", text).strip()[:_MAX_TEXT_CHARS]


def _mock_extract(url: str) -> str:
    return (
        f"Deterministic offline placeholder extraction for {url}. No network or parsing "
        f"library was available, so this stands in for real extracted content."
    )


def _offline_mode() -> bool:
    """Placeholder extraction is only legitimate offline. OFFLINE_MODE=1/0 forces it; when
    unset it is on only if no search provider is configured (mock search returns fake
    example.org URLs that can't be fetched anyway)."""
    flag = os.environ.get("OFFLINE_MODE", "").strip().lower()
    if flag:
        return flag in {"1", "true", "yes", "on"}
    return not (os.environ.get("EXA_API_KEY") or os.environ.get("TAVILY_API_KEY"))


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize(text: str) -> str:
    """Drop control characters. A NUL (\\u0000) in any state value makes the Postgres
    checkpointer's JSONB write fail and kills the whole run (seen on a real page)."""
    return _CONTROL_CHARS.sub("", text).encode("utf-8", "ignore").decode("utf-8")


def extract(url: str) -> Tuple[str, str, bool, bool, str]:
    text, extractor, fell_back, failed, error = _extract(url)
    return _sanitize(text), extractor, fell_back, failed, error


def _extract(url: str) -> Tuple[str, str, bool, bool, str]:
    """Extract text from one URL.

    Returns (text, extractor_used, fell_back, failed, error). `fell_back=True` means a
    lower-fidelity path was used but the source was NOT dropped; `failed=True` means
    every path failed and the source should be dropped (logged, not silently swallowed).
    """
    is_pdf = _looks_like_pdf(url)
    primary_name = "pdf" if is_pdf else "html"
    try:
        raw = _fetch(url)
    except Exception as fetch_err:
        if _offline_mode():
            return _mock_extract(url), "mock", True, False, ""
        return "", "none", False, True, f"fetch failed: {fetch_err}"

    try:
        return (_pdf_text(raw) if is_pdf else _html_text(raw)), primary_name, False, False, ""
    except Exception as primary_err:
        raw_err: Exception | str = "skipped for PDFs (raw bytes are not text)"
        if not is_pdf:
            try:
                return _raw_text(raw), "raw", True, False, ""
            except Exception as e:
                raw_err = e
        if _offline_mode():
            return _mock_extract(url), "mock", True, False, ""
        return "", "none", False, True, f"{primary_err}; raw fallback: {raw_err}"
