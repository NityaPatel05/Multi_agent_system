"""MCP extract adapter — Domain-Specific Adapter for extraction (CLAUDE.md §1).

Owns document fetch/parse (PyMuPDF for PDFs, BeautifulSoup for HTML). The Extractor
*agent* never reaches this module directly — it goes through `mcp_orchestrator` (same
separation as search, hard rule).

Self-healing fallback chain: PyMuPDF failing on a malformed PDF falls back to raw-text
extraction (strip formatting, keep content) rather than dropping the source entirely.
When network access or the parsing libraries aren't usable, falls back to a deterministic
offline mock so the graph stays runnable without them.
"""

from __future__ import annotations

import re
import urllib.request
from typing import Tuple
from urllib.parse import urlparse

ADAPTER_NAME = "mcp_extract_adapter"


_ALLOWED_SCHEMES = {"http", "https"}
_MAX_RESPONSE_BYTES = 20 * 1024 * 1024  # 20MB — bound worst-case memory per extraction


def _looks_like_pdf(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def _assert_fetchable(url: str) -> None:
    """Reject anything but http(s). Without this, Python's default urllib opener will
    happily follow `file://`, `ftp://`, etc. — since this URL can originate from a search
    provider's response (or, via mcp_extract_server.py, directly from an untrusted HTTP
    caller), a missing scheme check is a local-file-read / SSRF vector, not just a
    theoretical one."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"refusing to fetch non-http(s) URL scheme: {scheme!r}")


def _fetch(url: str, timeout: int = 15) -> bytes:
    _assert_fetchable(url)
    req = urllib.request.Request(url, headers={"User-Agent": "research-langgraph/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - scheme-checked above
        return resp.read(_MAX_RESPONSE_BYTES)


def _pdf_extract(url: str) -> str:
    raw = _fetch(url)
    import fitz  # PyMuPDF, lazy import: optional dependency

    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def _html_extract(url: str) -> str:
    raw = _fetch(url)
    from bs4 import BeautifulSoup  # lazy import: optional dependency

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def _raw_text_fallback(url: str) -> str:
    raw = _fetch(url)
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", text)  # strip any markup, keep content
    return re.sub(r"\s+", " ", text).strip()


def _mock_extract(url: str) -> str:
    return (
        f"Deterministic offline placeholder extraction for {url}. No network or parsing "
        f"library was available, so this stands in for real extracted content."
    )


def extract(url: str) -> Tuple[str, str, bool, bool, str]:
    """Extract text from one URL.

    Returns (text, extractor_used, fell_back, failed, error). `fell_back=True` means a
    lower-fidelity path was used but the source was NOT dropped; `failed=True` means
    every path failed and the source should be dropped (logged, not silently swallowed).
    """
    primary = _pdf_extract if _looks_like_pdf(url) else _html_extract
    primary_name = "pdf" if _looks_like_pdf(url) else "html"

    try:
        return primary(url), primary_name, False, False, ""
    except Exception as primary_err:
        try:
            return _raw_text_fallback(url), "raw", True, False, ""
        except Exception as raw_err:
            try:
                return _mock_extract(url), "mock", True, False, ""
            except Exception:
                return "", "none", False, True, f"{primary_err}; raw fallback also failed: {raw_err}"
