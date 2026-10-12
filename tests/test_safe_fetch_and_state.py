import pytest

from research_langgraph.agents.extraction_merge import extraction_merge
from research_langgraph.agents.search_merge import search_merge
from research_langgraph.agents.validator_agent import validator_agent
from research_langgraph.tools import mcp_extract_adapter, safe_fetch


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost:8000/", "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/", "http://192.168.1.1/", "http://[::1]/", "http://[::ffff:127.0.0.1]/",
    "http://0.0.0.0/", "file:///etc/passwd", "ftp://example.com/x", "http://user:pw@example.com/",
])
def test_unsafe_urls_are_refused(url):
    with pytest.raises(safe_fetch.UnsafeURLError):
        safe_fetch.fetch(url)


def test_redirect_to_private_address_is_revalidated(monkeypatch):
    """A public-looking first hop that 302s to an internal address must be refused."""
    import httpx

    monkeypatch.setattr(safe_fetch, "resolve_public_ip",
                        lambda host, port: "8.8.8.8" if host == "public.example" else
                        (_ for _ in ()).throw(safe_fetch.UnsafeURLError("non-public")))

    def handler(request):
        return httpx.Response(302, headers={"location": "http://internal.example/secret"})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(safe_fetch.UnsafeURLError):
        safe_fetch.fetch("http://public.example/")


def test_live_mode_failed_fetch_is_dropped_not_faked(monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "0")
    text, extractor, fell_back, failed, error = mcp_extract_adapter.extract("http://127.0.0.1/x")
    assert failed and text == "" and extractor == "none" and error


def test_extracted_text_has_no_control_chars():
    assert mcp_extract_adapter._sanitize("a\x00b\x07c\nd\te") == "abc\nd\te"


def test_offline_mode_uses_placeholder(monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "1")
    _, extractor, _, failed, _ = mcp_extract_adapter.extract("http://127.0.0.1/x")
    assert extractor == "mock" and not failed


def test_search_merge_keeps_all_dimensions_and_resets_broaden():
    raw = [
        {"id": "a", "url": "http://x/1", "dimension": "D1", "score": 0.5},
        {"id": "a", "url": "http://x/1", "dimension": "D2", "score": 0.9},
    ]
    out = search_merge({"raw_search_results": raw})
    assert out["sources"][0]["dimensions"] == ["D1", "D2"]
    assert out["sources"][0]["dimension"] == "D2"  # best-scoring hit
    assert out["is_broaden"] is False


def test_validator_only_judges_new_sources_and_never_loses_ids():
    s = [{"id": f"s{i}", "url": f"http://ok{i}.example/", "title": "", "snippet": ""} for i in range(3)]
    r1 = validator_agent({"sources": s[:2]})
    assert {x["id"] for x in r1["validated_sources"]} == {"s0", "s1"}
    state = {"sources": s, "validated_sources": r1["validated_sources"],
             "rejection_reasons": r1["rejection_reasons"]}
    r2 = validator_agent(state)
    assert [x["id"] for x in r2["validated_sources"]] == ["s0", "s1", "s2"]
    # nothing new -> no model call, no audit/tokens
    r3 = validator_agent({**state, "sources": s[:2]})
    assert r3["audit_log"] == [] and r3["token_usage"] == []


def test_extraction_merge_drops_docs_of_unvalidated_sources():
    state = {
        "validated_sources": [{"id": "keep"}],
        "extracted_docs": [
            {"source_id": "keep", "dimension": "A", "text": "t"},
            {"source_id": "gone", "dimension": "A", "text": "t"},
            {"source_id": "keep", "dimension": "A", "text": "t2"},
        ],
    }
    docs = extraction_merge(state)["documents"]
    assert [d["source_id"] for d in docs] == ["keep"] and docs[0]["text"] == "t2"
