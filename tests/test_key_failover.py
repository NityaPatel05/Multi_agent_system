import pytest

from research_langgraph.routing import api_keys
from research_langgraph.routing.model_router import ProviderError


class RateLimited(Exception):
    pass


class BadRequest(Exception):
    pass


def classify(e):
    return 60.0 if isinstance(e, RateLimited) else None


def test_parse_keys_splits_strips_dedupes(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", " a, b ,,a,c ")
    assert api_keys.parse_keys("GROQ_API_KEY") == ["a", "b", "c"]


def test_moves_to_next_key_on_rate_limit():
    pool, calls = api_keys.KeyPool(), []

    def attempt(k):
        calls.append(k)
        if k == "k0":
            raise RateLimited()
        return k

    assert api_keys.call_with_failover(pool, ["k0", "k1"], attempt, classify, "groq") == "k1"
    # exhausted key is on cooldown: the next call must not waste a request on it
    calls.clear()
    api_keys.call_with_failover(pool, ["k0", "k1"], attempt, classify, "groq")
    assert calls == ["k1"]


def test_non_key_error_does_not_rotate():
    calls = []

    def attempt(k):
        calls.append(k)
        raise BadRequest("model_not_found")

    with pytest.raises(ProviderError):
        api_keys.call_with_failover(api_keys.KeyPool(), ["k0", "k1", "k2"], attempt, classify, "groq")
    assert len(calls) == 1


def test_all_exhausted_error_names_positions_not_keys():
    def attempt(k):
        raise RateLimited()

    with pytest.raises(ProviderError) as ei:
        api_keys.call_with_failover(api_keys.KeyPool(), ["SECRET0", "SECRET1"], attempt, classify, "groq")
    msg = str(ei.value)
    assert "key#0" in msg and "key#1" in msg and "SECRET" not in msg


def test_transient_error_retries_same_key_without_rotating(monkeypatch):
    monkeypatch.setattr(api_keys, "TRANSIENT_BACKOFF_S", 0)
    calls = []

    def attempt(k):
        calls.append(k)
        if len(calls) < 3:
            raise BadRequest("503 overloaded")
        return "ok"

    out = api_keys.call_with_failover(api_keys.KeyPool(), ["k0", "k1"], attempt, classify, "gemini",
                                      is_transient=lambda e: "503" in str(e))
    assert out == "ok" and calls == ["k0", "k0", "k0"]


def test_empty_key_list():
    with pytest.raises(ProviderError):
        api_keys.call_with_failover(api_keys.KeyPool(), [], lambda k: k, classify, "groq")
