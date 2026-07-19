"""Model routing layer (CLAUDE.md §2, architecture.md §1b).

Hard rule: this file (and model_tiers.py) are the ONLY places a concrete model name may
appear. Every node references an abstract tier ("Small"/"Medium"/"High") via NODE_TIERS;
`call_structured()` resolves the tier to a binding, calls the provider, validates the
response against a Pydantic schema, and self-heals (Groq -> Gemini) on failure.

Runs with zero API keys configured: falls through to each agent's deterministic
`mock_fn`, so the graph stays runnable/testable offline (see tools/ adapters, which use
the same pattern for search/extract). This is what "self-healing tool fallback chain,
not just a circuit breaker" (guardrails layer) looks like applied to model calls.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Callable, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

from research_langgraph.routing.model_tiers import FALLBACK_BINDINGS, TIER_BINDINGS

T = TypeVar("T", bound=BaseModel)

# Per-node abstract tier assignment (architecture.md §1b). Budget modes shift these at
# call time in `_resolve_tier`; NODE_TIERS itself is the *default* mapping.
NODE_TIERS = {
    "planner": "Medium",
    "search_agent": "Small",
    "validator": "Small",
    "extractor": "Small",
    "synthesizer": "High",
    "evidence_verifier": "High",
    "critic": "Medium",
}

_TIER_ORDER = ["Small", "Medium", "High"]

# Budget-mode overrides (architecture.md §1b / "Remaining functionality ideas"):
# "fast" downgrades the expensive holistic nodes; "thorough" upgrades planning/validation.
_BUDGET_DOWNGRADE = {"synthesizer", "evidence_verifier"}
_BUDGET_UPGRADE = {"planner", "validator"}


def _step(tier: str, delta: int) -> str:
    idx = _TIER_ORDER.index(tier)
    idx = max(0, min(len(_TIER_ORDER) - 1, idx + delta))
    return _TIER_ORDER[idx]


def _resolve_tier(node_name: str, budget_mode: str = "balanced") -> str:
    tier = NODE_TIERS[node_name]
    if budget_mode == "fast" and node_name in _BUDGET_DOWNGRADE:
        tier = _step(tier, -1)
    elif budget_mode == "thorough" and node_name in _BUDGET_UPGRADE:
        tier = _step(tier, +1)
    return tier


# --------------------------------------------------------------------------- #
# Concurrency cap — Search/Extract fan out via Send(), so several Small-tier calls fire
# concurrently. A semaphore per provider keeps that under the free-tier requests/min cap.
# --------------------------------------------------------------------------- #
_GROQ_MAX_CONCURRENCY = int(os.environ.get("GROQ_MAX_CONCURRENCY", "4"))
_GEMINI_MAX_CONCURRENCY = int(os.environ.get("GEMINI_MAX_CONCURRENCY", "4"))
_semaphores = {
    "groq": threading.Semaphore(_GROQ_MAX_CONCURRENCY),
    "gemini": threading.Semaphore(_GEMINI_MAX_CONCURRENCY),
}


class ProviderError(RuntimeError):
    pass


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _call_groq(model: str, system: str, user: str) -> Tuple[str, int, int]:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ProviderError("GROQ_API_KEY not set")
    try:
        from groq import Groq  # lazy import: optional dependency
    except ImportError as e:
        raise ProviderError(f"groq package unavailable: {e}") from e

    with _semaphores["groq"]:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    content = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    prompt_tok = getattr(usage, "prompt_tokens", None) or _estimate_tokens(system + user)
    completion_tok = getattr(usage, "completion_tokens", None) or _estimate_tokens(content)
    return content, prompt_tok, completion_tok


def _call_gemini(model: str, system: str, user: str) -> Tuple[str, int, int]:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ProviderError("GOOGLE_API_KEY/GEMINI_API_KEY not set")
    try:
        import google.generativeai as genai  # lazy import: optional dependency
    except ImportError as e:
        raise ProviderError(f"google-generativeai package unavailable: {e}") from e

    with _semaphores["gemini"]:
        genai.configure(api_key=api_key)
        gm = genai.GenerativeModel(
            model,
            system_instruction=system,
            generation_config={"response_mime_type": "application/json", "temperature": 0.2},
        )
        resp = gm.generate_content(user)
    content = resp.text or ""
    meta = getattr(resp, "usage_metadata", None)
    prompt_tok = getattr(meta, "prompt_token_count", None) or _estimate_tokens(system + user)
    completion_tok = getattr(meta, "candidates_token_count", None) or _estimate_tokens(content)
    return content, prompt_tok, completion_tok


_PROVIDER_CALLERS: dict[str, Callable[[str, str, str], Tuple[str, int, int]]] = {
    "groq": _call_groq,
    "gemini": _call_gemini,
}


_ERROR_DETAIL_MAX = 200


def _short(err: object) -> str:
    text = str(err)
    return text if len(text) <= _ERROR_DETAIL_MAX else text[:_ERROR_DETAIL_MAX] + "..."


class RoutedCallResult:
    """Everything a caller needs to append to ResearchState (audit_log / token_usage /
    fallback_history)."""

    def __init__(self, node_name: str, tier: str, provider: str, model: str,
                 prompt_tokens: int, completion_tokens: int, used_mock: bool,
                 recovered_from: Optional[str] = None, failure_reason: Optional[str] = None):
        self.node_name = node_name
        self.tier = tier
        self.provider = provider
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.used_mock = used_mock
        self.recovered_from = recovered_from  # set when the primary provider failed
        self.failure_reason = failure_reason  # short, human-readable cause of the failover

    def token_usage_entry(self) -> dict:
        return {
            "node": self.node_name,
            "tier": self.tier,
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    def audit_entries(self) -> list[dict]:
        entries = [{
            "node": self.node_name,
            "kind": "model_call",
            "name": self.tier,
            "provider": self.provider,
            "params": {"model": self.model, "mock": self.used_mock},
            "ok": True,
            "detail": "mock fallback (no provider available)" if self.used_mock else "",
        }]
        if self.recovered_from:
            entries.append({
                "node": self.node_name,
                "kind": "recovery",
                "name": self.tier,
                "provider": self.provider,
                "params": {},
                "ok": True,
                "detail": f"{self.recovered_from} failed ({self.failure_reason}) -> "
                          f"failed over to {self.provider}",
            })
        return entries

    def fallback_entry(self) -> Optional[dict]:
        """Clean, structured record for ResearchState.fallback_history — what actually
        happened, independent of audit_log's free-text `detail` strings. `None` when the
        call succeeded on the primary provider (nothing to report)."""
        if not self.used_mock and not self.recovered_from:
            return None
        return {
            "node": self.node_name.replace("_", " ").title(),
            "reason": self.failure_reason or "unknown failure",
            "fallback": "Mock" if self.used_mock else self.provider.capitalize(),
        }


def _parse_structured(content: str, schema: Type[T]) -> T:
    """Parse LLM output into `schema`, tolerating common non-strict-JSON quirks that
    smaller/open models frequently produce even under json-mode: literal (unescaped)
    control characters inside long string values (e.g. a markdown report embedded as a
    JSON string), ```json ... ``` fences, and leading/trailing prose around the object.
    """
    try:
        return schema.model_validate_json(content)
    except (ValidationError, json.JSONDecodeError):
        pass

    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]

    # strict=False tolerates raw control characters (literal newlines/tabs) inside
    # string values, which is the #1 way LLM-generated JSON fails strict parsing.
    data = json.loads(text, strict=False)
    return schema.model_validate(data)


def call_structured(
    node_name: str,
    system: str,
    user: str,
    schema: Type[T],
    mock_fn: Callable[[], T],
    budget_mode: str = "balanced",
) -> Tuple[T, RoutedCallResult]:
    """Resolve node -> tier -> (Groq primary, Gemini fallback), validate against `schema`,
    retry once on a validation error with a corrective prompt, and fall back to `mock_fn`
    (deterministic, offline) if no provider is configured or both fail.

    Returns (validated_model, RoutedCallResult) — the caller appends the latter's
    audit/token entries to ResearchState.
    """
    tier = _resolve_tier(node_name, budget_mode)
    primary = TIER_BINDINGS[tier]
    fallback = FALLBACK_BINDINGS[tier]
    schema_hint = (
        f"\n\nRespond with ONLY a single JSON object matching this schema "
        f"(no markdown fences, no commentary):\n{json.dumps(schema.model_json_schema())}"
    )

    def _try(provider_name: str, model: str, corrective: str = "") -> Tuple[str, int, int]:
        caller = _PROVIDER_CALLERS[provider_name]
        return caller(model, system + schema_hint, user + corrective)

    attempts = [("primary", primary), ("fallback", fallback)]
    primary_error: Optional[Exception] = None
    last_error: Optional[Exception] = None
    recovered_from = None
    for label, binding in attempts:
        provider_name, model = binding["provider"], binding["model"]
        try:
            content, ptok, ctok = _try(provider_name, model)
        except Exception as e:  # provider down / rate-limited / missing key
            last_error = e
            if label == "primary":
                primary_error = e
                recovered_from = provider_name
            continue
        try:
            parsed = _parse_structured(content, schema)
        except (ValidationError, json.JSONDecodeError) as ve:
            # one error-corrective retry against the SAME provider that responded
            corrective = f"\n\nYour previous response was invalid ({ve}). Fix it and resend JSON only."
            try:
                content2, ptok2, ctok2 = _try(provider_name, model, corrective)
                parsed = _parse_structured(content2, schema)
                ptok, ctok = ptok + ptok2, ctok + ctok2
            except Exception as e2:
                last_error = e2
                if label == "primary":
                    primary_error = e2
                    recovered_from = provider_name
                continue
        failure_reason = (
            f"{primary['provider']} failed: {_short(primary_error)}"
            if label == "fallback" and primary_error else None
        )
        result = RoutedCallResult(
            node_name=node_name, tier=tier, provider=provider_name, model=model,
            prompt_tokens=ptok, completion_tokens=ctok, used_mock=False,
            recovered_from=recovered_from if label == "fallback" else None,
            failure_reason=failure_reason,
        )
        return parsed, result

    # Both providers unavailable/failing -> deterministic offline mock so the graph still runs.
    parsed = mock_fn()
    failure_reason = f"{primary['provider']} and {fallback['provider']} both failed: {_short(last_error)}"
    result = RoutedCallResult(
        node_name=node_name, tier=tier, provider="mock", model="mock",
        prompt_tokens=_estimate_tokens(user), completion_tokens=_estimate_tokens(str(parsed)),
        used_mock=True, recovered_from=primary["provider"], failure_reason=failure_reason,
    )
    return parsed, result


def fallback_history_entries(result: RoutedCallResult) -> list[dict]:
    """Convenience for agents: `"fallback_history": fallback_history_entries(result)`."""
    entry = result.fallback_entry()
    return [entry] if entry else []


def get_model_for(node_name: str, budget_mode: str = "balanced") -> dict:
    """Convenience accessor matching the shape sketched in architecture.md §1b."""
    tier = _resolve_tier(node_name, budget_mode)
    return TIER_BINDINGS[tier]
