"""Episodic memory — vector DB of the project's OWN past runs only.

Hard rule (CLAUDE.md / architecture.md §4): this is NOT a RAG layer over the live web —
agentic search stays agentic. This store exists for two things only: (1) new requests on
similar topics can retrieve prior validated sources first, cutting redundant search
calls, and (2) incremental "update this report" re-runs can diff against the last stored
run instead of starting over. It must never be wired into the live Search/Extract path.

Backed by Qdrant when QDRANT_URL is configured; falls back to a local on-disk JSON store
with a lightweight hashing "embedding" otherwise, so this stays usable without standing up
infrastructure for local development.
"""

from __future__ import annotations

import hashlib
import logging
import json
import os
import re
from pathlib import Path
from typing import List

from research_langgraph.routing.api_keys import KeyPool

_LOCAL_STORE_PATH = Path(os.environ.get("EPISODIC_STORE_PATH", ".episodic_store.json"))
_EMBED_DIM = 256
logger = logging.getLogger(__name__)


def _hash_embed(text: str) -> List[float]:
    """Deterministic, dependency-free stand-in for a real embedding model — good enough
    for local dev/similarity search. Swap in a real embedding client by setting
    EMBEDDING_PROVIDER for production use."""
    vec = [0.0] * _EMBED_DIM
    for token in re.findall(r"\w+", text.lower()):
        idx = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % _EMBED_DIM
        vec[idx] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


_embed_pool = KeyPool()


def _embed(text: str) -> List[float]:
    provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
    if provider == "gemini":
        try:
            from google import genai
            from google.genai import errors as genai_errors
            from google.genai import types as genai_types

            from research_langgraph.routing.api_keys import call_with_failover, parse_keys
            from research_langgraph.routing.model_tiers import EMBEDDING_BINDING

            def attempt(key: str) -> List[float]:
                resp = genai.Client(api_key=key).models.embed_content(
                    model=EMBEDDING_BINDING["model"],
                    contents=text,
                    config=genai_types.EmbedContentConfig(output_dimensionality=_EMBED_DIM),
                )
                vec = list(resp.embeddings[0].values)
                norm = sum(v * v for v in vec) ** 0.5 or 1.0  # truncated dims aren't unit-length
                return [v / norm for v in vec]

            def classify(e: Exception):
                code = getattr(e, "code", None) if isinstance(e, genai_errors.APIError) else None
                return 60.0 if code in (429, 401, 403) else None

            return call_with_failover(
                _embed_pool, parse_keys("GOOGLE_API_KEY", "GEMINI_API_KEY"), attempt, classify, "gemini"
            )
        except Exception as e:
            # self-heal to the offline embedding rather than fail the store, but say so:
            # hash vectors and Gemini vectors are not comparable with each other.
            logger.warning("Gemini embedding failed (%s); using offline hash embedding", e)
    return _hash_embed(text)


class _LocalStore:
    def __init__(self, path: Path):
        self.path = path
        text = path.read_text().strip() if path.exists() else ""
        self._records = json.loads(text) if text else []

    def upsert(self, record: dict) -> None:
        self._records = [r for r in self._records if r["run_id"] != record["run_id"]]
        self._records.append(record)
        self.path.write_text(json.dumps(self._records))

    def search(self, embedding: List[float], top_k: int) -> List[dict]:
        scored = [(_cosine(embedding, r["embedding"]), r) for r in self._records]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [r for _, r in scored[:top_k]]


class _QdrantStore:
    def __init__(self, url: str, collection: str = "research_runs"):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection, vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE)
            )

    def upsert(self, record: dict) -> None:
        from qdrant_client.models import PointStruct

        self.client.upsert(
            self.collection,
            points=[PointStruct(id=record["run_id"], vector=record["embedding"], payload=record)],
        )

    def search(self, embedding: List[float], top_k: int) -> List[dict]:
        hits = self.client.query_points(self.collection, query=embedding, limit=top_k).points
        return [h.payload for h in hits]


def _get_store():
    url = os.environ.get("QDRANT_URL")
    if url:
        try:
            return _QdrantStore(url)
        except Exception:
            pass  # self-heal to the local store rather than fail the run
    return _LocalStore(_LOCAL_STORE_PATH)


def store_run(run_id: str, topic: str, sources: list, final_report: str, evidence_map: dict) -> None:
    """Persist a completed run to episodic memory, keyed by an embedding of the topic."""
    store = _get_store()
    record = {
        "run_id": run_id,
        "topic": topic,
        "embedding": _embed(topic),
        "sources": sources,
        "final_report": final_report,
        "evidence_map": evidence_map,
    }
    store.upsert(record)


def retrieve_similar_runs(topic: str, top_k: int = 3) -> List[dict]:
    """Retrieve prior runs on similar topics — used to seed a new run with already-
    validated sources (cuts redundant search calls) or as the base for an incremental
    'update this report' diff re-run. Never used in the live web-search path."""
    store = _get_store()
    return store.search(_embed(topic), top_k)
