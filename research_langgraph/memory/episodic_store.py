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
import json
import os
import re
from pathlib import Path
from typing import List

_LOCAL_STORE_PATH = Path(os.environ.get("EPISODIC_STORE_PATH", ".episodic_store.json"))
_EMBED_DIM = 256


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


def _embed(text: str) -> List[float]:
    provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
    if provider == "gemini":
        try:
            import google.generativeai as genai

            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            resp = genai.embed_content(model="models/text-embedding-004", content=text)
            return resp["embedding"]
        except Exception:
            pass  # self-heal to the offline embedding rather than fail the store
    return _hash_embed(text)


class _LocalStore:
    def __init__(self, path: Path):
        self.path = path
        self._records = json.loads(path.read_text()) if path.exists() else []

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

        self.client = QdrantClient(url=url)
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
        hits = self.client.search(self.collection, query_vector=embedding, limit=top_k)
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
