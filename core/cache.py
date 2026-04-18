"""
Two-tier response cache.

Tier 1 — Redis      : exact SHA-256 hash match, O(1) lookup, allkeys-lru eviction
Tier 2 — Qdrant     : Jina v3 semantic similarity, cosine ≥ SEMANTIC_THRESHOLD
"""

import hashlib
import re
import uuid

import requests
import redis as _redis
from qdrant_client.models import Distance, PointStruct, VectorParams

from core.arabic_utils import normalize
from core.config import (
    CACHE_COLLECTION, EMBED_DIM, JINA_API_KEY, JINA_EMBED_MODEL,
    REDIS_DB, REDIS_HOST, REDIS_MAX_MEMORY,
    REDIS_PORT, SEMANTIC_THRESHOLD,
)
from core.qdrant_singleton import cache_client


class TwoTierCache:
    """
    Usage:
        cache = TwoTierCache()
        answer = cache.get(query)          # None on miss
        if answer is None:
            answer = ...generate...
            cache.set(query, answer)
    """

    def __init__(self) -> None:
        self._r: _redis.Redis | None = None
        self._q: QdrantClient | None = None
        self._init_redis()
        self._init_qdrant()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_redis(self) -> None:
        try:
            r = _redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                socket_connect_timeout=2, decode_responses=True,
            )
            r.ping()
            self._r = r
            try:
                r.config_set("maxmemory", REDIS_MAX_MEMORY)
                r.config_set("maxmemory-policy", "allkeys-lru")
            except _redis.ResponseError:
                pass  # managed Redis (e.g. Redis Cloud) blocks CONFIG SET
            print("[Cache] Tier 1 (Redis) connected — LRU eviction active.")
        except Exception as exc:
            print(f"[Cache] Tier 1 disabled — Redis unavailable: {exc}")

    def _init_qdrant(self) -> None:
        try:
            self._q = cache_client()
            if not self._q.collection_exists(CACHE_COLLECTION):
                self._q.create_collection(
                    collection_name=CACHE_COLLECTION,
                    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
                )
            print(f"[Cache] Tier 2 (Qdrant semantic cache) ready — threshold {SEMANTIC_THRESHOLD:.0%}.")
        except Exception as exc:
            print(f"[Cache] Tier 2 disabled — {exc}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(query: str) -> str:
        """Arabic-aware normalisation: diacritics, alef variants, punctuation, lowercase."""
        text = normalize(query)
        text = text.lower()
        text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _embed(self, text: str) -> list[float] | None:
        if not JINA_API_KEY:
            return None
        try:
            resp = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": JINA_EMBED_MODEL,
                    "input": [text],
                    "dimensions": EMBED_DIM,
                    "task": "retrieval.query",
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception:
            return None

    # ── Public API ───────────────────────────────────────────────────────────

    def get(self, query: str) -> str | None:
        """Return cached answer (str) or None on miss."""
        normalized = self._normalize(query)
        key = self._hash(normalized)

        # Tier 1 — exact hash match (microseconds)
        if self._r:
            try:
                hit = self._r.get(key)
                if hit:
                    return hit
            except Exception:
                pass

        # Tier 2 — semantic similarity via Jina v3 + Qdrant cosine search
        if self._q:
            vec = self._embed(normalized)
            if vec:
                try:
                    hits = self._q.query_points(
                        collection_name=CACHE_COLLECTION,
                        query=vec,
                        limit=1,
                        score_threshold=SEMANTIC_THRESHOLD,
                        with_payload=True,
                    ).points
                    if hits:
                        answer = hits[0].payload["answer"]
                        # Promote semantic hit into Tier 1 so next lookup is O(1)
                        if self._r:
                            try:
                                self._r.set(key, answer)
                            except Exception:
                                pass
                        return answer
                except Exception:
                    pass

        return None

    def set(self, query: str, answer: str) -> None:
        """Store answer in both tiers."""
        normalized = self._normalize(query)
        key = self._hash(normalized)

        # Tier 1
        if self._r:
            try:
                self._r.set(key, answer)
            except Exception:
                pass

        # Tier 2
        if self._q:
            vec = self._embed(normalized)
            if vec:
                try:
                    self._q.upsert(
                        collection_name=CACHE_COLLECTION,
                        points=[PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vec,
                            payload={
                                "normalized_query": normalized,
                                "original_query": query,
                                "answer": answer,
                            },
                        )],
                    )
                except Exception:
                    pass
