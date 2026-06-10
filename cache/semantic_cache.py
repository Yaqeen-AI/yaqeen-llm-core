from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Protocol

from orchestrator.models import AskResponse

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]:
        """Return a dense query embedding."""


class JinaEmbeddingProvider:
    def __init__(self, api_key: str | None = None, model: str = "jina-embeddings-v3") -> None:
        self.api_key = api_key or os.getenv("JINA_API_KEY", "")
        self.model = model

    async def embed(self, text: str) -> list[float]:
        if not self.api_key:
            raise RuntimeError("JINA_API_KEY is required for semantic cache embeddings.")
        return await asyncio.to_thread(self._embed_sync, text)

    def _embed_sync(self, text: str) -> list[float]:
        import urllib.request

        payload = json.dumps(
            {"model": self.model, "task": "retrieval.query", "input": [text]},
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.jina.ai/v1/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        return [float(value) for value in data["data"][0]["embedding"]]


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    key: str | None = None
    response: AskResponse | None = None
    similarity: float = 0.0


class SemanticCache:
    def __init__(
        self,
        redis_url: str | None = None,
        threshold: float | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        namespace: str = "yaqeen:semantic_cache",
    ) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.threshold = threshold if threshold is not None else float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.86"))
        self.embedding_provider = embedding_provider or JinaEmbeddingProvider()
        self.namespace = namespace
        self._redis: Any | None = None

    async def _client(self) -> Any:
        if self._redis is None:
            try:
                import redis.asyncio as redis  # type: ignore

                self._redis = redis.from_url(self.redis_url, decode_responses=True)
                await self._redis.ping()
            except Exception:
                logger.warning("Redis semantic cache unavailable; continuing without cache.", exc_info=True)
                self._redis = False
        return self._redis

    async def check(self, query: str) -> CacheLookup:
        redis = await self._client()
        if not redis:
            return CacheLookup(hit=False)

        normalized = _normalize(query)
        exact_key = self._key(normalized)
        exact_payload = await redis.get(exact_key)
        if exact_payload:
            return CacheLookup(hit=True, key=exact_key, response=_decode_response(exact_payload), similarity=1.0)

        try:
            query_embedding = await self.embedding_provider.embed(normalized)
        except Exception:
            logger.warning("Semantic cache embedding failed; using cache miss.", exc_info=True)
            return CacheLookup(hit=False)

        best_key: str | None = None
        best_payload: str | None = None
        best_similarity = 0.0
        async for key in redis.scan_iter(f"{self.namespace}:entry:*"):
            payload = await redis.get(key)
            if not payload:
                continue
            data = json.loads(payload)
            similarity = _cosine_similarity(query_embedding, data.get("embedding", []))
            if similarity > best_similarity:
                best_key = key
                best_payload = payload
                best_similarity = similarity

        if best_key and best_payload and best_similarity >= self.threshold:
            return CacheLookup(
                hit=True,
                key=best_key,
                response=_decode_response(best_payload),
                similarity=best_similarity,
            )
        return CacheLookup(hit=False)

    async def store(self, query: str, response: AskResponse) -> str | None:
        redis = await self._client()
        if not redis:
            return None
        normalized = _normalize(query)
        try:
            embedding = await self.embedding_provider.embed(normalized)
        except Exception:
            logger.warning("Skipping semantic cache store because embedding failed.", exc_info=True)
            embedding = []

        key = self._key(normalized)
        payload = {
            "original_query": query,
            "normalized_query": normalized,
            "final_answer": response.answer,
            "citations": [citation.model_dump() for citation in response.citations],
            "metadata": response.metadata,
            "response": response.model_dump(mode="json"),
            "embedding": embedding,
        }
        await redis.set(key, json.dumps(payload, ensure_ascii=False))
        return key

    async def invalidate(self, key: str) -> bool:
        redis = await self._client()
        if not redis:
            return False
        return bool(await redis.delete(key))

    def _key(self, normalized_query: str) -> str:
        digest = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()
        return f"{self.namespace}:entry:{digest}"


def _normalize(query: str) -> str:
    try:
        from utils.kuwain_preprocess_arabic_data import KawnPreprocessor
        return KawnPreprocessor().preprocess(query)
    except Exception as e:
        logger.warning("Could not use KawnPreprocessor for cache string normalization: %s", e)
        return " ".join(query.casefold().split())


def _decode_response(payload: str) -> AskResponse:
    data = json.loads(payload)
    response = data.get("response")
    if response:
        return AskResponse.model_validate(response)
    return AskResponse(
        answer=data.get("final_answer", ""),
        citations=data.get("citations", []),
        sources=[],
        cache_hit=True,
        metadata=data.get("metadata", {}),
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)

