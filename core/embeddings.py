"""
JinaEmbedding — LlamaIndex BaseEmbedding wrapper around Jina Embeddings v3 REST API.

Single source of truth for all Jina embedding calls in the project;
eliminates the duplicate _embed() functions that previously lived in
retriever.py and cache.py.
"""

import asyncio
import time
from typing import Any

import requests
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

from core.http import jina_session

_RETRY_STATUSES = {429, 500, 502, 503, 504}


class JinaEmbedding(BaseEmbedding):
    """Jina Embeddings v3 via REST API (shared session, 3-attempt retry)."""

    _http: requests.Session = PrivateAttr()

    def __init__(self, **kwargs) -> None:
        super().__init__(model_name="jina-embeddings-v3", **kwargs)
        self._http = jina_session

    # ------------------------------------------------------------------
    # Required abstract implementations
    # ------------------------------------------------------------------

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._call_jina([query], task="retrieval.query")[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._call_jina([text], task="retrieval.passage")[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._call_jina(texts, task="retrieval.passage")

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)

    # ------------------------------------------------------------------
    # Internal HTTP helper (called directly by ingest.py for batch work)
    # ------------------------------------------------------------------

    def _call_jina(self, texts: list[str], *, task: str) -> list[list[float]]:
        from core.config import EMBED_DIM, JINA_API_KEY, JINA_EMBED_MODEL  # lazy to avoid circular
        if not JINA_API_KEY:
            raise RuntimeError("JINA_API_KEY not set — add it to .env")

        for attempt in range(3):
            resp = self._http.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": JINA_EMBED_MODEL,
                    "input": texts,
                    "dimensions": EMBED_DIM,
                    "task": task,
                },
                timeout=15,
            )
            if resp.status_code in _RETRY_STATUSES and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in data]

        resp.raise_for_status()  # final attempt already exhausted above
        return []  # unreachable but satisfies type checker


class LocalJinaEmbedding(BaseEmbedding):
    """jina-embeddings-v3 running locally on GPU via sentence-transformers."""

    _model: Any = PrivateAttr()

    def __init__(self, device: str = "cuda", **kwargs) -> None:
        super().__init__(model_name="jina-embeddings-v3", **kwargs)
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        from core.config import LOCAL_EMBED_MODEL  # noqa: PLC0415
        self._model = SentenceTransformer(LOCAL_EMBED_MODEL, trust_remote_code=True, device=device)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._model.encode(
            query, task="retrieval.query", normalize_embeddings=True
        ).tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._model.encode(
            text, task="retrieval.passage", normalize_embeddings=True
        ).tolist()

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts, task="retrieval.passage", normalize_embeddings=True,
            batch_size=32, show_progress_bar=False,
        ).tolist()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._get_text_embedding, text)


def get_embedding_model() -> BaseEmbedding:
    """Return the active embedding model: local GPU if available, Jina API otherwise."""
    from core.config import USE_LOCAL_EMBED, EMBED_DEVICE  # noqa: PLC0415
    if USE_LOCAL_EMBED:
        return LocalJinaEmbedding(device=EMBED_DEVICE)
    return JinaEmbedding()
