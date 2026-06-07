"""
JinaReranker — LlamaIndex BaseNodePostprocessor wrapper around Jina Reranker v2.

Extracted from retriever.py so reranking is a composable LlamaIndex postprocessor
that can plug into any RetrieverQueryEngine.
"""

import logging
import time
from typing import Any, Optional

import requests
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from core.config import (
    JINA_API_KEY,
    JINA_RERANK_MODEL,
    RERANK_TRUNCATE_CHARS,
    TOP_K_FINAL,
)
from core.http import jina_session

_logger = logging.getLogger(__name__)
_RETRY_STATUSES = {429, 500, 502, 503, 504}


class JinaReranker(BaseNodePostprocessor):
    """Jina Reranker v2 as a LlamaIndex BaseNodePostprocessor (shared session, 3-attempt retry)."""

    top_n: int = TOP_K_FINAL
    truncate_chars: int = RERANK_TRUNCATE_CHARS

    _http: requests.Session = PrivateAttr()

    def __init__(
        self,
        top_n: int = TOP_K_FINAL,
        truncate_chars: int = RERANK_TRUNCATE_CHARS,
        **kwargs,
    ) -> None:
        super().__init__(top_n=top_n, truncate_chars=truncate_chars, **kwargs)
        self._http = jina_session

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes

        query = query_bundle.query_str
        docs = [n.node.get_content()[: self.truncate_chars] for n in nodes]

        for attempt in range(3):
            resp = self._http.post(
                "https://api.jina.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {JINA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": JINA_RERANK_MODEL,
                    "query": query,
                    "documents": docs,
                    "top_n": self.top_n,
                },
                timeout=20,
            )
            if resp.status_code in _RETRY_STATUSES and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            break

        reranked: list[NodeWithScore] = []
        for item in resp.json()["results"]:
            nws = nodes[item["index"]]
            score = item["relevance_score"]
            nws.node.metadata["rerank_score"] = score
            reranked.append(NodeWithScore(node=nws.node, score=score))

        if len(reranked) < self.top_n:
            _logger.warning(
                "JinaReranker returned %d results, expected %d (query: %.60s…)",
                len(reranked), self.top_n, query,
            )

        return reranked


class LocalJinaReranker(BaseNodePostprocessor):
    """jina-reranker-v2-base-multilingual running locally on GPU via sentence-transformers."""

    top_n: int = TOP_K_FINAL
    truncate_chars: int = RERANK_TRUNCATE_CHARS

    _model: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = TOP_K_FINAL,
        truncate_chars: int = RERANK_TRUNCATE_CHARS,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(top_n=top_n, truncate_chars=truncate_chars, **kwargs)
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        from core.config import LOCAL_RERANK_MODEL  # noqa: PLC0415
        self._model = CrossEncoder(LOCAL_RERANK_MODEL, trust_remote_code=True, device=device, max_length=512)

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes

        query = query_bundle.query_str
        docs  = [n.node.get_content()[: self.truncate_chars] for n in nodes]
        scores = self._model.predict(
            [(query, doc) for doc in docs], batch_size=len(docs), show_progress_bar=False
        )
        ranked = sorted(zip(scores, nodes), key=lambda x: x[0], reverse=True)[: self.top_n]

        reranked = []
        for score, nws in ranked:
            nws.node.metadata["rerank_score"] = float(score)
            reranked.append(NodeWithScore(node=nws.node, score=float(score)))
        return reranked


def get_reranker(top_n: int = TOP_K_FINAL) -> BaseNodePostprocessor:
    """Return the active reranker: local GPU if available, Jina API otherwise."""
    from core.config import USE_LOCAL_RERANK, EMBED_DEVICE  # noqa: PLC0415
    if USE_LOCAL_RERANK:
        return LocalJinaReranker(top_n=top_n, device=EMBED_DEVICE)
    return JinaReranker(top_n=top_n)
