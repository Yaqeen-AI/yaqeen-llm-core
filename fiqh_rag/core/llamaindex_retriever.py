"""
LlamaIndex BaseRetriever adapter over FiqhRetriever + JinaReranker.

Converts core.retriever.Result <-> llama_index.core.schema.NodeWithScore
so the retrieval layer speaks LlamaIndex's standard document interface.
Reranking is applied here as a JinaReranker (BaseNodePostprocessor) step.
"""
from __future__ import annotations

from uuid import uuid4

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from core.arabic_utils import detect_mazhabs, detect_fiqh_topic
from core.config import TOP_K_FETCH, TOP_K_FINAL
from core.reranker import get_reranker
from core.retriever import FiqhRetriever, Result
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from core.schema import NodeMetadata, QdrantPayload


def result_to_node(result: Result, index: int) -> NodeWithScore:
    """Convert a Result dataclass to a LlamaIndex NodeWithScore."""
    node = TextNode(
        id_=str(uuid4()),
        text=result.chunk_text,
        metadata=NodeMetadata(
            volume_id=result.volume_id,
            book_page=result.book_page,
            chunk_page=result.chunk_page,
            source_url=result.source_url,
            mazhabs=result.mazhabs or [],
            fiqh_topic=result.fiqh_topic or "",
            qdrant_score=result.qdrant_score,
            rerank_score=result.rerank_score,
            short_ref=result.short_ref(),
            rank=index,
        ),
        excluded_embed_metadata_keys=["qdrant_score", "rerank_score", "rank"],
        excluded_llm_metadata_keys=["qdrant_score", "rerank_score", "rank", "source_url"],
    )
    return NodeWithScore(node=node, score=result.qdrant_score)


def nodes_to_results(nodes: list[NodeWithScore]) -> list[Result]:
    """Reverse conversion: NodeWithScore → Result, for generator compatibility."""
    results = []
    for nws in nodes:
        m = nws.node.metadata
        results.append(Result(
            chunk_text=nws.node.text,
            volume_id=m["volume_id"],
            book_page=m["book_page"],
            chunk_page=m["chunk_page"],
            source_url=m["source_url"],
            qdrant_score=m["qdrant_score"],
            rerank_score=m["rerank_score"],
            mazhabs=m["mazhabs"] or None,
            fiqh_topic=m.get("fiqh_topic") or None,
        ))
    return results


class FiqhLlamaRetriever(BaseRetriever):
    """
    LlamaIndex BaseRetriever wrapping FiqhRetriever + JinaReranker.
    Hybrid search (BM25 + Qdrant RRF) lives in FiqhRetriever;
    reranking is applied here as a BaseNodePostprocessor step.
    """

    _retriever: FiqhRetriever = PrivateAttr()
    _reranker: BaseNodePostprocessor = PrivateAttr()
    _top_k_fetch: int = PrivateAttr()
    _top_k_final: int = PrivateAttr()

    def __init__(
        self,
        top_k_fetch: int = TOP_K_FETCH,
        top_k_final: int = TOP_K_FINAL,
    ) -> None:
        super().__init__()
        self._retriever = FiqhRetriever()
        self._reranker = get_reranker(top_n=top_k_final)
        self._top_k_fetch = top_k_fetch
        self._top_k_final = top_k_final

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        mazhab_filter = detect_mazhabs(query_bundle.query_str) or None
        topic_filter = detect_fiqh_topic(query_bundle.query_str)
        candidates: list[Result] = self._retriever.retrieve(
            query_bundle.query_str,
            top_k_fetch=self._top_k_fetch,
            mazhab_filter=mazhab_filter,
            topic_filter=topic_filter,
        )
        raw_nodes = [result_to_node(r, i) for i, r in enumerate(candidates)]
        return self._reranker.postprocess_nodes(raw_nodes, query_bundle)

    def retrieve_with_filter(
        self,
        query: str,
        mazhab_filter: list[str] | None = None,
        topic_filter: list[str] | None = None,
        precomputed_embedding: list[float] | None = None,
    ) -> list[NodeWithScore]:
        """Retrieve with explicit filters — used by the LangGraph pipeline.

        Pass precomputed_embedding to avoid re-embedding when the cache layer
        already computed the query vector during its Tier-2 lookup.
        topic_filter is a list — single topic narrows search; two topics run a
        union search across both slices (tie case from detect_fiqh_topic).
        """
        candidates: list[Result] = self._retriever.retrieve(
            query,
            top_k_fetch=self._top_k_fetch,
            mazhab_filter=mazhab_filter,
            topic_filter=topic_filter,
            precomputed_embedding=precomputed_embedding,
        )
        raw_nodes = [result_to_node(r, i) for i, r in enumerate(candidates)]
        query_bundle = QueryBundle(query_str=query)
        return self._reranker.postprocess_nodes(raw_nodes, query_bundle)
