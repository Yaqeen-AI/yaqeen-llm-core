"""
quran_rag/retriever.py

Query-time retriever for the Quranic RAG.

Stack:
  - Embedding:  Jina v3 via REST API (llama-index-embeddings-jinaai)
  - Index:      Qdrant Cloud (existing collection, read-only at query time)
  - Search:     Hybrid dense + sparse (Qdrant/bm25 fastembed)
  - Reranking:  Jina reranker v2 via REST API (llama-index-postprocessor-jinaai-rerank)

No local GPU required.

Public API:
    get_quran_retriever() -> QuranRetriever   — module-level singleton factory
    QuranRetriever.search(...)               — async search returning NodeWithScore list
"""
from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Union

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qdrant_models

from .config import QuranRagConfig, get_quran_config

logger = logging.getLogger(__name__)


class QuranRetriever:
    """
    Lazy-initialized retriever for the Quranic corpus.

    The Qdrant collection is expected to already exist (indexed via notebooks/scripts).
    This class only performs read operations.

    Lifecycle:
        retriever = QuranRetriever(cfg)
        retriever.setup()     # call once at app startup
        nodes = await retriever.search(query="...")
    """

    def __init__(self, cfg: QuranRagConfig) -> None:
        self.cfg = cfg
        self._index: Optional[VectorStoreIndex] = None
        self._embed_model: Optional[JinaEmbedding] = None
        self._reranker: Optional[JinaRerank] = None
        self._client: Optional[QdrantClient] = None
        self._executor = ThreadPoolExecutor(max_workers=4)

    def setup(self) -> None:
        """
        Initialize Qdrant client, LlamaIndex vector store, embedding model,
        and cached reranker. Called once at application startup.
        """
        logger.info("Initialising Quran retriever…")

        # 1. Embedding model in query mode
        self._embed_model = JinaEmbedding(
            api_key=self.cfg.jina_api_key,
            model=self.cfg.jina_embedding_model,
            task="retrieval.query",
            embed_batch_size=1024,
        )
        Settings.embed_model = self._embed_model

        # 2. Qdrant clients with timeouts
        sync_client = QdrantClient(
            url=self.cfg.qdrant_url,
            api_key=self.cfg.qdrant_api_key,
            timeout=30,
        )
        self._client = sync_client
        async_client = AsyncQdrantClient(
            url=self.cfg.qdrant_url,
            api_key=self.cfg.qdrant_api_key,
            timeout=30,
        )

        # 3. Vector store — text_key points to the formatted tafsir field
        vector_store = QdrantVectorStore(
            client=sync_client,
            aclient=async_client,
            collection_name=self.cfg.quran_collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
            dense_vector_name=self.cfg.dense_vector_name,
            sparse_vector_name=self.cfg.sparse_vector_name,
            text_key=self.cfg.text_key,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

        # Ensure payload indexes exist
        for field, schema in [
            ("chunk_id", "keyword"),
            ("edition", "keyword"),
            ("theme", "keyword"),
            ("ayah_text", "text"),
            ("ruku", "integer"),
            ("hizb_quarter", "integer"),
            ("surah", "integer"),
            ("juz", "integer"),
            ("revelation_type", "keyword"),
        ]:
            try:
                sync_client.create_payload_index(
                    collection_name=self.cfg.quran_collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass

        # 4. Cache the reranker — don't recreate per query!
        self._reranker = JinaRerank(
            api_key=self.cfg.jina_api_key,
            model=self.cfg.jina_reranker_model,
            top_n=self.cfg.rerank_top_n,
        )

        logger.info(
            "Quran retriever ready — collection: %s | text_key: %s",
            self.cfg.quran_collection_name,
            self.cfg.text_key,
        )

    @property
    def index(self) -> VectorStoreIndex:
        if self._index is None:
            raise RuntimeError("QuranRetriever.setup() has not been called. Call it during app startup.")
        return self._index

    # ── Metadata filter builder ──────────────────────────────────────────────

    def _build_filters(
        self,
        surah: Optional[Union[int, List[int]]] = None,
        revelation_type: Optional[Union[str, List[str]]] = None,
        juz: Optional[Union[int, List[int]]] = None,
        ruku: Optional[Union[int, List[int]]] = None,
        hizb_quarter: Optional[Union[int, List[int]]] = None,
        theme: Optional[Union[str, List[str]]] = None,
        edition: Optional[Union[str, List[str]]] = None,
        ayah_text: Optional[Union[str, List[str]]] = None,
    ) -> Optional[MetadataFilters]:
        filters = []

        for key, value in [
            ("surah", surah),
            ("juz", juz),
            ("ruku", ruku),
            ("hizb_quarter", hizb_quarter),
        ]:
            if value is not None:
                op = FilterOperator.IN if isinstance(value, list) else FilterOperator.EQ
                filters.append(MetadataFilter(key=key, value=value, operator=op))

        for key, value in [
            ("revelation_type", revelation_type),
            ("theme", theme),
            ("edition", edition),
        ]:
            if value is not None:
                op = FilterOperator.IN if isinstance(value, list) else FilterOperator.EQ
                filters.append(MetadataFilter(key=key, value=value, operator=op))

        if ayah_text is not None:
            if isinstance(ayah_text, list):
                for text in ayah_text:
                    filters.append(MetadataFilter(key="ayah_text", value=text, operator=FilterOperator.TEXT_MATCH_INSENSITIVE))
            else:
                filters.append(MetadataFilter(key="ayah_text", value=ayah_text, operator=FilterOperator.TEXT_MATCH_INSENSITIVE))

        return MetadataFilters(filters=filters, condition=FilterCondition.AND) if filters else None



    # ── Search ───────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        surah: Optional[Union[int, List[int]]] = None,
        revelation_type: Optional[Union[str, List[str]]] = None,
        juz: Optional[Union[int, List[int]]] = None,
        ruku: Optional[Union[int, List[int]]] = None,
        hizb_quarter: Optional[Union[int, List[int]]] = None,
        theme: Optional[Union[str, List[str]]] = None,
        edition: Optional[Union[str, List[str]]] = None,
        ayah_text: Optional[Union[str, List[str]]] = None,
        similarity_top_k: int = 10,          # ← agent-tunable: candidates from vector store
        rerank_top_n: int = 10,              # ← agent-tunable: final results after rerank
        skip_rerank: bool = False,
        include_parent_context: bool = True,
    ) -> List[NodeWithScore]:
        """
        Hybrid (dense + sparse) retrieval → Jina reranking → optional parent context.

        Args:
            query:              Arabic search query.
            mode:               "hybrid" | "dense" | "sparse"
            surah:              Filter by surah number(s).
            revelation_type:    Filter by "Meccan" | "Medinan".
            juz:                Filter by juz number(s).
            ruku:               Filter by ruku number(s).
            hizb_quarter:       Filter by hizb quarter number(s).
            theme:              Filter by thematic category.
            edition:            Filter by tafsir edition (e.g., "ar.ibnkathir").
            ayah_text:          Full-text ayah filter for direct verse lookups.
            similarity_top_k:   How many candidates to fetch from Qdrant before reranking.
            rerank_top_n:       How many results to return after reranking.
            skip_rerank:        Bypass reranker for max speed.
            include_parent_context: Fetch and attach parent summary for child chunks.

        Returns:
            Sorted list of NodeWithScore (best first).
        """
        mode_map = {"dense": "default", "sparse": "sparse", "hybrid": "hybrid"}
        qs_mode = mode_map.get(mode.lower(), "hybrid")

        filters = self._build_filters(
            surah=surah,
            revelation_type=revelation_type,
            juz=juz,
            ruku=ruku,
            hizb_quarter=hizb_quarter,
            theme=theme,
            edition=edition,
            ayah_text=ayah_text,
        )

        # 1. Retrieve candidates — agent controls fetch size
        retriever = self.index.as_retriever(
            similarity_top_k=similarity_top_k,   # ← agent-tunable
            vector_store_query_mode=qs_mode,
            filters=filters,
        )

        logger.debug(
            "Quran search | query=%r | mode=%s | sim_k=%d | rerank_n=%d | skip=%s",
            query, mode, similarity_top_k, rerank_top_n, skip_rerank,
        )
        candidates: List[NodeWithScore] = await retriever.aretrieve(query)

        if not candidates:
            return []

        # 2. Fast path: skip reranking
        if skip_rerank:
            results = candidates[:rerank_top_n]
            return self._attach_parent_context(results) if include_parent_context else results

        # 3. Rerank — agent controls output size
        loop = asyncio.get_running_loop()
        reranked = await loop.run_in_executor(
            self._executor,
            self._sync_rerank,
            candidates,
            query,
            rerank_top_n,                      # ← agent-tunable
        )

        if include_parent_context:
            return self._attach_parent_context(reranked)
        return reranked

    def _sync_rerank(
        self,
        candidates: List[NodeWithScore],
        query: str,
        top_n: int,
    ) -> List[NodeWithScore]:
        """Synchronous wrapper for thread-pool execution."""
        self._reranker.top_n = top_n
        return self._reranker.postprocess_nodes(
            nodes=candidates,
            query_bundle=QueryBundle(query_str=query),
        )

    def _attach_parent_context(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        if self._client is None:
            return nodes

        parent_ids = {
            str(node.node.metadata.get("parent_chunk_id", "")).strip()
            for node in nodes
            if node.node.metadata.get("parent_chunk_id")
        }
        if not parent_ids:
            return nodes

        parent_texts = self._fetch_parent_texts(parent_ids)
        enriched: List[NodeWithScore] = []
        for node in nodes:
            parent_id = str(node.node.metadata.get("parent_chunk_id", "")).strip()
            parent_text = parent_texts.get(parent_id, "")
            if not parent_text:
                enriched.append(node)
                continue

            child_text = node.node.get_content(metadata_mode="none")
            if parent_text in child_text:
                enriched.append(node)
                continue

            metadata = dict(node.node.metadata or {})
            enriched_node = TextNode(
                id_=getattr(node.node, "id_", None),
                text=f"{child_text}\n\n[ملخص موضوعي]\n{parent_text}",
                metadata=metadata,
            )
            enriched.append(NodeWithScore(node=enriched_node, score=node.score))
        return enriched

    def _fetch_parent_texts(self, parent_chunk_ids: set[str]) -> dict[str, str]:
        if self._client is None:
            return {}
        try:
            points, _ = self._client.scroll(
                collection_name=self.cfg.quran_collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="chunk_id",
                            match=qdrant_models.MatchAny(any=list(parent_chunk_ids)),
                        )
                    ]
                ),
                limit=len(parent_chunk_ids),
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            logger.debug("Could not fetch Quran parent chunks.", exc_info=True)
            return {}

        parent_texts: dict[str, str] = {}
        for point in points:
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id", "")).strip()
            if not chunk_id:
                continue
            text = _payload_text(payload, self.cfg.text_key)
            if text:
                parent_texts[chunk_id] = text
        return parent_texts

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Graceful shutdown — call during app teardown."""
        self._executor.shutdown(wait=True)
        logger.info("Quran retriever shut down.")


def _payload_text(payload: dict[str, Any], text_key: str) -> str:
    text = payload.get(text_key)
    if isinstance(text, str) and text.strip():
        return text.strip()
    node_content = payload.get("_node_content")
    if isinstance(node_content, str):
        try:
            parsed = json.loads(node_content)
            text = parsed.get("text")
            return text.strip() if isinstance(text, str) else ""
        except json.JSONDecodeError:
            return ""
    return ""


# ── Module-level singleton ───────────────────────────────────────────────────

_instance: Optional[QuranRetriever] = None


def get_quran_retriever() -> QuranRetriever:
    """
    Return the application-wide QuranRetriever singleton.
    Call setup() exactly once during FastAPI lifespan startup.
    """
    global _instance
    if _instance is None:
        _instance = QuranRetriever(get_quran_config())
    return _instance
