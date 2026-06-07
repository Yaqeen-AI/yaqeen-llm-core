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

import logging
from typing import List, Optional, Union

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
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

    def setup(self) -> None:
        """
        Initialize Qdrant client, LlamaIndex vector store, and embedding model.
        Called once at application startup (FastAPI lifespan).
        """
        logger.info("Initialising Quran retriever…")

        # Jina embedding model in query mode (faster, no passage-level prefix)
        self._embed_model = JinaEmbedding(
            api_key=self.cfg.jina_api_key,
            model=self.cfg.jina_embedding_model,
            task="retrieval.query",
            embed_batch_size=16,
        )

        # Set globally so LlamaIndex retriever picks it up.
        # If other RAGs also use LlamaIndex + Jina v3, this is fine (same model).
        # If they use a different model, move this to a per-call context.
        Settings.embed_model = self._embed_model

        # Qdrant clients — sync for setup, async for query
        sync_client = QdrantClient(
            url=self.cfg.qdrant_url, api_key=self.cfg.qdrant_api_key
        )
        async_client = AsyncQdrantClient(
            url=self.cfg.qdrant_url, api_key=self.cfg.qdrant_api_key
        )

        # Attach to existing collection (read-only — no ingestion here)
        vector_store = QdrantVectorStore(
            client=sync_client,
            aclient=async_client,
            collection_name=self.cfg.quran_collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
            dense_vector_name=self.cfg.dense_vector_name,
            sparse_vector_name=self.cfg.sparse_vector_name,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # from_vector_store connects to an existing collection without re-indexing
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

        logger.info("Quran retriever ready — collection: %s", self.cfg.quran_collection_name)

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
        is_parent: Optional[bool] = None,
    ) -> Optional[MetadataFilters]:
        filters = []

        if surah is not None:
            op = FilterOperator.IN if isinstance(surah, list) else FilterOperator.EQ
            filters.append(MetadataFilter(key="surah", value=surah, operator=op))

        if revelation_type is not None:
            op = FilterOperator.IN if isinstance(revelation_type, list) else FilterOperator.EQ
            filters.append(MetadataFilter(key="revelation_type", value=revelation_type, operator=op))

        if juz is not None:
            op = FilterOperator.IN if isinstance(juz, list) else FilterOperator.EQ
            filters.append(MetadataFilter(key="juz", value=juz, operator=op))

        if is_parent is not None:
            filters.append(
                MetadataFilter(key="is_parent", value=is_parent, operator=FilterOperator.EQ)
            )

        return MetadataFilters(filters=filters, condition=FilterCondition.AND) if filters else None

    # ── Search ───────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        surah: Optional[Union[int, List[int]]] = None,
        revelation_type: Optional[Union[str, List[str]]] = None,
        juz: Optional[Union[int, List[int]]] = None,
        is_parent: bool = False,         # default: return child chunks (tafsir slices)
        top_k: int = 10,
    ) -> List[NodeWithScore]:
        """
        Hybrid (dense + sparse) retrieval → Jina reranking.

        Args:
            query:           Arabic search query.
            mode:            "hybrid" | "dense" | "sparse"
            surah:           Filter by surah number(s).
            revelation_type: Filter by "Meccan" | "Medinan".
            juz:             Filter by juz number(s).
            is_parent:       True to retrieve parent summaries; False for tafsir slices.
            top_k:           Final results after reranking.

        Returns:
            Sorted list of NodeWithScore (best first).
        """
        mode_map = {"dense": "default", "sparse": "sparse", "hybrid": "hybrid"}
        qs_mode = mode_map.get(mode.lower(), "hybrid")

        filters = self._build_filters(
            surah=surah, revelation_type=revelation_type, juz=juz, is_parent=is_parent
        )

        retriever = self.index.as_retriever(
            similarity_top_k=self.cfg.retrieval_top_k,
            vector_store_query_mode=qs_mode,
            filters=filters,
        )

        logger.debug("Quran search | query=%r | mode=%s | filters=%s", query, mode, filters)
        candidates: List[NodeWithScore] = await retriever.aretrieve(query)

        if not candidates:
            return []

        reranker = JinaRerank(
            api_key=self.cfg.jina_api_key,
            model=self.cfg.jina_reranker_model,
            top_n=top_k,
        )
        return reranker.postprocess_nodes(
            nodes=candidates,
            query_bundle=QueryBundle(query_str=query),
        )


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
