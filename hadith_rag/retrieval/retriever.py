import logging
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union, NamedTuple, Dict
from functools import lru_cache
import hashlib
import json

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("HadithRetriever")


# ── Grade hierarchy ─────────────────────────────────────────────────────────

GRADE_ORDER = ["sahih", "hasan", "mawdu", "daif", "unknown"]
GRADE_RANK: Dict[str, int] = {g: i for i, g in enumerate(GRADE_ORDER)}


def _grade_at_least(grade: str, min_grade: str) -> bool:
    return GRADE_RANK.get(grade.lower(), 99) <= GRADE_RANK.get(min_grade.lower(), 99)


def _grades_from_min(min_grade: str) -> List[str]:
    min_rank = GRADE_RANK.get(min_grade.lower(), 0)
    return [g for g in GRADE_ORDER if GRADE_RANK[g] <= min_rank]


# ── Priority sources ────────────────────────────────────────────────────────

PRIORITY_MOHAADETH = {"البخاري", "مسلم"}
PRIORITY_BOOKS = {"صحيح البخاري", "صحيح مسلم"}
PRIORITY_BOOST = 1.15  # 15% score boost for Bukhari/Muslim


def _is_priority_source(node: NodeWithScore) -> bool:
    """Check if node is from Bukhari or Muslim."""
    meta = node.node.metadata
    mohadeth = meta.get("mohadeth", "")
    book = meta.get("book", "")
    return (
        mohadeth in PRIORITY_MOHAADETH
        or book in PRIORITY_BOOKS
    )


# ── Config ────────────────────────────────────────────────────────────────────

class HadithRagConfig(NamedTuple):
    jina_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    hadith_collection_name: str
    jina_embedding_model: str = "jina-embeddings-v3"
    jina_reranker_model: str = "jina-reranker-v3"
    retrieval_top_k: int = 8
    rerank_top_n: int = 5
    enable_hybrid: bool = False


class HadithRetriever:
    def __init__(self, cfg: HadithRagConfig) -> None:
        self.cfg = cfg
        self._index: Optional[VectorStoreIndex] = None
        self._embed_model: Optional[JinaEmbedding] = None
        self._reranker: Optional[JinaRerank] = None
        self._executor = ThreadPoolExecutor(max_workers=4)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self) -> None:
        logger.info("Initializing Hadith retriever structure...")
        
        self._embed_model = JinaEmbedding(
            api_key=self.cfg.jina_api_key,
            model=self.cfg.jina_embedding_model,
            task="retrieval.query",
            embed_batch_size=1024,
        )
        Settings.embed_model = self._embed_model

        sync_client = QdrantClient(
            url=self.cfg.qdrant_url,
            api_key=self.cfg.qdrant_api_key,
            timeout=30,
        )
        async_client = AsyncQdrantClient(
            url=self.cfg.qdrant_url,
            api_key=self.cfg.qdrant_api_key,
            timeout=30,
        )

        vector_store = QdrantVectorStore(
            client=sync_client,
            aclient=async_client,
            collection_name=self.cfg.hadith_collection_name,
            enable_hybrid=self.cfg.enable_hybrid,  # 🔥 FIX 2: Pass the flag dynamically from config
            text_key="document",
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

        self._reranker = JinaRerank(
            api_key=self.cfg.jina_api_key,
            model=self.cfg.jina_reranker_model,
            top_n=self.cfg.rerank_top_n,
        )

        logger.info(f"Hadith retriever ready — connected to: {self.cfg.hadith_collection_name}")
        
    @property
    def index(self) -> VectorStoreIndex:
        if self._index is None:
            raise RuntimeError("HadithRetriever.setup() has not been called.")
        return self._index

    # ------------------------------------------------------------------
    # Filter building
    # ------------------------------------------------------------------
    def _build_filters(
        self,
        book=None,
        grade=None,
        min_grade=None,
        rawi=None,
        category=None,
        subcategory_name=None,
        mohadeth=None,
        has_explanation=None,
        canonical_group_id=None,
    ) -> Optional[MetadataFilters]:
        filters = []
        string_filters = {
            "book": book,
            "grade": grade,
            "rawi": rawi,
            "category": category,
            "subcategory_name": subcategory_name,
            "mohadeth": mohadeth,
            "canonical_group_id": canonical_group_id,
        }
        for key, value in string_filters.items():
            if value is not None:
                op = FilterOperator.IN if isinstance(value, list) else FilterOperator.EQ
                filters.append(MetadataFilter(key=key, value=value, operator=op))

        if min_grade is not None:
            acceptable = _grades_from_min(min_grade)
            filters.append(MetadataFilter(key="grade", value=acceptable, operator=FilterOperator.IN))

        if has_explanation is not None:
            bool_as_str = "true" if has_explanation else "false"
            filters.append(MetadataFilter(key="hasExplanation", value=bool_as_str, operator=FilterOperator.EQ))

        return MetadataFilters(filters=filters, condition=FilterCondition.AND) if filters else None

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------
    def _cache_key(
        self,
        query: str,
        mode: str,
        filters: Optional[MetadataFilters],
        top_k: int,
        skip_rerank: bool,
    ) -> str:
        filter_dict = filters.dict() if filters else {}
        raw = json.dumps(
            {"q": query, "m": mode, "f": filter_dict, "k": top_k, "sr": skip_rerank},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    def _dedup_by_canonical_group(
        self,
        nodes: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        best_by_group: Dict[str, NodeWithScore] = {}
        
        for node in nodes:
            group_id = node.node.metadata.get("canonical_group_id")
            if group_id is None:
                continue
            if group_id not in best_by_group or node.score > best_by_group[group_id].score:
                best_by_group[group_id] = node

        seen_groups = set()
        result: List[NodeWithScore] = []
        
        for node in nodes:
            group_id = node.node.metadata.get("canonical_group_id")
            if group_id is None:
                result.append(node)
            elif group_id not in seen_groups:
                result.append(best_by_group[group_id])
                seen_groups.add(group_id)

        return result

    # ------------------------------------------------------------------
    # NEW: Priority boost for Bukhari / Muslim
    # ------------------------------------------------------------------
    def _apply_priority_boost(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Boost scores for Bukhari and Muslim hadiths so they rank higher.
        Applied after retrieval and after reranking.
        """
        boosted = []
        for node in nodes:
            if _is_priority_source(node):
                # Apply multiplicative boost
                new_score = node.score * PRIORITY_BOOST
                # Mark in metadata for transparency
                new_meta = dict(node.node.metadata)
                new_meta["priority_boosted"] = True
                new_meta["original_score"] = node.score
                
                new_node = TextNode(
                    id_=node.node.id_,
                    text=node.node.text,
                    metadata=new_meta,
                    excluded_embed_metadata_keys=node.node.excluded_embed_metadata_keys,
                    excluded_llm_metadata_keys=node.node.excluded_llm_metadata_keys,
                )
                boosted.append(NodeWithScore(node=new_node, score=new_score))
            else:
                boosted.append(node)

        # Re-sort by boosted score (descending)
        boosted.sort(key=lambda n: n.score, reverse=True)
        return boosted

    # ------------------------------------------------------------------
    # Search (async)
    # ------------------------------------------------------------------
    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        book=None,
        grade=None,
        min_grade=None,
        rawi=None,
        category=None,
        subcategory_name=None,
        mohadeth=None,
        has_explanation: Optional[bool] = None,
        canonical_group_id=None,
        top_k: int = 5,
        similarity_top_k: int = 20,
        rerank_top_n: int = 10,
        skip_rerank: bool = False,
        dedup_canonical: bool = True,
        prioritize_sahihayn: bool = True,   # ← NEW: toggle Bukhari/Muslim boost
    ) -> List[NodeWithScore]:
        """
        Search with optional priority boost for Sahihayn (Bukhari & Muslim).
        """
        mode_map = {"dense": "default", "sparse": "sparse", "hybrid": "hybrid"}
        qs_mode = mode_map.get(mode.lower(), "hybrid")

        filters = self._build_filters(
            book=book,
            grade=grade,
            min_grade=min_grade,
            rawi=rawi,
            category=category,
            subcategory_name=subcategory_name,
            mohadeth=mohadeth,
            has_explanation=has_explanation,
            canonical_group_id=canonical_group_id,
        )

        # 1. Retrieve candidates
        retriever = self.index.as_retriever(
            similarity_top_k=similarity_top_k,
            vector_store_query_mode=qs_mode,
            filters=filters,
        )
        candidates: List[NodeWithScore] = await retriever.aretrieve(query)
        if not candidates:
            return []

        # 2. Pre-rerank boost (helps Bukhari/Muslim enter reranker window)
        if prioritize_sahihayn:
            candidates = self._apply_priority_boost(candidates)

        # 3. Deduplicate
        if dedup_canonical:
            candidates = self._dedup_by_canonical_group(candidates)

        # 4. Fast path: skip reranking
        if skip_rerank:
            return candidates[:top_k]

        # 5. Rerank
        loop = asyncio.get_running_loop()
        reranked = await loop.run_in_executor(
            self._executor,
            self._sync_rerank,
            candidates,
            query,
            rerank_top_n,
        )

        # 6. Post-rerank boost (final elevation after cross-attention)
        if prioritize_sahihayn:
            reranked = self._apply_priority_boost(reranked)

        # 7. Final dedup
        if dedup_canonical:
            reranked = self._dedup_by_canonical_group(reranked)

        return reranked[:top_k]

    def _sync_rerank(
        self,
        candidates: List[NodeWithScore],
        query: str,
        top_n: int,
    ) -> List[NodeWithScore]:
        self._reranker.top_n = top_n
        return self._reranker.postprocess_nodes(
            nodes=candidates,
            query_bundle=QueryBundle(query_str=query),
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
        logger.info("Hadith retriever shut down.")


# ── Module-level singleton ───────────────────────────────────────────────────

_instance: Optional[HadithRetriever] = None


def get_hadith_retriever(cfg: HadithRagConfig) -> HadithRetriever:
    global _instance
    if _instance is None:
        _instance = HadithRetriever(cfg)
    return _instance