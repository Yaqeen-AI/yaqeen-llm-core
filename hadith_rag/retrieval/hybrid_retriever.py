# ============================================================
# YaqeenAI — Hybrid Retriever with Reciprocal Rank Fusion
# ============================================================
# Combines dense (ChromaDB / Jina v3) and sparse (TF-IDF char n-gram)
# retrieval using Reciprocal Rank Fusion (RRF).
#
# Architecture:
#   1. Dense : ChromaDB cosine similarity (Jina v3 embeddings)
#   2. Sparse: TF-IDF with character n-grams (3-5) — morphology-aware
#   3. Fusion : RRF with k=60
#   4. Dedup  : Canonical group deduplication
#
# Why TF-IDF char n-grams over BM25 for Arabic hadiths:
#   - Arabic morphology: "الصلاة" / "صلاتك" / "يصلي" share root chars
#   - Narrator name variants handled via subword overlap
#   - No OOV problem for unseen word forms
#   - scipy sparse matrix: 155K × 300K fits in < 1 GB RAM
#   - Query latency: ~50-100 ms (vectorised NumPy dot product)
#
# RRF formula: score(d) = Σ 1 / (k + rank_i(d))
# where k=60 (standard), rank_i(d) is the rank of document d
# in retriever i.

import logging
import re
import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Optional, Literal
from dataclasses import dataclass, field

from pipeline.config import resolve_grade_bucket, settings
from pipeline.answer_policy import grade_priority
from pipeline.embed_query import JinaQueryEmbedder
from pipeline.retrieve import HadithRetriever, RetrievedHadith, RetrievalResult
from retrieval.bm25_service import BM25Service, get_bm25_service
from retrieval.tfidf_service import TFIDFService, get_tfidf_service
from retrieval.query_preprocessor import preprocess_query, ProcessedQuery

logger = logging.getLogger(__name__)


_SOURCE_NORMALIZER = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]+")
_SOURCE_WHITESPACE = re.compile(r"\s+")


def _normalize_source_name(source: str) -> str:
    text = _SOURCE_NORMALIZER.sub("", str(source or "").strip().lower())
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    return _SOURCE_WHITESPACE.sub(" ", text).strip()


def _source_priority(masdar: str) -> int:
    normalized = _normalize_source_name(masdar)

    if "صحيح البخاري" in normalized or normalized == "البخاري" or "bukhari" in normalized:
        return 0

    if "صحيح مسلم" in normalized or normalized == "مسلم" or "muslim" in normalized:
        return 1

    return 2


class EmbeddingCache:
    """Thread-safe in-memory TTL + LRU cache for query embeddings."""

    def __init__(self, max_size: int, ttl_seconds: int):
        self._max_size = max(1, int(max_size))
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._items: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[list[float]]:
        now = time.monotonic()
        with self._lock:
            cached = self._items.get(key)
            if cached is None:
                return None

            expires_at, value = cached
            if expires_at <= now:
                del self._items[key]
                return None

            self._items.move_to_end(key)
            return value

    def put(self, key: str, value: list[float]) -> None:
        expires_at = time.monotonic() + self._ttl_seconds
        with self._lock:
            if key in self._items:
                del self._items[key]
            self._items[key] = (expires_at, value)
            self._items.move_to_end(key)

            while len(self._items) > self._max_size:
                self._items.popitem(last=False)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._items)


# ============================================================
# Configuration
# ============================================================

# RRF fusion constant (standard value from the RRF paper)
RRF_K = 60

# Default retrieval depths
DENSE_TOP_K = 30   # Fetch more from dense to give RRF good candidates
SPARSE_TOP_K = 30  # Same for TF-IDF sparse

# Final fused result count (before reranking)
FUSED_TOP_K = 20


@dataclass
class HybridResult:
    """Result from hybrid retrieval (dense + TF-IDF sparse + RRF fusion)."""
    query: ProcessedQuery
    hadiths: list[RetrievedHadith] = field(default_factory=list)
    dense_count: int = 0
    sparse_count: int = 0
    fused_count: int = 0
    dedup_removed: int = 0
    timing: dict = field(default_factory=dict)


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) to merge multiple ranked lists.
    
    Args:
        ranked_lists: List of ranked lists, each containing (doc_id, score) tuples
                      sorted by score descending.
        k: RRF constant (default 60). Higher k = more weight to lower-ranked docs.
        
    Returns:
        Fused list of (doc_id, rrf_score) tuples sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    
    for ranked_list in ranked_lists:
        for rank, (doc_id, _original_score) in enumerate(ranked_list, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by RRF score descending
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


# ============================================================
# Canonical Group Deduplication
# ============================================================

def deduplicate_by_canonical_group(
    hadiths: list[RetrievedHadith],
) -> tuple[list[RetrievedHadith], int]:
    """
    Deduplicate hadiths by canonical_group_id.
    
    For near-duplicate hadiths (same normalized matn), keep only the
    highest-ranked one from each canonical group.
    
    Args:
        hadiths: List of hadiths sorted by relevance (highest first)
        
    Returns:
        Tuple of (deduplicated hadiths, number removed)
    """
    grouped: dict[str, list[tuple[int, RetrievedHadith]]] = {}

    for index, hadith in enumerate(hadiths):
        group_id = hadith.canonical_group_id if hadith.canonical_group_id else hadith.id
        grouped.setdefault(group_id, []).append((index, hadith))

    deduped_with_order: list[tuple[int, RetrievedHadith]] = []
    removed = 0

    for members in grouped.values():
        selected_index, selected_hadith = min(
            members,
            key=lambda item: (
                _source_priority(item[1].masdar),
                grade_priority(item[1].grade),
                item[1].distance,
                item[0],
            ),
        )
        removed += len(members) - 1
        deduped_with_order.append((selected_index, selected_hadith))

    deduped = [hadith for _, hadith in sorted(deduped_with_order, key=lambda item: item[0])]
    return deduped, removed


# ============================================================
# Hybrid Retriever
# ============================================================

class HybridRetriever:
    """
    Hybrid retriever combining dense (Jina/ChromaDB) and sparse (TF-IDF).

    Pipeline:
    1. Preprocess query (normalize, classify, expand)
    2. Dense retrieval via ChromaDB (Jina v3 embeddings)
    3. Sparse retrieval via TF-IDF char n-gram index
    4. RRF fusion
    5. Canonical group deduplication
    6. Return top-K fused results for reranking
    """

    def __init__(
        self,
        embedder: Optional[JinaQueryEmbedder] = None,
        dense_retriever: Optional[HadithRetriever] = None,
        tfidf_service: Optional[TFIDFService] = None,
        bm25_service: Optional[BM25Service] = None,
        tfidf_cache_path: Optional[Path] = None,
        bm25_cache_path: Optional[Path] = None,
        embedding_cache_size: Optional[int] = None,
        embedding_cache_ttl_seconds: Optional[int] = None,
    ):
        self.embedder = embedder
        self.dense_retriever = dense_retriever
        self.tfidf = tfidf_service or get_tfidf_service()
        self.bm25 = bm25_service or get_bm25_service()
        self.tfidf_cache_path = tfidf_cache_path or (
            settings.DATA_DIR / "tfidf_index.pkl"
        )
        self.bm25_cache_path = bm25_cache_path or (
            settings.DATA_DIR / "bm25_index.pkl"
        )
        self.embedding_cache = EmbeddingCache(
            max_size=embedding_cache_size or settings.EMBEDDING_CACHE_SIZE,
            ttl_seconds=embedding_cache_ttl_seconds or settings.EMBEDDING_CACHE_TTL_SECONDS,
        )

        self._initialized = False
    
    def initialize(self) -> None:
        """
        Lazy initialization of heavy components.
        Call this explicitly or it will be called on first retrieve().
        """
        if self._initialized:
            return
        
        logger.info("Initializing HybridRetriever...")
        
        # Dense retriever
        if self.embedder is None:
            self.embedder = JinaQueryEmbedder()
        if self.dense_retriever is None:
            self.dense_retriever = HadithRetriever()
        
        # TF-IDF: try loading from cache
        if not self.tfidf.is_built:
            loaded = self.tfidf.load(self.tfidf_cache_path)
            if not loaded:
                logger.warning(
                    f"TF-IDF index not found at {self.tfidf_cache_path}. "
                    "Hybrid retrieval will use dense-only mode until TF-IDF is built. "
                    "Run: python -m retrieval.build_tfidf_index"
                )

        if not self.bm25.is_built:
            loaded = self.bm25.load(self.bm25_cache_path)
            if not loaded:
                logger.warning(
                    f"BM25 index not found at {self.bm25_cache_path}. "
                    "BM25 retrieval mode will fall back to dense-only until BM25 is built. "
                    "Run: python -m retrieval.build_bm25_index"
                )
        
        self._initialized = True
        logger.info("HybridRetriever initialized")
    
    def retrieve(
        self,
        query: str,
        dense_top_k: int = DENSE_TOP_K,
        sparse_top_k: int = SPARSE_TOP_K,
        fused_top_k: int = FUSED_TOP_K,
        grade_filter: Optional[str | list[str]] = None,
        masdar_filter: Optional[str | list[str]] = None,
        enable_dedup: bool = True,
        retrieval_mode: Literal["tfidf", "bm25", "both"] = "both",
    ) -> HybridResult:
        """
        Execute hybrid retrieval: dense + sparse → RRF fusion → dedup.
        
        Args:
            query: User's search query (raw text)
            dense_top_k: Number of dense (ChromaDB) results
            sparse_top_k: Number of sparse (TF-IDF/BM25) results
            fused_top_k: Number of results after RRF fusion
            grade_filter: Optional grade filter(s)
            masdar_filter: Optional source book filter (str or list of canonical names)
            enable_dedup: Whether to deduplicate by canonical group
            retrieval_mode: Sparse mode to use: tfidf, bm25, or both
            
        Returns:
            HybridResult with fused, deduplicated hadiths
        """
        if not self._initialized:
            self.initialize()

        if retrieval_mode not in {"tfidf", "bm25", "both"}:
            raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode}")
        
        timing = {}
        total_start = time.time()
        
        # ── Step 1: Preprocess query ──
        t0 = time.time()
        processed = preprocess_query(query)
        timing["preprocess"] = time.time() - t0
        
        # ── Safety: skip retrieval for greeting/out-of-scope ──
        if processed.skip_retrieval:
            timing["total"] = time.time() - total_start
            logger.info(f"Query marked skip_retrieval ({processed.query_type.value}), returning empty")
            return HybridResult(
                query=processed,
                hadiths=[],
                timing=timing,
            )

        # ── Use auto-detected book filter if no explicit masdar_filter was provided ──
        effective_masdar_filter = masdar_filter or processed.extracted_masdar or None
        if effective_masdar_filter and effective_masdar_filter != masdar_filter:
            logger.info(f"Auto-detected masdar_filter from query: {repr(effective_masdar_filter)}")

        # ── Step 2: Dense retrieval (Jina → ChromaDB) ──
        t0 = time.time()
        model_name = getattr(self.embedder, "model", "")
        dimensions = getattr(self.embedder, "dimensions", "")
        cache_key = f"{model_name}|{dimensions}|{processed.dense_query.strip()}"

        query_vector = self.embedding_cache.get(cache_key)
        if query_vector is None:
            query_vector = self.embedder.embed_query(processed.dense_query)
            self.embedding_cache.put(cache_key, query_vector)
        
        dense_result = self.dense_retriever.retrieve(
            query_embedding=query_vector,
            top_k=dense_top_k,
            grade_filter=grade_filter,
            masdar_filter=effective_masdar_filter,
        )
        # Build dense ranked list: (doc_id, similarity_score)
        dense_ranked = [
            (h.id, h.similarity_score) for h in dense_result.hadiths
        ]
        timing["dense"] = time.time() - t0
        
        sparse_queries_to_run: list[str] = []
        seen_sq: set[str] = set()
        for sq in [processed.sparse_query] + processed.multi_queries:
            sq_stripped = sq.strip()
            if sq_stripped and sq_stripped not in seen_sq:
                sparse_queries_to_run.append(sq_stripped)
                seen_sq.add(sq_stripped)

        # ── Step 3a: Sparse retrieval (TF-IDF char n-gram) — Multi-Query ──
        tfidf_ranked: list[tuple[str, float]] = []
        tfidf_ranked_lists: list[list[tuple[str, float]]] = []
        if retrieval_mode in {"tfidf", "both"} and self.tfidf.is_built:
            t0 = time.time()
            for sq in sparse_queries_to_run:
                result_list = self.tfidf.search(sq, top_k=sparse_top_k)
                if result_list:
                    tfidf_ranked_lists.append(result_list)

            if tfidf_ranked_lists:
                tfidf_ranked = reciprocal_rank_fusion(tfidf_ranked_lists)[:sparse_top_k]
            timing["tfidf"] = time.time() - t0
        elif retrieval_mode in {"tfidf", "both"}:
            logger.info("TF-IDF not available for selected mode")
            timing["tfidf"] = 0.0

        # ── Step 3b: Sparse retrieval (BM25) — Multi-Query ──
        bm25_ranked: list[tuple[str, float]] = []
        bm25_ranked_lists: list[list[tuple[str, float]]] = []
        if retrieval_mode in {"bm25", "both"} and self.bm25.is_built:
            t0 = time.time()
            for sq in sparse_queries_to_run:
                result_list = self.bm25.search(sq, top_k=sparse_top_k)
                if result_list:
                    bm25_ranked_lists.append(result_list)

            if bm25_ranked_lists:
                bm25_ranked = reciprocal_rank_fusion(bm25_ranked_lists)[:sparse_top_k]
            timing["bm25"] = time.time() - t0
        elif retrieval_mode in {"bm25", "both"}:
            logger.info("BM25 not available for selected mode")
            timing["bm25"] = 0.0

        timing["sparse"] = timing.get("tfidf", 0.0) + timing.get("bm25", 0.0)
        logger.info(
            f"Sparse mode={retrieval_mode}: queries={len(sparse_queries_to_run)} "
            f"tfidf={len(tfidf_ranked)} bm25={len(bm25_ranked)}"
        )

        # ── Step 4: RRF Fusion ──
        t0 = time.time()

        ranked_lists = [dense_ranked]
        if tfidf_ranked:
            ranked_lists.append(tfidf_ranked)
        if bm25_ranked:
            ranked_lists.append(bm25_ranked)
        
        fused = reciprocal_rank_fusion(ranked_lists)
        
        # Take top-K fused IDs
        fused_top = fused[:fused_top_k]
        
        # Build hadiths from the dense results (they have full metadata)
        # Map doc_id → hadith from dense results
        dense_map = {h.id: h for h in dense_result.hadiths}
        
        # Identify TF-IDF-only IDs that need metadata from ChromaDB
        sparse_only_ids = [
            doc_id for doc_id, _ in fused_top if doc_id not in dense_map
        ]
        
        # Fetch full metadata for sparse-only results from ChromaDB
        sparse_map: dict[str, RetrievedHadith] = {}
        if sparse_only_ids:
            logger.info(
                f"Fetching {len(sparse_only_ids)} TF-IDF-only results from ChromaDB"
            )
            try:
                fetched = self.dense_retriever.collection.get(
                    ids=sparse_only_ids,
                    include=["documents", "metadatas"],
                )
                if fetched["ids"]:
                    for i, doc_id in enumerate(fetched["ids"]):
                        metadata = fetched["metadatas"][i] if fetched["metadatas"] else {}
                        raw_grade = metadata.get("grade", "")
                        raw_grade_ar = metadata.get("grade_ar", "")
                        raw_ruling = metadata.get("ruling", "")
                        sparse_map[doc_id] = RetrievedHadith(
                            id=doc_id,
                            text_ar=fetched["documents"][i] if fetched["documents"] else "",
                            distance=0.5,  # Placeholder — RRF score used for ordering
                            grade=resolve_grade_bucket(raw_grade, raw_grade_ar, raw_ruling),
                            grade_ar=raw_grade_ar,
                            ruling=raw_ruling,
                            rawi=metadata.get("rawi", ""),
                            muhaddith=metadata.get("mohadeth", ""),       # stored as 'mohadeth' in ChromaDB
                            masdar=metadata.get("book", ""),              # stored as 'book' in ChromaDB
                            safha_raqam=str(metadata.get("numberOrPage", "")),  # stored as 'numberOrPage' in ChromaDB
                            category=metadata.get("category", ""),
                            subcategory_name=metadata.get("subcategory_name", ""),
                            hadith_tag=metadata.get("hadith_tag", ""),
                            has_explanation=str(metadata.get("hasExplanation", "False")).lower() == "true",  # stored as 'hasExplanation'
                            canonical_group_id=metadata.get("canonical_group_id", ""),
                        )
            except Exception as e:
                logger.warning(f"Failed to fetch sparse-only results: {e}")
        
        # Merge: prefer dense (has embedding distance), fall back to sparse lookup
        fused_hadiths = []
        for doc_id, rrf_score in fused_top:
            if doc_id in dense_map:
                hadith = dense_map[doc_id]
                hadith.distance = 1.0 - rrf_score  # Convert to distance
                fused_hadiths.append(hadith)
            elif doc_id in sparse_map:
                hadith = sparse_map[doc_id]
                hadith.distance = 1.0 - rrf_score
                fused_hadiths.append(hadith)
            else:
                logger.debug(f"Skipping {doc_id}: not found in ChromaDB")
        
        timing["fusion"] = time.time() - t0
        
        # ── Step 4b: Post-fusion grade/masdar filter ──
        # Dense retrieval applies the filter at the ChromaDB level, but
        # sparse (TF-IDF) results fetched via collection.get() bypass the
        # filter.  Re-apply here so no unfiltered hadiths leak through.
        if grade_filter or effective_masdar_filter:
            t0 = time.time()
            pre_filter_count = len(fused_hadiths)
            allowed_grades: set[str] | None = None
            if grade_filter:
                if isinstance(grade_filter, str):
                    allowed_grades = {grade_filter}
                elif isinstance(grade_filter, list):
                    allowed_grades = set(grade_filter)

            filtered = []
            for h in fused_hadiths:
                if allowed_grades and h.grade not in allowed_grades:
                    continue
                if effective_masdar_filter:
                    if isinstance(effective_masdar_filter, list):
                        if h.masdar not in effective_masdar_filter:
                            continue
                    elif h.masdar != effective_masdar_filter:
                        continue
                filtered.append(h)
            fused_hadiths = filtered
            timing["post_filter"] = time.time() - t0
            logger.info(
                f"Post-fusion filter: {pre_filter_count} → {len(fused_hadiths)} "
                f"(grade={grade_filter}, masdar={effective_masdar_filter})"
            )

        # ── Step 5: Deduplication ──
        dedup_removed = 0
        if enable_dedup:
            t0 = time.time()
            fused_hadiths, dedup_removed = deduplicate_by_canonical_group(
                fused_hadiths
            )
            timing["dedup"] = time.time() - t0
        
        timing["total"] = time.time() - total_start
        
        logger.info(
            f"mode={retrieval_mode}, "
            f"Hybrid retrieval: dense={len(dense_ranked)}, "
            f"tfidf={len(tfidf_ranked)}, bm25={len(bm25_ranked)}, fused={len(fused_hadiths)}, "
            f"dedup_removed={dedup_removed}, "
            f"expansion_tokens={len(processed.expansion_tokens)}, "
            f"multi_queries={len(processed.multi_queries)}, "
            f"total={timing['total']:.3f}s"
        )
        
        return HybridResult(
            query=processed,
            hadiths=fused_hadiths,
            dense_count=len(dense_ranked),
            sparse_count=len(tfidf_ranked) + len(bm25_ranked),
            fused_count=len(fused_hadiths),
            dedup_removed=dedup_removed,
            timing=timing,
        )


# ============================================================
# Module-level singleton
# ============================================================

_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get or create the singleton hybrid retriever."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "ما صحة حديث من غشنا فليس منا"
    
    retriever = HybridRetriever()
    result = retriever.retrieve(query)
    
    print(f"\n{'='*60}")
    print(f"Query: {result.query.original}")
    print(f"Type: {result.query.query_type.value}")
    print(f"Dense: {result.dense_count} | Sparse: {result.sparse_count} | Fused: {result.fused_count}")
    print(f"Dedup removed: {result.dedup_removed}")
    print(f"Timing: {result.timing}")
    print(f"\nTop 5 results:")
    for i, h in enumerate(result.hadiths[:5], 1):
        print(f"  [{i}] {h.id}: {h.text_ar[:80]}...")
