"""
Hybrid retriever:
  Dense  — Jina Embeddings v3 (semantic, via JinaEmbedding LlamaIndex wrapper)
  Dense  — BM25 hashed vector (keyword)
  Fusion — Reciprocal Rank Fusion via Qdrant
  Rerank — delegated to JinaReranker (BaseNodePostprocessor) in llamaindex_retriever.py

Latency optimisations:
  - Accepts precomputed_embedding to skip re-embedding on cache miss
  - Parallelises dense embed + BM25 encode/score when embedding from scratch
  - Restricts BM25 ranking to the Qdrant-filtered subset when a filter is active
"""

import logging
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, cast

_logger = logging.getLogger(__name__)

from qdrant_client.models import (
    FieldCondition, Filter, Fusion, FusionQuery, MatchAny, Prefetch,
)

from core.config import (
    COLLECTION_NAME,
    BM25_PATH, BM25_DENSE_DIM, BM25_USE_GPU, TOP_K_FETCH,
)
from core.embeddings import get_embedding_model
from core.bm25 import BM25Okapi
from core.arabic_utils import detect_mazhabs, detect_fiqh_topic, format_citation, normalize_query
from core.qdrant_singleton import rag_client


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class Result:
    chunk_text:   str
    volume_id:    str
    book_page:    str
    chunk_page:   str
    source_url:   str
    qdrant_score: float            = 0.0
    rerank_score: float            = 0.0
    mazhabs:      list[str] | None = None
    fiqh_topic:   str | None       = None

    def __post_init__(self):
        if self.mazhabs is None:
            self.mazhabs = []

    def short_ref(self) -> str:
        return format_citation(self.volume_id, self.book_page, self.chunk_page)

    def mazhab_tag(self) -> str:
        return "، ".join(self.mazhabs) if self.mazhabs else ""


# Module-level embedding instance (shared across all FiqhRetriever instances)
_jina_embed = get_embedding_model()


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def _load_bm25():
    if not BM25_PATH.exists():
        sys.exit(f"BM25 corpus not found at {BM25_PATH}. Run: python -m scripts.ingest")
    with open(BM25_PATH, "rb") as f:
        loaded = pickle.load(f)
        if isinstance(loaded, BM25Okapi):
            loaded.dense_dim = BM25_DENSE_DIM
            loaded.use_gpu = BM25_USE_GPU
            return loaded
        if isinstance(loaded, tuple) and len(loaded) == 2 and isinstance(loaded[0], BM25Okapi):
            loaded[0].dense_dim = BM25_DENSE_DIM
            loaded[0].use_gpu = BM25_USE_GPU
            return loaded[0]
        raise TypeError(f"Unsupported BM25 artifact: {type(loaded)!r}")


def _simple_tokenize(text: str) -> list[str]:
    return text.split()


def _bm25_encode(text: str, bm25: BM25Okapi) -> list[float]:
    return bm25.dense_vector_for_query(_simple_tokenize(normalize_query(text)))


def _build_bm25_rank(
    bm25_scores: list[float],
    points: list,
    has_filter: bool,
) -> dict[int, int]:
    """
    Map doc_idx → BM25 rank.
    When a Qdrant filter was applied, rank only within the returned point IDs
    so that excluded documents cannot inflate the rank of included ones.
    """
    if has_filter and points:
        filtered_ids: set[int] = set()
        for p in points:
            try:
                filtered_ids.add(int(p.id))
            except Exception:
                pass
        if filtered_ids:
            scored = [(i, bm25_scores[i]) for i in filtered_ids if i < len(bm25_scores)]
            ranked = sorted(scored, key=lambda x: x[1], reverse=True)
            return {doc_idx: rank for rank, (doc_idx, _) in enumerate(ranked)}

    ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    return {doc_idx: rank for rank, doc_idx in enumerate(ranked)}


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class FiqhRetriever:
    def __init__(self) -> None:
        self.client = rag_client()
        self.bm25 = _load_bm25()
        if not self.client.collection_exists(COLLECTION_NAME):
            sys.exit(f"Collection '{COLLECTION_NAME}' not found. Run: python -m scripts.ingest")

        for field in ("mazhabs", "volume_id", "fiqh_topic"):
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                pass

        self.has_bm25_dense = False
        try:
            info = self.client.get_collection(COLLECTION_NAME)
            params = info.config.params
            vectors = getattr(params, "vectors", None) or {}
            if isinstance(vectors, dict) and "bm25_dense" in vectors:
                self.has_bm25_dense = True
        except Exception:
            self.has_bm25_dense = False

    def retrieve(
        self,
        query: str,
        top_k_fetch: int = TOP_K_FETCH,
        mazhab_filter: list[str] | None = None,
        topic_filter: list[str] | None = None,
        precomputed_embedding: list[float] | None = None,
    ) -> list[Result]:
        """
        Run hybrid retrieval.

        precomputed_embedding: pass the query vector already computed by the
        cache layer to avoid a redundant Jina API call (~300ms saved per miss).
        topic_filter: list of Fiqh topics to restrict search to (MatchAny).
          Single topic → narrow slice; two topics → union of tied categories.
          Falls back to full-corpus search if the filter returns 0 results
          (graceful degradation before enrich_payloads.py has been run).
        """
        must_conditions = []
        if mazhab_filter:
            must_conditions.append(
                FieldCondition(key="mazhabs", match=MatchAny(any=mazhab_filter))
            )
        if topic_filter:
            if isinstance(topic_filter, str):
                topic_filter = [topic_filter]
            must_conditions.append(
                FieldCondition(key="fiqh_topic", match=MatchAny(any=topic_filter))
            )
        qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        bm25_rank: dict[int, int] = {}
        bm25_scores: list[float] = []

        if self.has_bm25_dense:
            # --- Server-side RRF: need both dense vector and BM25 dense vector ---
            if precomputed_embedding is not None:
                dense = precomputed_embedding
                bm25_dense = _bm25_encode(query, self.bm25)
            else:
                # Parallelise: Jina embed (I/O) + BM25 encode (CPU) concurrently
                with ThreadPoolExecutor(max_workers=2) as ex:
                    f_dense = ex.submit(_jina_embed.get_query_embedding, query)
                    f_bm25d = ex.submit(_bm25_encode, query, self.bm25)
                    dense    = f_dense.result()
                    bm25_dense = f_bm25d.result()

            try:
                response = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    prefetch=[
                        Prefetch(query=dense,      using="dense",      limit=top_k_fetch, filter=qdrant_filter),
                        Prefetch(query=bm25_dense, using="bm25_dense", limit=top_k_fetch, filter=qdrant_filter),
                    ],
                    query=cast(Any, FusionQuery(fusion=Fusion.RRF)),
                    limit=top_k_fetch,
                    with_payload=True,
                )
            except Exception:
                # Server-side fusion failed — dense-only + local BM25 fallback
                response = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=cast(Any, dense),
                    using="dense",
                    query_filter=qdrant_filter,
                    limit=top_k_fetch,
                    with_payload=True,
                )
                bm25_scores = self.bm25.get_scores(_simple_tokenize(normalize_query(query)))
                bm25_rank = _build_bm25_rank(bm25_scores, response.points, qdrant_filter is not None)

        else:
            # --- No bm25_dense vector: dense Qdrant query + local BM25 RRF ---
            query_tokens = _simple_tokenize(normalize_query(query))

            if precomputed_embedding is not None:
                dense = precomputed_embedding
                bm25_scores = self.bm25.get_scores(query_tokens)
            else:
                # Parallelise: Jina embed (I/O) + BM25 scoring (CPU) concurrently
                with ThreadPoolExecutor(max_workers=2) as ex:
                    f_dense = ex.submit(_jina_embed.get_query_embedding, query)
                    f_bm25s = ex.submit(self.bm25.get_scores, query_tokens)
                    dense       = f_dense.result()
                    bm25_scores = f_bm25s.result()

            response = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=cast(Any, dense),
                using="dense",
                query_filter=qdrant_filter,
                limit=top_k_fetch,
                with_payload=True,
            )
            bm25_rank = _build_bm25_rank(bm25_scores, response.points, qdrant_filter is not None)

        candidates: list[Result] = []
        for i, p in enumerate(response.points):
            text   = p.payload["chunk_text"]
            mazhabs = p.payload.get("mazhabs") or detect_mazhabs(text)
            qdrant_score = p.score
            if bm25_rank:
                try:
                    doc_id = int(p.id)
                except Exception:
                    doc_id = p.payload.get("_doc_idx")
                if doc_id is not None and doc_id in bm25_rank:
                    dense_rank = i
                    b_rank     = bm25_rank.get(doc_id, len(bm25_scores))
                    qdrant_score = (1.0 / (1 + dense_rank)) + (1.0 / (1 + b_rank))

            candidates.append(Result(
                chunk_text=text,
                volume_id=p.payload["volume_id"],
                book_page=p.payload["book_page"],
                chunk_page=p.payload["chunk_page"],
                source_url=p.payload.get("source_url", ""),
                qdrant_score=qdrant_score,
                mazhabs=mazhabs,
                fiqh_topic=p.payload.get("fiqh_topic") or None,
            ))

        # Fallback: topic filter returned nothing (fiqh_topic not yet on this collection).
        # Retry without topic filter so the query is never silently abandoned.
        # dense is already computed, so the retry only costs a fast Qdrant round-trip.
        if not candidates and topic_filter:
            _logger.warning(
                "topic_filter %s returned 0 results — retrying without it "
                "(run scripts/enrich_payloads.py to enable topic filtering)",
                topic_filter,
            )
            return self.retrieve(
                query, top_k_fetch, mazhab_filter,
                topic_filter=None,
                precomputed_embedding=dense,
            )

        return candidates
