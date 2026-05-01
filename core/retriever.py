"""
Hybrid retriever:
  Dense  — Jina Embeddings v3 (semantic)
  Dense  — BM25 hashed vector (keyword)
  Fusion — Reciprocal Rank Fusion via Qdrant
  Rerank — Jina Reranker v2 multilingual
"""

import pickle
import sys
from dataclasses import dataclass
from typing import Any, cast

import requests
from qdrant_client.models import Fusion, FusionQuery, Prefetch

# Persistent HTTP session — reuses TCP connections to Jina API (saves ~500ms/call)
_http = requests.Session()

from core.config import (
    COLLECTION_NAME, EMBED_DIM, JINA_API_KEY,
    JINA_EMBED_MODEL, JINA_RERANK_MODEL,
    BM25_PATH, BM25_DENSE_DIM, BM25_USE_GPU, TOP_K_FETCH, TOP_K_FINAL,
)
from core.bm25 import BM25Okapi
from core.arabic_utils import detect_mazhabs, format_citation, normalize_query
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
    qdrant_score: float     = 0.0
    rerank_score: float     = 0.0
    mazhabs:      list[str] | None = None  # detected at retrieval time from payload or text

    def __post_init__(self):
        if self.mazhabs is None:
            self.mazhabs = []

    def short_ref(self) -> str:
        return format_citation(self.volume_id, self.book_page, self.chunk_page)

    def mazhab_tag(self) -> str:
        return "، ".join(self.mazhabs) if self.mazhabs else ""


# ---------------------------------------------------------------------------
# Jina helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    if not JINA_API_KEY:
        sys.exit("JINA_API_KEY not set — add it to .env")
    return {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}


def _embed(text: str) -> list[float]:
    resp = _http.post(
        "https://api.jina.ai/v1/embeddings",
        headers=_headers(),
        json={"model": JINA_EMBED_MODEL, "input": [text],
              "dimensions": EMBED_DIM, "task": "retrieval.query"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _rerank(query: str, candidates: list[Result], top_n: int) -> list[Result]:
    if not candidates:
        return candidates
    resp = _http.post(
        "https://api.jina.ai/v1/rerank",
        headers=_headers(),
        json={"model": JINA_RERANK_MODEL, "query": query,
              "documents": [c.chunk_text for c in candidates], "top_n": top_n},
        timeout=20,
    )
    resp.raise_for_status()
    out = []
    for item in resp.json()["results"]:
        r = candidates[item["index"]]
        r.rerank_score = item["relevance_score"]
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# BM25 dense encoder
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
    """Simple whitespace tokenizer for Arabic text."""
    return text.split()


def _bm25_encode(text: str, bm25: BM25Okapi) -> list[float]:
    query_tokens = _simple_tokenize(normalize_query(text))
    return bm25.dense_vector_for_query(query_tokens)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class FiqhRetriever:
    def __init__(self) -> None:
        self.client = rag_client()
        self.bm25 = _load_bm25()
        if not self.client.collection_exists(COLLECTION_NAME):
            sys.exit(f"Collection '{COLLECTION_NAME}' not found. Run: python -m scripts.ingest")

        # Detect whether the collection exposes the named BM25 dense vector
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
        top_k_final: int = TOP_K_FINAL,
    ) -> list[Result]:
        dense = _embed(query)

        bm25_rank = {}
        bm25_scores = []
        response = None

        if self.has_bm25_dense:
            # Use Qdrant fusion on two named dense vectors (preferred)
            bm25_dense = _bm25_encode(query, self.bm25)
            try:
                response = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    prefetch=[
                        Prefetch(query=dense,  using="dense",  limit=top_k_fetch),
                        Prefetch(query=bm25_dense, using="bm25_dense", limit=top_k_fetch),
                    ],
                    query=cast(Any, FusionQuery(fusion=Fusion.RRF)),
                    limit=top_k_fetch,
                    with_payload=True,
                )
            except Exception:
                # Server-side fusion failed — fall back to dense-only query + local fusion
                response = self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=cast(Any, dense),
                    using="dense",
                    limit=top_k_fetch,
                    with_payload=True,
                )
                # Compute BM25 scores for local fusion
                query_tokens = _simple_tokenize(normalize_query(query))
                bm25_scores = self.bm25.get_scores(query_tokens)
                sorted_by_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
                bm25_rank = {doc_idx: rank for rank, doc_idx in enumerate(sorted_by_bm25)}
        else:
            # Collection doesn't have bm25_dense — dense-only query + local BM25 fusion
            response = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=cast(Any, dense),
                using="dense",
                limit=top_k_fetch,
                with_payload=True,
            )
            # Compute BM25 scores across the corpus (cheap for ~16k docs)
            query_tokens = _simple_tokenize(normalize_query(query))
            bm25_scores = self.bm25.get_scores(query_tokens)
            # Build ranking map (doc_idx -> rank) for BM25 scores
            sorted_by_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            bm25_rank = {doc_idx: rank for rank, doc_idx in enumerate(sorted_by_bm25)}

        candidates = []
        for i, p in enumerate(response.points):
            text = p.payload["chunk_text"]
            # Use pre-computed mazhabs from payload if available, else detect live
            mazhabs = p.payload.get("mazhabs") or detect_mazhabs(text)
            # If bm25_dense absent or server fusion failed, combine dense rank with BM25 lexical rank
            qdrant_score = p.score
            if not self.has_bm25_dense or bm25_rank:
                try:
                    doc_id = int(p.id)
                except Exception:
                    # fallback: try payload mapping or assume sequential
                    doc_id = p.payload.get("_doc_idx") if p.payload.get("_doc_idx") is not None else None
                if doc_id is not None and doc_id in bm25_rank:
                    dense_rank = i
                    b_rank = bm25_rank.get(doc_id, len(bm25_scores))
                    # Reciprocal rank fusion (simple): sum of 1/(1+rank)
                    qdrant_score = (1.0 / (1 + dense_rank)) + (1.0 / (1 + b_rank))

            candidates.append(Result(
                chunk_text=text,
                volume_id=p.payload["volume_id"],
                book_page=p.payload["book_page"],
                chunk_page=p.payload["chunk_page"],
                source_url=p.payload.get("source_url", ""),
                qdrant_score=qdrant_score,
                mazhabs=mazhabs,
            ))

        return _rerank(query, candidates, top_n=top_k_final)
