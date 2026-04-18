"""
Hybrid retriever:
  Dense  — Jina Embeddings v3 (semantic)
  Sparse — TF-IDF character n-grams (keyword)
  Fusion — Reciprocal Rank Fusion via Qdrant
  Rerank — Jina Reranker v2 multilingual
"""

import pickle
import sys
from dataclasses import dataclass

import requests
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from core.config import (
    COLLECTION_NAME, EMBED_DIM, JINA_API_KEY,
    JINA_EMBED_MODEL, JINA_RERANK_MODEL,
    TFIDF_PATH, TOP_K_FETCH, TOP_K_FINAL,
)
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
    mazhabs:      list[str] = None  # detected at retrieval time from payload or text

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
    resp = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers=_headers(),
        json={"model": JINA_EMBED_MODEL, "input": [text],
              "dimensions": EMBED_DIM, "task": "retrieval.query"},
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _rerank(query: str, candidates: list[Result], top_n: int) -> list[Result]:
    if not candidates:
        return candidates
    resp = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers=_headers(),
        json={"model": JINA_RERANK_MODEL, "query": query,
              "documents": [c.chunk_text for c in candidates], "top_n": top_n},
    )
    resp.raise_for_status()
    out = []
    for item in resp.json()["results"]:
        r = candidates[item["index"]]
        r.rerank_score = item["relevance_score"]
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# TF-IDF sparse encoder
# ---------------------------------------------------------------------------

def _load_tfidf():
    if not TFIDF_PATH.exists():
        sys.exit(f"TF-IDF model not found at {TFIDF_PATH}. Run: python ingest.py")
    with open(TFIDF_PATH, "rb") as f:
        return pickle.load(f)


def _tfidf_encode(text: str, tfidf) -> SparseVector:
    vec = tfidf.transform([normalize_query(text)]).tocoo()
    return SparseVector(indices=vec.col.tolist(), values=vec.data.tolist())


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class FiqhRetriever:
    def __init__(self) -> None:
        self.client = rag_client()
        self.tfidf  = _load_tfidf()
        if not self.client.collection_exists(COLLECTION_NAME):
            sys.exit(f"Collection '{COLLECTION_NAME}' not found. Run: python ingest.py")

    def retrieve(
        self,
        query: str,
        top_k_fetch: int = TOP_K_FETCH,
        top_k_final: int = TOP_K_FINAL,
    ) -> list[Result]:
        dense  = _embed(query)
        sparse = _tfidf_encode(query, self.tfidf)

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense,  using="dense",  limit=top_k_fetch),
                Prefetch(query=sparse, using="sparse", limit=top_k_fetch),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k_fetch,
            with_payload=True,
        )

        candidates = []
        for p in response.points:
            text = p.payload["chunk_text"]
            # Use pre-computed mazhabs from payload if available, else detect live
            mazhabs = p.payload.get("mazhabs") or detect_mazhabs(text)
            candidates.append(Result(
                chunk_text=text,
                volume_id=p.payload["volume_id"],
                book_page=p.payload["book_page"],
                chunk_page=p.payload["chunk_page"],
                source_url=p.payload.get("source_url", ""),
                qdrant_score=p.score,
                mazhabs=mazhabs,
            ))

        return _rerank(query, candidates, top_n=top_k_final)
