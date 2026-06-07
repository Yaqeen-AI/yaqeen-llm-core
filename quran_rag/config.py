"""
quran_rag/config.py

All configuration for the Quranic RAG module.
Backed by environment variables (see .env).

Required env vars:
    QDRANT_URL          Qdrant Cloud cluster URL
    QDRANT_API_KEY      Qdrant Cloud API key
    JINA_API_KEY        Jina AI API key (for embedding + reranking)
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class QuranRagConfig(BaseSettings):
    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field(..., alias="QDRANT_URL")
    qdrant_api_key: str = Field(..., alias="QDRANT_API_KEY")
    quran_collection_name: str = Field(
        "quranic_rag_1024d_int8", alias="QURAN_COLLECTION_NAME"
    )
    dense_vector_name: str = "text-dense"
    sparse_vector_name: str = "text-sparse"
    embedding_dimensions: int = 1024

    # ── Jina API ──────────────────────────────────────────────────────────────
    jina_api_key: str = Field(..., alias="JINA_API_KEY")
    jina_embedding_model: str = "jina-embeddings-v3"
    # v2-base-multilingual is faster/cheaper; swap to jina-reranker-v3 for higher recall
    jina_reranker_model: str = "jina-reranker-v3"

    # ── Retrieval knobs ───────────────────────────────────────────────────────
    # How many candidates to pull from Qdrant before reranking
    retrieval_top_k: int = Field(50, alias="QURAN_RETRIEVAL_TOP_K")
    # Final results returned after reranking
    rerank_top_n: int = Field(10, alias="QURAN_RERANK_TOP_N")

    # ── Data source ───────────────────────────────────────────────────────────
    quran_api_url: str = "https://api.quranhub.com"

    model_config = {"env_file": ".env", "extra": "ignore", "populate_by_name": True}


@lru_cache(maxsize=1)
def get_quran_config() -> QuranRagConfig:
    """Return the singleton config (parsed once, cached forever)."""
    return QuranRagConfig()
