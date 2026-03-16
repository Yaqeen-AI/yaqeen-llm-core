# ============================================================
# YaqeenAI — Application Configuration
# ============================================================

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """
    Central configuration for the Quran RAG system.
    All values can be overridden via environment variables or .env file.
    """

    # --- Quran API ---
    quran_api_base_url: str = Field(
        default="https://api.quranhub.com/v1",
        description="Base URL for the Quran Hub API"
    )
    quran_default_edition: str = Field(
        default="quran-uthmani",
        description="Default Quran edition identifier used during ingestion",
    )

    # --- Chroma ---
    chroma_persist_directory: str = Field(
        default="data/quran_full",
        description="Persist directory for the notebook-built Chroma database",
    )
    chroma_collection_name: str = Field(
        default="quran_tafsir_ar",
        description="Collection name used inside Chroma for the Arabic Quran + tafsir corpus",
    )

    # --- Embedding ---
    embedding_model_name: str = Field(
        default="jinaai/jina-embeddings-v3",
        description=(
            "Default production model for the full-Quran pipeline. "
            "Jina v3 supports multilingual retrieval with task-specific embeddings."
        )
    )
    embedding_dimension: int = Field(default=1024)
    embedding_batch_size: int = Field(default=16)
    embedding_max_length: int = Field(
        default=1024,
        description="Maximum token length for embedding models that expose this control",
    )
    embedding_task_passage: str = Field(
        default="retrieval.passage",
        description="Task name passed to Jina embeddings for document encoding",
    )
    embedding_task_query: str = Field(
        default="retrieval.query",
        description="Task name passed to Jina embeddings for query encoding",
    )
    embedding_prefix_passage: str = Field(
        default="passage: ",
        description="Optional prefix for document texts when using asymmetric encoders"
    )
    embedding_prefix_query: str = Field(
        default="query: ",
        description="Optional prefix for query texts when using asymmetric encoders"
    )

    # --- Reranker ---
    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description=(
            "Multilingual reranker for final precision over dense + lexical candidates."
        )
    )

    # --- Retrieval ---
    semantic_top_k: int = Field(default=30, description="Top-K from dense/semantic search")
    bm25_top_k: int = Field(default=30, description="Top-K from BM25 sparse search")
    rrf_top_k: int = Field(default=15, description="Top-K after RRF fusion (fed to reranker)")
    rerank_top_k: int = Field(default=5, description="Final top-K after reranking")
    rrf_k: int = Field(default=60, description="RRF constant (standard = 60)")
    bm25_k1: float = Field(default=1.5, description="BM25 k1 param (1.5 for short docs)")
    bm25_b: float = Field(default=0.5, description="BM25 b param (0.5 for short docs)")
    bm25_score_threshold: float = Field(
        default=0.0,
        description="Minimum BM25 score to include in results (0.0 = include all positives)"
    )
    retrieval_context_window: int = Field(
        default=1,
        description="How many neighboring ayahs to add on each side during answer generation",
    )

    # --- Collections ---
    quran_collection_name: str = Field(default="quran_tafsir_ar")

    # --- Generation ---
    generation_provider: str = Field(
        default="google_genai",
        description="Provider for answer generation. Currently supports google_genai.",
    )
    generation_model_name: str = Field(
        default="gemma-3-27b-it",
        description="Default answer generation model. Override if your provider exposes a different Gemma 3 id.",
    )
    generation_temperature: float = Field(default=0.2)
    generation_top_p: float = Field(default=0.9)
    generation_max_output_tokens: int = Field(default=1024)
    google_api_key: Optional[str] = Field(
        default=None,
        description="API key for Google GenAI / Gemini-hosted Gemma generation.",
    )

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=True)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton for settings."""
    return Settings()
