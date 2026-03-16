# ============================================================
# YaqeenAI — Hadith RAG Configuration
# ============================================================
# Central configuration for the Hadith RAG pipeline.
# All values can be overridden via environment variables or .env file.

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from hadith_rag root
_HADITH_RAG_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_HADITH_RAG_ROOT / ".env")


class Settings:
    """Configuration for the Hadith RAG system."""

    # --- Paths ---
    HADITH_RAG_ROOT: Path = _HADITH_RAG_ROOT
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR",
        str(_HADITH_RAG_ROOT / "chroma_db" / "hadith_chroma_db"),
    )
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "hadiths"
    )
    DATA_DIR: Path = _HADITH_RAG_ROOT / "data"
    DATASET_STATS_PATH: Path = _HADITH_RAG_ROOT / "data" / "dataset_stats.json"

    # --- Jina API (query-time embedding) ---
    JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")
    JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"
    JINA_EMBEDDING_MODEL: str = os.getenv(
        "JINA_EMBEDDING_MODEL", "jina-embeddings-v3"
    )
    JINA_EMBEDDING_DIM: int = int(os.getenv("JINA_EMBEDDING_DIM", "1024"))

    # --- Reranker ---
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
    )

    # --- Retrieval ---
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))

    # --- TF-IDF Sparse Retrieval ---
    TFIDF_INDEX_PATH: str = os.getenv(
        "TFIDF_INDEX_PATH",
        str(_HADITH_RAG_ROOT / "data" / "tfidf_index.pkl"),
    )
    TFIDF_MAX_FEATURES: int = int(os.getenv("TFIDF_MAX_FEATURES", "300000"))

    # --- Hybrid Retrieval ---
    DENSE_TOP_K: int = int(os.getenv("DENSE_TOP_K", "30"))
    SPARSE_TOP_K: int = int(os.getenv("SPARSE_TOP_K", "30"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))

    # --- Caching ---
    EMBEDDING_CACHE_SIZE: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))

    # --- Gemini --- FREE
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemma-3-27b-it")

    # --- Claude (Anthropic) --- PAID
    # CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")
    # CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022")

    # --- Groq --- FREE (commented, kept for quick rollback)
    # GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    # GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # --- ChromaDB HNSW Settings (used during collection creation on Colab) ---
    HNSW_SPACE: str = "cosine"
    HNSW_CONSTRUCTION_EF: int = 200
    HNSW_M: int = 32
    HNSW_SEARCH_EF: int = 150

    # --- Grade Mappings ---
    GRADE_MAP = {
        "sahih": "صحيح",
        "hasan": "حسن",
        "daif": "ضعيف",
        "mawdu": "موضوع",
        "unknown": "غير محدد",
    }


def resolve_grade_label(
    grade: str = "",
    grade_ar: str = "",
    ruling: str = "",
) -> str:
    """
    Resolve the best Arabic label to display for a hadith ruling.

    Prefer the canonical mapped grade when it is known. If the indexed grade is
    unknown but the dataset still carries an Arabic label or detailed ruling
    such as "مرسل", surface that instead of the generic "غير محدد".
    """
    grade = str(grade or "").strip()
    grade_ar = str(grade_ar or "").strip()
    ruling = str(ruling or "").strip()

    unknown_tokens = {"", "unknown", "غير محدد", "غير معروف", "ليس محدد"}

    if grade and grade != "unknown":
        return Settings.GRADE_MAP.get(grade, grade_ar or ruling or grade)

    if grade_ar not in unknown_tokens:
        return grade_ar

    if ruling not in unknown_tokens:
        return ruling

    if grade in Settings.GRADE_MAP:
        return Settings.GRADE_MAP[grade]

    return Settings.GRADE_MAP["unknown"]


settings = Settings()
