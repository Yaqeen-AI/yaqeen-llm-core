"""
quran_rag — Quranic RAG module

Public exports for use in the unified API and router.

Usage (in api/app.py lifespan):
    from quran_rag.retriever import get_quran_retriever
    get_quran_retriever().setup()
"""
from .config import QuranRagConfig, get_quran_config
from .retriever import QuranRetriever, get_quran_retriever

__all__ = [
    "QuranRagConfig",
    "get_quran_config",
    "QuranRetriever",
    "get_quran_retriever",
]
