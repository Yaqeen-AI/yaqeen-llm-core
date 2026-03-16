# ============================================================
# YaqeenAI — FastAPI Retrieval API Router
# ============================================================
# Production-grade API endpoints for the Quran RAG retrieval system.

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from app.models.schemas import (
    AnswerRequest,
    AnswerResponse,
    ContentType,
    RetrievalRequest,
    RetrievalResponse,
    Language,
)
from app.services.hybrid_retrieval import HybridRetrievalPipeline
from app.services.generation_service import GenerationService
from app.services.query_router import QueryRouter, RoutingDecision

router = APIRouter(prefix="/api/v1", tags=["Retrieval"])


# ─── Dependency injection (set by main.py on startup) ───
_pipeline: Optional[HybridRetrievalPipeline] = None
_router: Optional[QueryRouter] = None
_generator: Optional[GenerationService] = None


def set_pipeline(pipeline: HybridRetrievalPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def set_query_router(query_router: QueryRouter) -> None:
    global _router
    _router = query_router


def set_generation_service(generator: Optional[GenerationService]) -> None:
    global _generator
    _generator = generator


def get_pipeline() -> HybridRetrievalPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Retrieval pipeline not initialized")
    return _pipeline


def get_query_router() -> QueryRouter:
    if _router is None:
        raise HTTPException(status_code=503, detail="Query router not initialized")
    return _router


def get_generator() -> GenerationService:
    if _generator is None or not _generator.is_available:
        raise HTTPException(
            status_code=503,
            detail="Generation service not initialized. Configure GOOGLE_API_KEY and the Gemma provider first.",
        )
    return _generator


# ═══════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """
    🔍 Full hybrid retrieval pipeline.
    
    Executes: Semantic Search → BM25 → RRF Fusion → Reranking → MMR
    
    Returns ranked results with full metadata and pipeline diagnostics.
    """
    pipeline = get_pipeline()

    try:
        response = pipeline.retrieve(request)
        return response
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


@router.get("/search")
async def search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=50, description="Number of results"),
    language: Optional[str] = Query(default=None, description="Language filter: ar/en"),
    content_type: Optional[str] = Query(default=None, description="Content type filter: quran_ayah/tafsir"),
    edition_identifier: Optional[str] = Query(default=None, description="Edition identifier filter"),
    surah: Optional[int] = Query(default=None, ge=1, le=114, description="Surah filter"),
    juz: Optional[int] = Query(default=None, ge=1, le=30, description="Juz filter"),
    use_reranking: bool = Query(default=True, description="Apply reranking"),
    use_hybrid: bool = Query(default=True, description="Use hybrid retrieval"),
):
    """
    🔍 Simple search endpoint (GET).
    
    Convenience wrapper around the full retrieval pipeline.
    
    Examples:
        /api/v1/search?q=آية الكرسي&top_k=5
        /api/v1/search?q=بسم الله الرحمن الرحيم&surah=1
        /api/v1/search?q=prayer&language=en
    """
    pipeline = get_pipeline()

    lang = Language(language) if language else None
    ct = ContentType(content_type) if content_type else None

    request = RetrievalRequest(
        query=q,
        top_k=top_k,
        language=lang,
        content_type_filter=ct,
        edition_identifier_filter=edition_identifier,
        surah_filter=surah,
        juz_filter=juz,
        use_reranking=use_reranking,
        use_hybrid=use_hybrid,
    )

    try:
        response = pipeline.retrieve(request)
        return response
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.post("/answer", response_model=AnswerResponse)
async def answer(request: AnswerRequest):
    """
    Generate a grounded answer using the retriever + Gemma 3 generation layer.

    The answer service is intentionally strict:
    - retrieve ayah evidence first
    - add a small local ayah window for context
    - ask the model to cite the Quran evidence explicitly
    """
    generator = get_generator()

    try:
        return generator.answer(request)
    except Exception as e:
        logger.exception("Answer generation failed")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


@router.get("/route")
async def route_query(
    q: str = Query(..., min_length=1, max_length=500, description="Query to route"),
):
    """
    🧭 Query Router — classifies intent and routes to sub-RAGs.
    
    Returns which sub-RAG(s) the query should be sent to.
    Useful for debugging and understanding routing decisions.
    """
    query_router = get_query_router()
    decision = query_router.route(q)

    return {
        "query": q,
        "targets": [t.value for t in decision.targets],
        "detected_language": decision.detected_language,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "detected_keywords": decision.detected_keywords,
    }


@router.get("/health")
async def health():
    """Health check endpoint."""
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "pipeline": "initialized",
        "embedding_model": pipeline._embedding.model_name,
        "bm25_ready": pipeline._bm25.is_built,
        "bm25_corpus_size": pipeline._bm25.corpus_size,
        "generation_ready": _generator.is_available if _generator is not None else False,
    }
