# ============================================================
# YaqeenAI — Hadith RAG FastAPI Application
# ============================================================
# Production API server for the Hadith RAG pipeline.
#
# Endpoints:
#   POST /query          — Full RAG pipeline (retrieve + rerank + generate)
#   POST /search         — Retrieval-only (no generation)
#   GET  /health         — Health check
#   GET  /stats          — Index statistics
#
# Features:
#   - Structured JSON responses
#   - Request/response logging with timing
#   - Error handling with Arabic messages
#   - CORS support
#   - Configurable via environment variables
#
# Run:
#   uvicorn api.app:app --host 0.0.0.0 --port 8000

import logging
import time
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from pipeline.config import settings, resolve_grade_label
from pipeline.rag_pipeline import HadithRAGPipeline
from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

# ============================================================
# Request / Response Models
# ============================================================

class QueryRequest(BaseModel):
    """Request body for /query endpoint."""
    query: str = Field(..., min_length=2, max_length=1000, description="User's question in Arabic or English")
    grade_filter: Optional[list[str]] = Field(None, description="Filter by grade: sahih, hasan, daif, mawdu, unknown")
    masdar_filter: Optional[str] = Field(None, description="Filter by source book name (Arabic)")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Number of results before reranking")
    rerank_top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of results after reranking")
    temperature: float = Field(0.3, ge=0.0, le=1.0, description="Generation temperature")


class SearchRequest(BaseModel):
    """Request body for /search endpoint (retrieval only, no generation)."""
    query: str = Field(..., min_length=2, max_length=1000, description="Search query")
    grade_filter: Optional[list[str]] = Field(None, description="Filter by grade")
    masdar_filter: Optional[str] = Field(None, description="Filter by source book")
    top_k: Optional[int] = Field(None, ge=1, le=100, description="Number of results")


class HadithResponse(BaseModel):
    """A single hadith in the response."""
    id: str
    text_ar: str
    grade: str
    grade_ar: str = ""
    ruling: str = ""
    rawi: str = ""
    muhaddith: str = ""
    masdar: str = ""
    safha_raqam: str = ""
    category: str = ""
    subcategory_name: str = ""
    has_explanation: bool = False


class CitationResponse(BaseModel):
    """A citation in the generated response."""
    hadith_index: int
    hadith_id: str
    matn_snippet: str
    grade: str
    grade_ar: str
    masdar: str
    rawi: str
    muhaddith: str
    is_weak: bool


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""
    request_id: str
    query: str
    answer: str
    query_type: str = "general"
    citations: list[CitationResponse] = []
    warnings: list[str] = []
    grounding_verified: bool = False
    hadiths: list[HadithResponse] = []
    timing: dict = {}


class SearchResponse(BaseModel):
    """Response body for /search endpoint."""
    request_id: str
    query: str
    hadiths: list[HadithResponse] = []
    total: int = 0
    timing: dict = {}


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str
    version: str = "1.0.0"
    components: dict = {}


class StatsResponse(BaseModel):
    """Response body for /stats endpoint."""
    chromadb_documents: int = 0
    tfidf_documents: int = 0
    embedding_cache_size: int = 0


# ============================================================
# Application Lifecycle
# ============================================================

# Global pipeline instance
_pipeline: Optional[HadithRAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown the RAG pipeline."""
    global _pipeline
    
    logger.info("🚀 Starting YaqeenAI Hadith RAG API...")
    
    try:
        _pipeline = HadithRAGPipeline()
        _pipeline.hybrid_retriever.initialize()
        logger.info("✅ Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        raise
    
    yield
    
    logger.info("🛑 Shutting down YaqeenAI Hadith RAG API...")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="YaqeenAI — Hadith RAG API",
    description=(
        "Production API for the YaqeenAI Hadith Retrieval-Augmented Generation system. "
        "Searches 155,000+ authenticated hadiths using hybrid retrieval (dense + TF-IDF char n-gram) "
        "with citation-grounded generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Middleware: Request Logging
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration:.3f}s)"
    )
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration:.3f}s"
    
    return response


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    components = {
        "pipeline": "ok" if _pipeline is not None else "not_initialized",
        "chromadb": "ok",
        "tfidf": "ok" if (_pipeline and _pipeline.hybrid_retriever.tfidf.is_built) else "not_built",
    }
    
    status = "healthy" if all(v == "ok" for v in components.values()) else "degraded"
    
    return HealthResponse(
        status=status,
        components=components,
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get index statistics."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    chroma_count = 0
    try:
        chroma_count = _pipeline.hybrid_retriever.dense_retriever.collection.count()
    except Exception:
        pass
    
    tfidf_count = 0
    if _pipeline.hybrid_retriever.tfidf.is_built:
        tfidf_count = len(_pipeline.hybrid_retriever.tfidf._doc_ids)
    
    return StatsResponse(
        chromadb_documents=chroma_count,
        tfidf_documents=tfidf_count,
        embedding_cache_size=_pipeline._embedding_cache.size,
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_hadith(request: QueryRequest):
    """
    Full RAG pipeline: retrieve → rerank → generate with citations.
    
    Combines hybrid retrieval (dense + TF-IDF char n-gram), cross-encoder reranking,
    and Gemini generation with citation grounding verification.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    request_id = str(uuid.uuid4())[:8]
    
    try:
        result = _pipeline.query(
            user_query=request.query,
            grade_filter=request.grade_filter,
            masdar_filter=request.masdar_filter,
            retrieval_top_k=request.top_k,
            rerank_top_k=request.rerank_top_k,
            temperature=request.temperature,
        )
        
        # Build response
        hadiths = [
            HadithResponse(
                id=h.id,
                text_ar=h.text_ar,
                grade=h.grade,
                grade_ar=resolve_grade_label(h.grade, h.grade_ar, h.ruling),
                ruling=h.ruling,
                rawi=h.rawi,
                muhaddith=h.muhaddith,
                masdar=h.masdar,
                safha_raqam=h.safha_raqam,
                category=h.category,
                subcategory_name=h.subcategory_name,
                has_explanation=h.has_explanation,
            )
            for h in result.reranked_hadiths
        ]
        
        citations = []
        warnings = []
        grounding_verified = False
        
        if result.generation:
            citations = [
                CitationResponse(
                    hadith_index=c.hadith_index,
                    hadith_id=c.hadith_id,
                    matn_snippet=c.matn_snippet,
                    grade=c.grade,
                    grade_ar=resolve_grade_label(c.grade, c.grade_ar, ""),
                    masdar=c.masdar,
                    rawi=c.rawi,
                    muhaddith=c.muhaddith,
                    is_weak=c.is_weak,
                )
                for c in result.generation.citations
            ]
            warnings = result.generation.warnings
            grounding_verified = result.generation.grounding_verified
        
        return QueryResponse(
            request_id=request_id,
            query=result.query,
            answer=result.answer,
            query_type=result.query_type,
            citations=citations,
            warnings=warnings,
            grounding_verified=grounding_verified,
            hadiths=hadiths,
            timing=result.timing,
        )
    
    except Exception as e:
        logger.error(f"[{request_id}] Query failed: {e}", exc_info=True)
        # Surface LLM rate-limit errors as 429 instead of 500
        err_msg = str(e)
        if "429" in err_msg or "rate_limit" in err_msg.lower() or "rate limit" in err_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="تم تجاوز حد الاستخدام لخدمة الذكاء الاصطناعي. يرجى المحاولة لاحقاً."
            )
        raise HTTPException(
            status_code=500,
            detail=f"حدث خطأ أثناء معالجة الاستعلام: {err_msg}"
        )


@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
async def search_hadiths(request: SearchRequest):
    """
    Retrieval-only search (no generation).
    
    Returns ranked hadiths from hybrid retrieval without
    passing them to the LLM for answer generation.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    request_id = str(uuid.uuid4())[:8]
    
    try:
        # Use hybrid retriever directly + reranker
        hybrid_result = _pipeline.hybrid_retriever.retrieve(
            query=request.query,
            fused_top_k=request.top_k or 20,
            grade_filter=request.grade_filter,
            masdar_filter=request.masdar_filter,
        )
        
        # Rerank
        reranked = _pipeline.reranker.rerank(
            query=request.query,
            candidates=hybrid_result.hadiths,
            top_k=request.top_k or 10,
        )
        
        hadiths = [
            HadithResponse(
                id=h.id,
                text_ar=h.text_ar,
                grade=h.grade,
                grade_ar=resolve_grade_label(h.grade, h.grade_ar, h.ruling),
                ruling=h.ruling,
                rawi=h.rawi,
                muhaddith=h.muhaddith,
                masdar=h.masdar,
                safha_raqam=h.safha_raqam,
                category=h.category,
                subcategory_name=h.subcategory_name,
                has_explanation=h.has_explanation,
            )
            for h in reranked
        ]
        
        return SearchResponse(
            request_id=request_id,
            query=request.query,
            hadiths=hadiths,
            total=len(hadiths),
            timing=hybrid_result.timing,
        )
    
    except Exception as e:
        logger.error(f"[{request_id}] Search failed: {e}", exc_info=True)
        err_msg = str(e)
        if "429" in err_msg or "rate_limit" in err_msg.lower() or "rate limit" in err_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="تم تجاوز حد الاستخدام لخدمة الذكاء الاصطناعي. يرجى المحاولة لاحقاً."
            )
        raise HTTPException(
            status_code=500,
            detail=f"حدث خطأ أثناء البحث: {err_msg}"
        )


# ============================================================
# Web UI
# ============================================================

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def web_ui():
    """Simple web UI for testing the RAG pipeline."""
    from pathlib import Path
    html_path = Path(__file__).parent / "ui.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ============================================================
# Error Handlers
# ============================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "detail": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "detail": "حدث خطأ داخلي في الخادم",
            "status_code": 500,
        },
    )
