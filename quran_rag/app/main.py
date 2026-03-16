# ============================================================
# YaqeenAI — Main FastAPI Application
# ============================================================
# Entry point for the Quran RAG retrieval system.
#
# Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Or:       python -m app.main
#
# On startup:
#   1. Loads the persisted Chroma database built by the Colab notebook
#   2. Rebuilds the BM25 index from stored payloads
#   3. Initializes retrieval + optional generation services
#
# API Docs: http://localhost:8000/docs

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# ─── Ensure project root is in path ───
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.services.embedding_service import EmbeddingService
from app.services.bm25_service import BM25RetrievalService
from app.services.reranker_service import RerankerService
from app.services.hybrid_retrieval import HybridRetrievalPipeline
from app.services.generation_service import GenerationService
from app.services.query_router import QueryRouter
from app.services.vector_store_factory import build_vector_store
from app.api.retrieval_router import (
    router as retrieval_router,
    set_generation_service,
    set_pipeline,
    set_query_router,
)
from app.api.ui_router import router as ui_router


# ─── Configure logging ───
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/yaqeen_rag.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    
    On startup:
        1. Load the persisted Chroma collection
        2. Rebuild the BM25 index from stored ayahs
        3. Wire up retrieval and optional generation services
    """
    settings = get_settings()
    logger.info("🚀 YaqeenAI Quran RAG — Starting up...")

    # ─── Step 1: Load persisted collection ───
    vector_store = build_vector_store()
    bm25_service = BM25RetrievalService()

    try:
        info = vector_store.collection_info()
        points_count = info["points_count"]
        logger.info(f"Existing collection: {points_count} points")

        if points_count > 0:
            logger.info(
                "Rebuilding BM25 from the persisted Chroma payloads so query-time "
                "normalization always matches the latest code."
            )
            texts_normalized, chunk_ids, raw_texts, metadatas = (
                vector_store.get_all_texts_and_ids()
            )
            bm25_service.build_index(
                texts=texts_normalized,
                chunk_ids=chunk_ids,
                raw_texts=raw_texts,
                metadatas=metadatas,
            )
            logger.info(f"BM25 index ready: {bm25_service.corpus_size} docs")
        else:
            logger.warning(
                "No persisted Quran + tafsir vectors were found in {}. Build the Chroma zip "
                "from the notebook first, then extract it locally before starting the app.",
                settings.chroma_persist_directory,
            )
    except Exception:
        logger.exception("Failed to load the persisted Chroma collection")
        logger.warning(
            "Starting with an empty retrieval layer. Build the dataset from the notebook first."
        )

    # ─── Step 2: Initialize services ───
    embedding_service = EmbeddingService()

    try:
        reranker_service = RerankerService()
        logger.info(f"Reranker loaded: {reranker_service.model_name}")
    except Exception as e:
        logger.warning(f"Reranker failed to load: {e}. Running without reranking.")
        reranker_service = None

    query_router = QueryRouter()

    # ─── Step 3: Wire up pipeline ───
    pipeline = HybridRetrievalPipeline(
        embedding_service=embedding_service,
        vector_store=vector_store,
        bm25_service=bm25_service,
        reranker_service=reranker_service,
    )

    set_pipeline(pipeline)
    set_query_router(query_router)
    try:
        generation_service = GenerationService(
            retrieval_pipeline=pipeline,
            vector_store=vector_store,
        )
    except Exception as e:
        logger.warning(f"Generation service failed to initialize: {e}")
        generation_service = None
    set_generation_service(generation_service)

    logger.info("✅ YaqeenAI Quran RAG — Ready!")
    logger.info(f"📖 API docs: http://localhost:{settings.port}/docs")

    yield

    # ─── Shutdown ───
    logger.info("🛑 YaqeenAI Quran RAG — Shutting down...")


# ─── Create FastAPI app ───
app = FastAPI(
    title="YaqeenAI — Quran RAG Retrieval System",
    description=(
        "Production-grade Islamic knowledge retrieval system.\n\n"
        "**Pipeline:** Semantic Search → BM25 → RRF Fusion → Reranking → MMR\n\n"
        "**Architecture:** Multi-RAG with Query Routing\n\n"
        "Built for accuracy above all else. ☪️"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS (for frontend integration) ───
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Include routers ───
app.include_router(retrieval_router)
app.include_router(ui_router)


@app.get("/")
async def root():
    return {
        "name": "YaqeenAI — Quran RAG Retrieval System",
        "version": "1.0.0",
        "docs": "/docs",
        "ui": "/ui",
        "search": "/api/v1/search?q=بسم الله الرحمن الرحيم",
    }


# ─── Run directly ───
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
