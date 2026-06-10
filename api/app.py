from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from orchestrator.models import AskRequest, AskResponse
from orchestrator.workflow import MultiAgentRagWorkflow, build_default_workflow

logger = logging.getLogger(__name__)

_workflow: MultiAgentRagWorkflow | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _workflow
    logger.info("Starting Yaqeen multi-agent RAG API.")
    _workflow = build_default_workflow()
    
    # Warm up adapters / retrievers
    logger.info("Warming up RAG retrievers...")
    for source, adapter in _workflow.adapters.items():
        try:
            logger.info("Initializing %s retriever...", source)
            retriever = getattr(adapter, "retriever", None)
            if retriever is not None and hasattr(retriever, "setup"):
                retriever.setup()
        except BaseException as exc:
            logger.exception("Failed to warm up %s retriever: %s", source, exc)
            
    yield
    
    logger.info("Stopping Yaqeen multi-agent RAG API. Shutting down retrievers...")
    for source, adapter in _workflow.adapters.items():
        try:
            retriever = getattr(adapter, "retriever", None)
            if retriever is not None and hasattr(retriever, "shutdown"):
                retriever.shutdown()
        except Exception as exc:
            logger.error("Error shutting down %s retriever: %s", source, exc)


app = FastAPI(
    title="Yaqeen AI Multi-Agent RAG API",
    version="1.0.0",
    description="Orchestration layer over Quran, Hadith, and Fiqh RAG systems.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s %s %.1fms", request_id, request.method, request.url.path, elapsed_ms)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
async def health() -> dict[str, Any]:
    status = "ok" if _workflow is not None else "initializing"
    details = {}
    if _workflow is not None:
        for source, adapter in _workflow.adapters.items():
            try:
                retriever = getattr(adapter, "retriever", None)
                is_ready = False
                if source == "fiqh":
                    is_ready = retriever is not None
                else:
                    is_ready = retriever is not None and getattr(retriever, "_index", None) is not None
                details[source] = "ready" if is_ready else "not_ready"
            except Exception as e:
                details[source] = f"error: {str(e)}"
    return {
        "status": status,
        "workflow_initialized": _workflow is not None,
        "components": details,
    }


@app.get("/ask", response_model=AskResponse)
async def ask_get(query: str) -> AskResponse:
    if _workflow is None:
        raise HTTPException(status_code=503, detail="Workflow is not initialized.")
    if not query or len(query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters long.")
    try:
        return await _workflow.ask(query)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to process GET /ask request.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    if _workflow is None:
        raise HTTPException(status_code=503, detail="Workflow is not initialized.")
    try:
        return await _workflow.ask(request.query)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to process /ask request.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

