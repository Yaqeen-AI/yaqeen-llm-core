from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from evaluation import EvaluationOptions, run_full_evaluation, run_retrieval_evaluation, run_single_source_retrieval
from orchestrator.models import AskRequest, AskResponse
from orchestrator.models import RagSource, SourceRetrievalConfig
from orchestrator.workflow import MultiAgentRagWorkflow, build_default_workflow

logger = logging.getLogger(__name__)

_workflow: MultiAgentRagWorkflow | None = None


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=5, ge=1, le=50)
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    filters: dict[str, Any] = Field(default_factory=dict)
    skip_rerank: bool = False
    use_workflow_rewrite: bool = True


class EvaluationRequest(BaseModel):
    questions_path: str = "yaqeen_rag_evaluation_questions.xlsx"
    output_dir: str = "evaluation_results"
    top_k: int = Field(default=5, ge=1, le=50)
    limit: int | None = Field(default=None, ge=1)
    write_csv: bool = True
    use_llm_judge: bool = True
    max_similarity_top_k: int = Field(default=10, ge=1, le=100)
    max_rerank_top_n: int = Field(default=5, ge=1, le=50)
    proactive_jina_rate_limit: bool = False
    jina_tpm_limit: int = Field(default=100_000, ge=1)
    jina_tpm_safety: float = Field(default=0.8, gt=0.0, le=1.0)
    jina_tokens_per_retrieval: int = Field(default=5_000, ge=1)
    jina_retry_wait_seconds: float = Field(default=70.0, ge=0.0)
    jina_max_retries: int = Field(default=2, ge=0, le=10)
    use_workflow_rewrite_for_retrieval_eval: bool = False
    retrieval_timeout_seconds: float = Field(default=45.0, gt=0.0)
    judge_timeout_seconds: float = Field(default=20.0, gt=0.0)


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


@app.post("/retrieve/quran")
async def retrieve_quran(request: RetrievalRequest) -> dict[str, Any]:
    return await _retrieve_source(RagSource.QURAN, request)


@app.post("/retrieve/hadith")
async def retrieve_hadith(request: RetrievalRequest) -> dict[str, Any]:
    return await _retrieve_source(RagSource.HADITH, request)


@app.post("/retrieve/fiqh")
async def retrieve_fiqh(request: RetrievalRequest) -> dict[str, Any]:
    return await _retrieve_source(RagSource.FIQH, request)


@app.post("/eval/retrieval/quran")
async def evaluate_quran_retrieval(request: EvaluationRequest | None = None) -> dict[str, Any]:
    request = request or EvaluationRequest()
    return await _evaluate_retrieval(RagSource.QURAN, request)


@app.post("/eval/retrieval/hadith")
async def evaluate_hadith_retrieval(request: EvaluationRequest | None = None) -> dict[str, Any]:
    request = request or EvaluationRequest()
    return await _evaluate_retrieval(RagSource.HADITH, request)


@app.post("/eval/retrieval/fiqh")
async def evaluate_fiqh_retrieval(request: EvaluationRequest | None = None) -> dict[str, Any]:
    request = request or EvaluationRequest()
    return await _evaluate_retrieval(RagSource.FIQH, request)


@app.post("/eval/full")
async def evaluate_full_multi_agent_multi_rag(request: EvaluationRequest | None = None) -> dict[str, Any]:
    request = request or EvaluationRequest()
    if _workflow is None:
        raise HTTPException(status_code=503, detail="Workflow is not initialized.")
    try:
        result = await run_full_evaluation(_workflow, _evaluation_options(request))
        return {"summary": result.summary, "rows": result.rows, "output_files": result.output_files}
    except Exception as exc:
        logger.exception("Failed to run full evaluation.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _retrieve_source(source: RagSource, request: RetrievalRequest) -> dict[str, Any]:
    if _workflow is None:
        raise HTTPException(status_code=503, detail="Workflow is not initialized.")
    config = SourceRetrievalConfig(
        top_k=request.top_k,
        similarity_top_k=request.similarity_top_k,
        rerank_top_n=request.rerank_top_n,
        mode=request.mode,
        filters=request.filters,
        skip_rerank=request.skip_rerank,
    )
    if request.use_workflow_rewrite:
        config = None
    try:
        result = await run_single_source_retrieval(
            _workflow,
            source,
            request.query,
            config=config,
            top_k=request.top_k,
            use_workflow_rewrite=request.use_workflow_rewrite,
        )
        return {
            "source": result.source.value,
            "query": result.query,
            "retrieval_query": result.retrieval_query,
            "latency_ms": round(result.latency_ms, 2),
            "documents": [document.model_dump(mode="json") for document in result.documents],
        }
    except Exception as exc:
        logger.exception("Failed to retrieve from %s.", source)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _evaluate_retrieval(source: RagSource, request: EvaluationRequest) -> dict[str, Any]:
    if _workflow is None:
        raise HTTPException(status_code=503, detail="Workflow is not initialized.")
    try:
        result = await run_retrieval_evaluation(_workflow, source, _evaluation_options(request))
        return {
            "source": result.source.value,
            "summary": result.summary,
            "rows": result.rows,
            "output_files": result.output_files,
        }
    except Exception as exc:
        logger.exception("Failed to run %s retrieval evaluation.", source)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _evaluation_options(request: EvaluationRequest) -> EvaluationOptions:
    return EvaluationOptions(
        questions_path=Path(request.questions_path),
        output_dir=Path(request.output_dir),
        top_k=request.top_k,
        limit=request.limit,
        write_csv=request.write_csv,
        use_llm_judge=request.use_llm_judge,
        max_similarity_top_k=request.max_similarity_top_k,
        max_rerank_top_n=request.max_rerank_top_n,
        proactive_jina_rate_limit=request.proactive_jina_rate_limit,
        jina_tpm_limit=request.jina_tpm_limit,
        jina_tpm_safety=request.jina_tpm_safety,
        jina_tokens_per_retrieval=request.jina_tokens_per_retrieval,
        jina_retry_wait_seconds=request.jina_retry_wait_seconds,
        jina_max_retries=request.jina_max_retries,
        use_workflow_rewrite_for_retrieval_eval=request.use_workflow_rewrite_for_retrieval_eval,
        retrieval_timeout_seconds=request.retrieval_timeout_seconds,
        judge_timeout_seconds=request.judge_timeout_seconds,
    )
