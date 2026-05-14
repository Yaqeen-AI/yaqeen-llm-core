"""
FiqhRAG Retrieval API — FastAPI

POST /retrieve  →  returns documents related to the query (no generation).
The response is consumed by the query router; generation happens downstream.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio

from fastapi import FastAPI, HTTPException
from llama_index.core import Settings
from pydantic import BaseModel

from core.embeddings import JinaEmbedding
from core.generator import GeminiLLM
from core.graph import fiqh_graph

Settings.embed_model = JinaEmbedding()
Settings.llm = GeminiLLM()

app = FastAPI(
    title="FiqhRAG Retrieval API",
    description="Hybrid Arabic Fiqh retrieval — returns relevant documents for a query router.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


class DocumentOut(BaseModel):
    rank: int
    text: str
    volume_id: str
    book_page: str
    chunk_page: str
    source_url: str
    mazhabs: list[str]
    rerank_score: float
    qdrant_score: float
    short_ref: str


class RetrievalResponse(BaseModel):
    query: str
    count: int
    documents: list[DocumentOut]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: QueryRequest):
    """
    Run hybrid retrieval (BM25 + Jina v3 → Qdrant RRF → Jina reranker)
    and return the top-10 documents related to the query.
    No generation — the caller (query router) decides what to do next.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        state = await asyncio.to_thread(fiqh_graph.invoke, {"query": request.query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    nodes = state.get("documents", [])

    documents = []
    for nws in nodes:
        m = nws.node.metadata
        documents.append(DocumentOut(
            rank=m["rank"],
            text=nws.node.text,
            volume_id=m["volume_id"],
            book_page=m["book_page"],
            chunk_page=m["chunk_page"],
            source_url=m["source_url"],
            mazhabs=m["mazhabs"],
            rerank_score=float(m["rerank_score"]),
            qdrant_score=float(m["qdrant_score"]),
            short_ref=m["short_ref"],
        ))

    return RetrievalResponse(
        query=request.query,
        count=len(documents),
        documents=documents,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
