import sys
import os
from langchain_core.documents import Document
from state import AgentState

# ---------------------------------------------------------------------------
# Path resolution: add quran_rag root so its internal imports work
# e.g.  from app.services.hybrid_retrieval import ...
# ---------------------------------------------------------------------------
_QUERY_ROUTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_QUERY_ROUTER_DIR)
_QURAN_RAG_ROOT = os.path.join(_PROJECT_ROOT, "quran_rag")

if _QURAN_RAG_ROOT not in sys.path:
    sys.path.insert(0, _QURAN_RAG_ROOT)

# ---------------------------------------------------------------------------
# Lazy singleton for the heavyweight services
# ---------------------------------------------------------------------------
_generation_service = None
_vector_store = None


def _get_services():
    """Initialize Quran RAG services once (lazy)."""
    global _generation_service, _vector_store
    if _generation_service is not None:
        return _generation_service

    from app.services.vector_store_factory import build_vector_store
    from app.services.embedding_service import EmbeddingService
    from app.services.bm25_service import BM25RetrievalService
    from app.services.reranker_service import RerankerService
    from app.services.hybrid_retrieval import HybridRetrievalPipeline
    from app.services.generation_service import GenerationService

    print("   [Quran Agent] -> Initializing Quran RAG services (first call)...")

    _vector_store = build_vector_store()
    bm25 = BM25RetrievalService()

    # Rebuild BM25 from stored payloads
    try:
        texts, ids, raws, metas = _vector_store.get_all_texts_and_ids()
        bm25.build_index(texts=texts, chunk_ids=ids, raw_texts=raws, metadatas=metas)
    except Exception as e:
        print(f"   [Quran Agent] -> BM25 rebuild skipped: {e}")

    embedding = EmbeddingService()

    try:
        reranker = RerankerService()
    except Exception:
        reranker = None

    pipeline = HybridRetrievalPipeline(
        embedding_service=embedding,
        vector_store=_vector_store,
        bm25_service=bm25,
        reranker_service=reranker,
    )

    _generation_service = GenerationService(
        retrieval_pipeline=pipeline,
        vector_store=_vector_store,
    )

    print("   [Quran Agent] -> Quran RAG services ready")
    return _generation_service


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def quran_agent_node(state: AgentState):
    """Retrieves relevant Quranic context based on the user's question."""
    query = state["question"]
    print(f"   [Quran Agent] -> Retrieving for: '{query[:60]}...'")

    try:
        from app.models.schemas import AnswerRequest

        service = _get_services()

        if service.is_available:
            # Full pipeline: retrieval + generation
            request = AnswerRequest(query=query)
            response = service.answer(request)

            # ── Map AnswerCitation → list[Document] ──
            docs = []
            for c in response.citations:
                docs.append(Document(
                    page_content=c.text,
                    metadata={
                        "source": "Quran RAG",
                        "surah_number": c.surah_number,
                        "ayah_number": c.ayah_number_in_surah,
                        "ayah_ref": c.ayah_ref,
                        "surah_name": c.surah_name_english,
                        "content_type": str(c.content_type.value) if c.content_type else None,
                        "edition": c.edition_identifier,
                        "score": c.score,
                    },
                ))

            # If generation produced an answer but no citations, wrap it
            if not docs and response.answer:
                docs = [Document(
                    page_content=response.answer,
                    metadata={"source": "Quran RAG"},
                )]
        else:
            # Generation service unavailable — retrieval-only mode
            from app.models.schemas import RetrievalRequest
            pipeline = service._pipeline
            req = RetrievalRequest(query=query)
            retrieval = pipeline.retrieve(req)

            docs = []
            for r in retrieval.results:
                docs.append(Document(
                    page_content=r.text,
                    metadata={
                        "source": "Quran RAG",
                        "surah_number": r.metadata.surah_number,
                        "ayah_number": r.metadata.ayah_number_in_surah,
                        "ayah_ref": r.metadata.ayah_ref,
                        "score": r.score,
                    },
                ))

        print(f"   [Quran Agent] -> Retrieved {len(docs)} documents")

    except Exception as e:
        print(f"   [Quran Agent] -> ERROR: {e}")
        docs = [Document(
            page_content=f"Quran retrieval error: {str(e)}",
            metadata={"source": "Quran RAG", "error": True},
        )]

    return {"retrieved_context": docs}