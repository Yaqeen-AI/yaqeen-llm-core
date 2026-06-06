"""
Quran Worker Agent — retrieves Quranic context via the quran_rag pipeline.
"""

import sys
import os
from langchain_core.documents import Document

# ── Path resolution ──
_MMAS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_MMAS_DIR)
_QURAN_RAG_ROOT = os.path.join(_PROJECT_ROOT, "quran_rag")

if _QURAN_RAG_ROOT not in sys.path:
    sys.path.insert(0, _QURAN_RAG_ROOT)

# ── Lazy singleton ──
_generation_service = None


def _get_services():
    """Initialize Quran RAG services once (lazy)."""
    global _generation_service
    if _generation_service is not None:
        return _generation_service

    from app.services.vector_store_factory import build_vector_store
    from app.services.embedding_service import EmbeddingService
    from app.services.bm25_service import BM25RetrievalService
    from app.services.reranker_service import RerankerService
    from app.services.hybrid_retrieval import HybridRetrievalPipeline
    from app.services.generation_service import GenerationService

    print("   [Quran Worker] -> Initializing Quran RAG services (first call)...")
    vs = build_vector_store()

    bm25 = BM25RetrievalService()
    try:
        texts, ids, raws, metas = vs.get_all_texts_and_ids()
        bm25.build_index(texts=texts, chunk_ids=ids, raw_texts=raws, metadatas=metas)
    except Exception as e:
        print(f"   [Quran Worker] -> BM25 rebuild skipped: {e}")

    embedding = EmbeddingService()
    try:
        reranker = RerankerService()
    except Exception:
        reranker = None

    pipeline = HybridRetrievalPipeline(
        embedding_service=embedding,
        vector_store=vs,
        bm25_service=bm25,
        reranker_service=reranker,
    )
    _generation_service = GenerationService(
        retrieval_pipeline=pipeline,
        vector_store=vs,
    )
    print("   [Quran Worker] -> Quran RAG services ready")
    return _generation_service


def quran_agent_node(state: dict) -> dict:
    """Retrieves relevant Quranic context based on the user's question."""
    query = state["question"]
    print(f"   [Quran Worker] -> Retrieving for: '{query[:60]}...'")

    try:
        try:
            from app.models.schemas import AnswerRequest
        except Exception:
            AnswerRequest = None

        service = _get_services()

        if AnswerRequest is not None and getattr(service, "is_available", False):
            request = AnswerRequest(query=query)
            response = service.answer(request)
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
            if not docs and getattr(response, "answer", None):
                docs = [Document(
                    page_content=response.answer,
                    metadata={"source": "Quran RAG"},
                )]
        else:
            try:
                from app.models.schemas import RetrievalRequest
            except Exception:
                RetrievalRequest = None

            if RetrievalRequest is not None:
                pipeline = getattr(service, "_pipeline", None)
                if pipeline:
                    req = RetrievalRequest(query=query)
                    retrieval = pipeline.retrieve(req)
                    docs = [Document(
                        page_content=r.text,
                        metadata={
                            "source": "Quran RAG",
                            "surah_number": r.metadata.surah_number,
                            "ayah_number": r.metadata.ayah_number_in_surah,
                            "ayah_ref": r.metadata.ayah_ref,
                            "score": r.score,
                        },
                    ) for r in retrieval.results]
                else:
                    docs = [Document(page_content="Quran retrieval unavailable.",
                                     metadata={"source": "Quran RAG", "error": True})]
            else:
                docs = [Document(page_content="Quran agent disabled.",
                                 metadata={"source": "Quran RAG", "error": True})]

        print(f"   [Quran Worker] -> Retrieved {len(docs)} documents")

    except Exception as e:
        print(f"   [Quran Worker] -> ERROR: {e}")
        docs = [Document(
            page_content=f"Quran retrieval error: {str(e)}",
            metadata={"source": "Quran RAG", "error": True},
        )]

    return {"retrieved_context": docs}