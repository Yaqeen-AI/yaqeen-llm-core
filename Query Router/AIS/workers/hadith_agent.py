"""
Hadith Worker Agent — retrieves Hadith context via the hadith_rag pipeline.
"""

import sys
import os
from langchain_core.documents import Document

# ── Path resolution ──
_AIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_AIS_DIR)
_HADITH_RAG_ROOT = os.path.join(_PROJECT_ROOT, "hadith_rag")

if _HADITH_RAG_ROOT not in sys.path:
    sys.path.insert(0, _HADITH_RAG_ROOT)

# ── Lazy singleton ──
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from pipeline.rag_pipeline import HadithRAGPipeline
            _pipeline = HadithRAGPipeline()
        except Exception as e:
            print(f"   [Hadith Worker] -> WARNING: Failed to import pipeline: {e}")
            class _DummyPipeline:
                def query(self, query):
                    class _Result:
                        answer = None
                        reranked_hadiths = []
                    return _Result()
            _pipeline = _DummyPipeline()
    return _pipeline


def hadith_agent_node(state: dict) -> dict:
    """Retrieves relevant Hadith context based on the user's question."""
    query = state["question"]
    print(f"   [Hadith Worker] -> Retrieving for: '{query[:60]}...'")

    try:
        pipeline = _get_pipeline()
        response = pipeline.query(query)

        docs = []
        for h in response.reranked_hadiths:
            docs.append(Document(
                page_content=h.text_ar,
                metadata={
                    "source": "Hadith RAG",
                    "masdar": h.masdar,
                    "rawi": h.rawi,
                    "grade": h.grade,
                    "muhaddith": h.muhaddith,
                    "safha_raqam": h.safha_raqam,
                    "category": getattr(h, "category", ""),
                },
            ))

        if not docs and response.answer:
            docs = [Document(
                page_content=response.answer,
                metadata={"source": "Hadith RAG"},
            )]

        print(f"   [Hadith Worker] -> Retrieved {len(docs)} documents")

    except Exception as e:
        print(f"   [Hadith Worker] -> ERROR: {e}")
        docs = [Document(
            page_content=f"Hadith retrieval error: {str(e)}",
            metadata={"source": "Hadith RAG", "error": True},
        )]

    return {"retrieved_context": docs}