import sys
import os
from langchain_core.documents import Document
from state import AgentState

# ---------------------------------------------------------------------------
# Path resolution: add hadith_rag root so its internal imports work
# e.g.  from pipeline.config import ...
# ---------------------------------------------------------------------------
_QUERY_ROUTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_QUERY_ROUTER_DIR)
_HADITH_RAG_ROOT = os.path.join(_PROJECT_ROOT, "hadith_rag")

if _HADITH_RAG_ROOT not in sys.path:
    sys.path.insert(0, _HADITH_RAG_ROOT)

# ---------------------------------------------------------------------------
# Lazy singleton for the heavyweight pipeline
# ---------------------------------------------------------------------------
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline.rag_pipeline import HadithRAGPipeline
        _pipeline = HadithRAGPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------

def hadith_agent_node(state: AgentState):
    """Retrieves relevant Hadith context based on the user's question."""
    query = state["question"]
    print(f"   [Hadith Agent] -> Retrieving hadiths for: '{query[:60]}...'")

    try:
        pipeline = _get_pipeline()
        response = pipeline.query(query)

        # ── Convert RAGResponse → list[Document] ──
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

        # If pipeline returned an answer but no individual hadiths
        # (e.g. greeting / out-of-scope), wrap the answer text.
        if not docs and response.answer:
            docs = [Document(
                page_content=response.answer,
                metadata={"source": "Hadith RAG"},
            )]

        print(f"   [Hadith Agent] -> Retrieved {len(docs)} documents")

    except Exception as e:
        print(f"   [Hadith Agent] -> ERROR: {e}")
        docs = [Document(
            page_content=f"Hadith retrieval error: {str(e)}",
            metadata={"source": "Hadith RAG", "error": True},
        )]

    return {"retrieved_context": docs}