"""
Fiqh Worker Agent — retrieves Fiqh (jurisprudence) context via the fiqh_rag pipeline.
"""

import sys
import os
import importlib.util
from langchain_core.documents import Document

# ── Path resolution ──
_MMAS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_MMAS_DIR)
_FIQH_RAG_ROOT = os.path.join(_PROJECT_ROOT, "fiqh_rag")

if _FIQH_RAG_ROOT not in sys.path:
    sys.path.insert(0, _FIQH_RAG_ROOT)


def _import_from_fiqh_rag(module_name: str, file_path: str):
    """Import a module from fiqh_rag by explicit file path (avoids core/ collision)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Lazy singleton ──
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        try:
            retriever_path = os.path.join(_FIQH_RAG_ROOT, "core", "retriever.py")
            retriever_mod = _import_from_fiqh_rag("fiqh_core_retriever", retriever_path)
            FiqhRetriever = retriever_mod.FiqhRetriever
            print("   [Fiqh Worker] -> Initializing Fiqh retriever (first call)...")
            _retriever = FiqhRetriever()
            print("   [Fiqh Worker] -> Fiqh retriever ready")
        except Exception as e:
            print(f"   [Fiqh Worker] -> WARNING: Failed to load retriever: {e}")
            class _DummyRetriever:
                def retrieve(self, query):
                    return []
            _retriever = _DummyRetriever()
    return _retriever


def fiqh_agent_node(state: dict) -> dict:
    """Retrieves relevant Fiqh context based on the user's question."""
    query = state["question"]
    print(f"   [Fiqh Worker] -> Retrieving for: '{query[:60]}...'")

    try:
        retriever = _get_retriever()
        results = retriever.retrieve(query)

        docs = []
        for r in results:
            docs.append(Document(
                page_content=r.chunk_text,
                metadata={
                    "source": "Fiqh RAG",
                    "volume_id": r.volume_id,
                    "book_page": r.book_page,
                    "chunk_page": r.chunk_page,
                    "source_url": r.source_url,
                    "mazhabs": r.mazhab_tag(),
                    "qdrant_score": r.qdrant_score,
                    "rerank_score": r.rerank_score,
                    "short_ref": r.short_ref(),
                },
            ))

        # Try Fiqh generator (LM Studio)
        if docs:
            try:
                gen_path = os.path.join(_FIQH_RAG_ROOT, "core", "generator.py")
                gen_mod = _import_from_fiqh_rag("fiqh_core_generator", gen_path)
                answer = gen_mod.generate_answer(query, results)
                docs.insert(0, Document(
                    page_content=answer,
                    metadata={"source": "Fiqh RAG", "type": "generated_answer"},
                ))
            except Exception as gen_err:
                print(f"   [Fiqh Worker] -> Generation skipped: {gen_err}")

        print(f"   [Fiqh Worker] -> Retrieved {len(docs)} documents")

    except Exception as e:
        print(f"   [Fiqh Worker] -> ERROR: {e}")
        docs = [Document(
            page_content=f"Fiqh retrieval error: {str(e)}",
            metadata={"source": "Fiqh RAG", "error": True},
        )]

    return {"retrieved_context": docs}