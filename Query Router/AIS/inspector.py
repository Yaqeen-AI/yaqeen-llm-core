"""
AIS Inspector (Adaptive Response - Affinity Scoring & Clonal Selection).

Evaluates the affinity (relevance score) of each retrieved chunk (antibody)
against the original user query (antigen), and selects the top-k highest-scoring clones.
"""

import os
import logging
from typing import List
from langchain_core.documents import Document
from state import AgentState

logger = logging.getLogger("ais.inspector")


def inspector_node(state: AgentState) -> dict:
    """
    Inspector Node (Phase 3):
    1. Computes affinity scores of all documents in initial_context against the original question.
    2. Performs Clonal Selection by selecting the top-k highest-scoring chunks.
    """
    original_query = state["question"]
    docs = state.get("initial_context", [])

    if not docs:
        print("   [Inspector] -> No initial antibodies retrieved.")
        return {"clones": []}

    # Load top-k configurable parameter (default to 5)
    try:
        top_k = int(os.getenv("TOP_K_CLONAL_SELECTION", "5"))
    except ValueError:
        top_k = 5

    # Try to load reranker lazily
    reranker = None
    try:
        from shared_nodes.models.reranker import reranker
    except Exception as e:
        logger.warning(f"Failed to import local reranker: {e}")

    scored_docs = []

    if reranker is not None:
        try:
            print(f"   [Inspector] -> Scoring affinity for {len(docs)} chunks using CrossEncoder...")
            sentence_pairs = [[original_query, doc.page_content] for doc in docs]
            scores = reranker.predict(sentence_pairs, batch_size=32)
            
            for doc, score in zip(docs, scores):
                try:
                    doc.metadata["affinity_score"] = float(score)
                except (AttributeError, TypeError):
                    pass
                scored_docs.append((doc, float(score)))
        except Exception as e:
            logger.error(f"Reranker failed, falling back to simple heuristic scoring: {e}")
            reranker = None

    if reranker is None:
        # Fallback heuristic: count keyword overlap as affinity score
        print("   [Inspector] -> Falling back to token-overlap scoring...")
        query_words = set(original_query.lower().split())
        for doc in docs:
            doc_words = set(doc.page_content.lower().split())
            overlap = len(query_words & doc_words)
            score = float(overlap) / max(len(query_words), 1)
            try:
                doc.metadata["affinity_score"] = score
            except (AttributeError, TypeError):
                pass
            scored_docs.append((doc, score))

    # Clonal Selection: Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Isolate top-k clones
    clones = [doc for doc, score in scored_docs[:top_k]]
    
    print(f"   [Inspector] -> Clonal Selection complete. Selected top {len(clones)} highest-affinity clones:")
    for i, doc in enumerate(clones, 1):
        source = doc.metadata.get("source", "Unknown")
        score = doc.metadata.get("affinity_score", 0.0)
        print(f"      [{i}] [Score: {score:.3f}] [{source}] '{doc.page_content[:60]}...'")

    return {
        "clones": clones
    }