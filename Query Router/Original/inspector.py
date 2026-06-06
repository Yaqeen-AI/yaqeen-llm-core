from state import AgentState
# Lazy import reranker inside function to avoid torch import errors on environments without GPU libraries


def inspector_node(state: AgentState):
    """
    Rerank retrieved documents, scoring each document against its own
    sub-query (not the full compound question).

    Documents are tagged with metadata["sub_query"] by the dispatcher.
    Documents without a sub_query tag fall back to the original question.
    """
    # Try to import reranker lazily – if torch or its DLLs are missing we simply skip reranking
    try:
        from models.reranker import reranker
    except Exception as e:
        print(f"   [Inspector] -> Reranker unavailable ({e}), skipping rerank step.")
        # Return docs unchanged
        docs = state.get("retrieved_context", [])
        return {"retrieved_context": docs}

    # If reranker module loaded but the actual model failed to init
    if reranker is None:
        print("   [Inspector] -> Reranker model not loaded, skipping rerank step.")
        return {"retrieved_context": state.get("retrieved_context", [])}

    original_query = state["question"]
    docs = state.get("retrieved_context", [])

    if not docs:
        return {"retrieved_context": []}

    # ── Group documents by their sub-query for accurate reranking ──
    groups: dict[str, list] = {}
    for doc in docs:
        sq = doc.metadata.get("sub_query", original_query)
        groups.setdefault(sq, []).append(doc)

    # ── Rerank each group against its own sub-query ──
    approved_docs = []
    threshold = 0.0

    for sub_query, group_docs in groups.items():
        sentence_pairs = [[sub_query, doc.page_content] for doc in group_docs]
        scores = reranker.predict(sentence_pairs, batch_size=32)
        scored = list(zip(group_docs, scores))

        # Sort by score descending within each group
        scored.sort(key=lambda x: x[1], reverse=True)

        for doc, score in scored:
            if score > threshold:
                doc.metadata["rerank_score"] = float(score)
                approved_docs.append(doc)

    print(f"   [Inspector] -> {len(docs)} docs in, {len(approved_docs)} approved "
          f"across {len(groups)} sub-quer{'ies' if len(groups) > 1 else 'y'}")

    return {"retrieved_context": approved_docs}