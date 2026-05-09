from state import AgentState
from models.reranker import reranker

def inspector_node(state: AgentState):
    query = state["question"]
    docs = state.get("retrieved_context", [])

    if not docs:
        return {"retrieved_context": []}

    sentence_pairs = [[query, doc.page_content] for doc in docs]
    # Process all pairs in parallel batches on the GPU
    scores = reranker.predict(sentence_pairs, batch_size=32)
    scored_docs = list(zip(docs, scores))

    threshold = 0.0
    approved_docs = [doc for doc, score in scored_docs if score > threshold]

    return {"retrieved_context": approved_docs}