from langchain_core.documents import Document
from state import AgentState
from models.llm_loader import tokenizer

def compression_node(state: AgentState):
    docs = state.get("retrieved_context", [])
    if not docs:
        return {"retrieved_context": []}

    raw_text = "\n\n".join([f"Source [{doc.metadata.get('source', 'Unknown')}]: {doc.page_content}" for doc in docs])
    tokens = tokenizer.encode(raw_text)
    
    MAX_TOKENS = 750

    if len(tokens) <= MAX_TOKENS:
        compressed_text = raw_text
    else:
        safe_tokens = tokens[:MAX_TOKENS]
        compressed_text = tokenizer.decode(safe_tokens, skip_special_tokens=True)
        compressed_text += "\n\n[System Note: Context truncated for memory constraints.]"

    optimized_doc = Document(page_content=compressed_text, metadata={"compressed": True})
    return {"retrieved_context": [optimized_doc]}