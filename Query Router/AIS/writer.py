import os
from state import AgentState
from models.llm_loader import get_llm
# NOTE: PromptTemplate and StrOutputParser are imported lazily inside writer_node to avoid heavy dependencies (e.g., torch) at module load time.


_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. DEFINE THE WRITER NODE
# ==========================================
def writer_node(state: AgentState):
    print("   [Writer] -> Synthesizing final answer...")
    query = state["question"]
    docs = state.get("retrieved_context", [])

    # Build context from all retrieved documents, preserving source info
    if docs:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[{i}] [Source: {source}]\n{doc.page_content}")
        context_text = "\n\n".join(context_parts)
    else:
        context_text = "No relevant context found in the database."

    # Lazy import of LangChain classes – may fail if heavy deps (torch) are unavailable
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        use_langchain = True
    except Exception as e:
        print(f"[Writer] -> LangChain import failed ({e}), falling back to simple formatting.")
        use_langchain = False

    # Load writer prompt template from file
    with open(os.path.join(_DIR, "prompts", "writer_prompt.txt"), "r") as f:
        writer_prompt = f.read()

    if use_langchain:
        prompt = PromptTemplate(template=writer_prompt, input_variables=["context", "question"])
        chain = prompt | get_llm() | StrOutputParser()
        final_answer = chain.invoke({"context": context_text, "question": query})
    else:
        # Fallback without LangChain: return context directly as the answer
        # (no LLM synthesis available)
        final_answer = context_text

    print("   [Writer] -> Synthesis complete.")
    return {"final_answer": final_answer}
