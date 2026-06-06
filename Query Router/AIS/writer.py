"""
AIS Writer Node.

Synthesizes the final answer using the deduplicated, matured context array (suppressed_context).
Uses the singleton LLM loaded in models/llm_loader.py.
"""

import os
from state import AgentState
from models.llm_loader import get_llm

_DIR = os.path.dirname(os.path.abspath(__file__))

def writer_node(state: AgentState) -> dict:
    """
    Writer Node (Phase 4):
    Synthesizes a response using the final suppressed_context array and the antigen (query).
    """
    print("   [Writer] -> Synthesizing final answer...")
    query = state["question"]
    
    # Use suppressed_context if available, otherwise fallback to clones or initial_context
    docs = state.get("suppressed_context", [])
    if not docs:
        docs = state.get("clones", [])
    if not docs:
        docs = state.get("initial_context", [])
        
    # Build context from documents, preserving source info
    if docs:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[{i}] [Source: {source}]\n{doc.page_content}")
        context_text = "\n\n".join(context_parts)
    else:
        context_text = "No relevant context found in the database."

    # Lazy import of LangChain classes
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        use_langchain = True
    except Exception as e:
        print(f"[Writer] -> LangChain import failed ({e}), using fallback context output.")
        use_langchain = False

    # Load prompt template from prompts directory
    prompt_path = os.path.join(_DIR, "prompts", "writer_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        writer_prompt = f.read()

    if use_langchain:
        prompt = PromptTemplate(template=writer_prompt, input_variables=["context", "question"])
        chain = prompt | get_llm() | StrOutputParser()
        final_answer = chain.invoke({"context": context_text, "question": query})
    else:
        final_answer = context_text

    print("   [Writer] -> Synthesis complete.")
    return {"final_answer": final_answer}
