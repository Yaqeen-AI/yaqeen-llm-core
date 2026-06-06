"""
AIS Writer Node.

Synthesizes the final answer using the deduplicated, matured context array (suppressed_context).
Uses the singleton LLM loaded in models/llm_loader.py.
"""

import os
from state import AgentState

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

    # Lazy import of LangChain chain classes (could fail due to torch DLL issues)
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        use_langchain_chain = True
    except Exception:
        use_langchain_chain = False

    # Load prompt template from prompts directory
    prompt_path = os.path.join(_DIR, "prompts", "writer_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        writer_prompt = f.read()

    if use_langchain_chain:
        from models.llm_loader import get_llm
        prompt = PromptTemplate(template=writer_prompt, input_variables=["context", "question"])
        chain = prompt | get_llm() | StrOutputParser()
        final_answer = chain.invoke({"context": context_text, "question": query})
    else:
        # Fallback manual interpolation when torch DLL issues prevent importing prompts/output_parsers
        print("   [Writer] -> Using direct manual template invocation fallback...")
        formatted_prompt = writer_prompt.replace("{context}", context_text).replace("{question}", query)
        try:
            from models.llm_loader import get_llm
            llm = get_llm()
            final_answer = llm.invoke(formatted_prompt)
            if hasattr(final_answer, "content"):
                final_answer = final_answer.content
        except Exception as e:
            print(f"   [Writer] -> Fallback invocation failed ({e}), using raw context.")
            final_answer = context_text

    print("   [Writer] -> Synthesis complete.")
    return {"final_answer": final_answer}
