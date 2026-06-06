"""
Writer — Final Answer Synthesis.

Receives compressed context from the Compressor and uses the LLM to
generate a coherent final answer.  The answer is then pushed to the
Semantic Cache for future hits.
"""

import os
import logging
from state import AgentState

logger = logging.getLogger("mmas.writer")

_DIR = os.path.dirname(os.path.abspath(__file__))


def writer_node(state: AgentState):
    """
    Synthesize a final answer from the compressed context.

    Uses the LLM (via LangChain prompt chain) if available,
    otherwise falls back to returning the raw context.
    """
    print("   [Writer] -> Synthesizing final answer...")
    query = state["question"]
    docs = state.get("retrieved_context", [])

    # Build context from documents, preserving source info
    if docs:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[{i}] [Source: {source}]\n{doc.page_content}")
        context_text = "\n\n".join(context_parts)
    else:
        context_text = "No relevant context found in the database."

    # Lazy imports
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        use_langchain = True
    except Exception as e:
        logger.warning(f"LangChain import failed ({e}), using fallback")
        use_langchain = False

    # Load prompt template
    prompt_path = os.path.join(_DIR, "prompts", "writer_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        writer_prompt = f.read()

    if use_langchain:
        from models.llm_loader import get_llm
        prompt = PromptTemplate(template=writer_prompt, input_variables=["context", "question"])
        chain = prompt | get_llm() | StrOutputParser()
        final_answer = chain.invoke({"context": context_text, "question": query})
    else:
        final_answer = context_text

    print("   [Writer] -> Synthesis complete.")
    return {"final_answer": final_answer}
