"""
Unified Writer Node.

Synthesizes final answer from retrieved context.
Supports Original, MMAS (retrieved_context) and AIS (ainet_context/clones/initial_context).
"""

import os
import re
from state import AgentState

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def clean_arabic_text(text: str) -> str:
    """
    Cleans Arabic text for the LLM context.
    Removes zero-width characters, tatweel, mid-word spaces, and duplicate letters.
    """
    if not text:
        return text
    # Remove zero-width characters, invisible characters, and tatweel
    text = re.sub(r'[\u200B-\u200D\uFEFFـ]', '', text)
    # Removed mid-word spaces regex because it concatenates all Arabic words
    # Remove duplicate consecutive Arabic characters (e.g., ندمم -> ندم)
    text = re.sub(r'([ا-ي])\1+', r'\1', text)
    # Normalize extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def writer_node(state: AgentState):
    """
    Synthesize a final answer from the retrieved context.
    """
    print("   [Writer] -> Synthesizing final answer...")
    query = state["question"]
    
    # Try AIS specific contexts first, then fallback to standard retrieved_context
    docs = state.get("ainet_context", [])
    if not docs:
        docs = state.get("clones", [])
    if not docs:
        docs = state.get("initial_context", [])
    if not docs:
        docs = state.get("retrieved_context", [])

    cached_ref = state.get("cached_answer", "")

    if not docs and not cached_ref:
        print("   [Writer] -> Empty context. Returning fallback message.")
        return {"final_answer": "لم أتمكن من العثور على معلومات كافية للإجابة على سؤالك. يرجى إعادة صياغة السؤال أو تحديده أكثر."}

    # Build context from all retrieved documents, preserving source info
    if docs:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            cleaned_content = clean_arabic_text(doc.page_content)
            context_parts.append(f"[المصدر {i}: {source}]\n{cleaned_content}")
        context_text = "\n\n".join(context_parts)
    else:
        context_text = "No relevant context found in the database."

    # Lazy imports
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        use_langchain_chain = True
    except Exception:
        use_langchain_chain = False

    # Check if there is a cached reference to rewrite
    if cached_ref:
        print("   [Writer] -> Rephrasing cached answer for consistency and variety...")
        writer_prompt = (
            "You are an Islamic knowledge assistant. You previously generated the following answer for a user's question.\n"
            "Rewrite the answer in natural, fresh Arabic/English. You MUST preserve the exact same religious ruling, facts, "
            "and references. Do not contradict the previous ruling or make up new information.\n\n"
            "Question: {question}\n\n"
            "Previous Answer: {context}\n\n"
            "New Response:"
        )
        context_text = cached_ref
    else:
        # Load prompt template
        prompt_path = os.path.join(_DIR, "Original", "prompts", "writer_prompt.txt")
        # Wait, since prompts might be in Original/prompts, let's just use a relative path from _DIR
        # Better: check if it exists in Original/prompts, Parallel_MMAS/prompts, AIS/prompts
        for arch in ["Original", "Parallel_MMAS", "AIS"]:
            p = os.path.join(_DIR, arch, "prompts", "writer_prompt.txt")
            if os.path.exists(p):
                prompt_path = p
                break
                
        with open(prompt_path, "r", encoding="utf-8") as f:
            writer_prompt = f.read()

    if use_langchain_chain:
        from shared_nodes.models.llm_loader import get_llm
        prompt = PromptTemplate(template=writer_prompt, input_variables=["context", "question"])
        chain = prompt | get_llm(reasoning=True) | StrOutputParser()
        final_answer = chain.invoke({"context": context_text, "question": query})
    else:
        # Fallback manual interpolation when torch DLL issues prevent importing prompts/output_parsers
        print("   [Writer] -> Using direct manual template invocation fallback...")
        formatted_prompt = writer_prompt.replace("{context}", context_text).replace("{question}", query)
        try:
            from shared_nodes.models.llm_loader import get_llm
            llm = get_llm(reasoning=True)
            final_answer = llm.invoke(formatted_prompt)
            if hasattr(final_answer, "content"):
                final_answer = final_answer.content
        except Exception as e:
            print(f"   [Writer] -> Fallback invocation failed ({e}), using raw context.")
            final_answer = context_text

    from nsa import strip_think_tags
    final_answer = strip_think_tags(final_answer)

    print("   [Writer] -> Synthesis complete.")
    return {"final_answer": final_answer}
