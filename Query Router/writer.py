import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import AgentState
from models.llm_loader import get_llm

_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. DEFINE THE WRITER NODE
# ==========================================
def writer_node(state: AgentState):
    print("   [Writer] -> Synthesizing final answer...")
    query = state["question"]
    docs = state.get("retrieved_context", [])

    # Extract the compressed text from the single optimized document
    context_text = docs[0].page_content if docs else "No relevant context found in the database."
    with open(os.path.join(_DIR, "prompts", "writer_prompt.txt"), "r") as f:
        writer_prompt = f.read()

    prompt = PromptTemplate(template=writer_prompt, input_variables=["context", "question"])

    # We reuse the 4-bit Llama-3 pipeline from Step 2
    chain = prompt | get_llm() | StrOutputParser()

    # Generate the final response
    final_answer = chain.invoke({"context": context_text, "question": query})
    print("   [Writer] -> Synthesis complete.")

    return {"final_answer": final_answer}
