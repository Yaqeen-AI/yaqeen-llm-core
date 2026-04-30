from langchain_core.documents import Document
from state import AgentState

def direct_answer_node(state: AgentState):

    dummy_doc = Document(
        page_content="This is a general conversational query or greeting. Respond politely and directly to the user as VectorMind.",
        metadata={"source": "Direct Answer"}
    )
    
    return {"retrieved_context": [dummy_doc]}