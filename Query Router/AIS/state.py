import operator
from typing import TypedDict, List, Sequence, Annotated
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class AgentState(TypedDict):
    question: str
    current_agent: str
    selected_agents: List[str]
    retrieved_context: List[Document]
    reranker_score: float
    final_answer: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sub_queries: List[str]
    sub_query_agents: dict  # {"sub_query_text": ["agent_name", ...]}
    loop_step: int