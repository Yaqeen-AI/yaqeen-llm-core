import operator
from typing import TypedDict, List, Sequence, Annotated
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document

class AgentState(TypedDict):
    question: str
    current_agent: str
    retrieved_context: List[Document]
    reranker_score: float
    final_answer: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sub_queries: List[str]
    loop_step: int