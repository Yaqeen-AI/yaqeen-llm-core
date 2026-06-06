"""
State definition for the AIS (Artificial Immune System) Query Router.
"""

from typing import TypedDict, List, Sequence
from langchain_core.documents import Document

class AgentState(TypedDict):
    # Antigen (User Query)
    question: str
    
    # Innate Response
    sub_queries: List[str]
    initial_context: List[Document]
    
    # Adaptive Response
    clones: List[Document]
    matured_sub_queries: List[str]
    secondary_context: List[Document]
    
    # Suppression & Synthesis
    suppressed_context: List[Document]
    final_answer: str
    
    # Cache / Memory Cell
    cache_hit: bool
    cached_answer: str
    
    # Flow management
    loop_step: int