"""
AgentState — Shared state for the MMAS multi-agent LangGraph pipeline.

Extends the base Query Router state with MMAS-specific fields for
colony tracking, pheromone data, and inspector scores.
"""

import operator
from typing import TypedDict, List, Sequence, Annotated, Any
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class AgentState(TypedDict):
    # ── Original query fields ──
    question: str
    current_agent: str
    selected_agents: List[str]
    retrieved_context: List[Document]
    reranker_score: float
    final_answer: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    loop_step: int

    # ── Manager / decomposer fields ──
    sub_queries: List[str]
    sub_query_agents: dict              # {"sub_query_text": ["agent_name", ...]}

    # ── MMAS colony fields ──
    colony_results: dict                # {"sub_query": {"worker_name": [docs], ...}}
    colony_pheromones: dict             # {"sub_query": PheromoneTable snapshot}
    inspector_scores: dict              # {"sub_query": {"worker_name": float}}
    pheromone_log: List[dict]           # [{tau_min, tau_max, trails, ...}]

    # ── Cache control ──
    cache_hit: bool
    cached_answer: str