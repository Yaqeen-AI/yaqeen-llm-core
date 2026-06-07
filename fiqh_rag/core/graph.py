"""
FiqhRAG LangGraph pipeline — retrieval-only, no generation node.

Graph topology:  START → extract_filter → retrieve → END

The extract_filter node detects mazhab mentions in the query and stores them
in state so the retrieve node can apply a pre-search Qdrant filter.

Usage:
    from core.graph import fiqh_graph
    result = fiqh_graph.invoke({"query": "ما حكم الوضوء بالماء المستعمل؟"})
    docs = result["documents"]   # list[NodeWithScore]
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from langgraph.graph import END, START, StateGraph

from core.arabic_utils import detect_mazhabs, detect_fiqh_topic
from core.llamaindex_retriever import FiqhLlamaRetriever


class FiqhRAGState(TypedDict):
    query: str
    mazhab_filter: list[str] | None
    topic_filter: list[str] | None     # Fiqh topic(s) — narrows Qdrant search space; list supports union search on ties
    precomputed_embedding: list[float] | None  # reuse cache embed vector to skip re-embed
    documents: Annotated[list[NodeWithScore], operator.add]


_retriever_instance: FiqhLlamaRetriever | None = None


def _get_retriever() -> FiqhLlamaRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = FiqhLlamaRetriever()
    return _retriever_instance


def extract_filter_node(state: FiqhRAGState) -> dict:
    """Detect mazhab mentions and Fiqh topic in the query — both become Qdrant pre-filters."""
    mazhabs = detect_mazhabs(state["query"])
    topic = detect_fiqh_topic(state["query"])
    return {
        "mazhab_filter": mazhabs if mazhabs else None,
        "topic_filter": topic,
    }


def retrieve_node(state: FiqhRAGState) -> dict:
    """Run hybrid retrieval with optional mazhab + topic pre-filters, populate state['documents']."""
    nodes = _get_retriever().retrieve_with_filter(
        state["query"],
        mazhab_filter=state.get("mazhab_filter"),
        topic_filter=state.get("topic_filter"),
        precomputed_embedding=state.get("precomputed_embedding"),
    )
    return {"documents": nodes}


def build_fiqh_graph():
    """Build and compile the FiqhRAG retrieval graph."""
    builder = StateGraph(FiqhRAGState)
    builder.add_node("extract_filter", extract_filter_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_edge(START, "extract_filter")
    builder.add_edge("extract_filter", "retrieve")
    builder.add_edge("retrieve", END)
    return builder.compile()


fiqh_graph = build_fiqh_graph()


def build_query_engine(
    retriever: FiqhLlamaRetriever | None = None,
) -> RetrieverQueryEngine:
    """
    LlamaIndex-native RetrieverQueryEngine (Settings.llm must be configured first).
    FiqhLlamaRetriever handles reranking internally, so no extra postprocessors needed.
    """
    if retriever is None:
        retriever = _get_retriever()
    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[],
        response_synthesizer=get_response_synthesizer(),
    )
