"""
LangGraph Workflow — AIS (Artificial Immune System) Query Router.

Flow:
  START → cache_check
        → [HIT]  → END (return cached answer)
        → [MISS] → manager (decompose query)
                  → dispatcher (innate response, parallel workers)
                  → [If greeting/direct_answer] → writer → cache_store → END
                  → [Normal RAG workflow]
                    → inspector (affinity scoring & clonal selection)
                    → maturation (query expansion & secondary retrieval)
                    → suppression (regulation/deduplication)
                    → writer (response synthesis)
                    → cache_store (memory cell formation)
                    → END
"""

from langgraph.graph import StateGraph, START, END
from state import AgentState

# Import nodes
from manager import manager_node
from dispatcher import dispatcher_node
from inspector import inspector_node
from maturation import maturation_node
from suppression import suppression_node
from writer import writer_node

# Lazy cache singleton
_cache = None

def _get_cache():
    """Lazy-initialize the MemoryCellsCache."""
    global _cache
    if _cache is None:
        try:
            from cache import MemoryCellsCache
            _cache = MemoryCellsCache()
        except Exception as e:
            print(f"[Cache] Disabled — {e}")
            _cache = False
    return _cache if _cache else None


# ═══════════════════════════════════════════════════════════════════════════════
# Cache / Memory Cell Nodes
# ═══════════════════════════════════════════════════════════════════════════════

def cache_check_node(state: AgentState):
    """
    Checks the Memory Cells (Semantic Cache) for a previous hit.
    """
    question = state["question"]
    cache = _get_cache()

    if cache:
        cached_answer = cache.get(question)
        if cached_answer is not None:
            return {
                "cache_hit": True,
                "cached_answer": cached_answer,
                "final_answer": cached_answer,
            }

    return {
        "cache_hit": False,
        "cached_answer": "",
    }


def cache_store_node(state: AgentState):
    """
    Forms a new Memory Cell (cache update) using the generated answer.
    """
    # Only store if it wasn't a cache hit
    if state.get("cache_hit"):
        return {}

    cache = _get_cache()
    answer = state.get("final_answer", "")
    question = state["question"]

    if cache and answer:
        try:
            cache.set(question, answer)
            print(f"   [Memory Cells] -> Created new memory cell for: '{question[:40]}...'")
        except Exception:
            pass

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional Routing Functions
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_cache(state: AgentState):
    """Route based on memory cell hit or miss."""
    if state.get("cache_hit"):
        return "end"
    return "manager"


def route_after_dispatch(state: AgentState):
    """
    If the sub-queries were greetings (only direct_answer worker was used),
    skip adaptive/maturation phases and head straight to response synthesis.
    """
    # Look at whether direct_answer was the only source in initial_context
    docs = state.get("initial_context", [])
    if docs and all(doc.metadata.get("source") == "Direct Answer" for doc in docs):
        return "writer"
    return "inspector"


# ═══════════════════════════════════════════════════════════════════════════════
# Compile Graph
# ═══════════════════════════════════════════════════════════════════════════════

workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("cache_check", cache_check_node)
workflow.add_node("manager", manager_node)
workflow.add_node("dispatcher", dispatcher_node)
workflow.add_node("inspector", inspector_node)
workflow.add_node("maturation", maturation_node)
workflow.add_node("suppression", suppression_node)
workflow.add_node("writer", writer_node)
workflow.add_node("cache_store", cache_store_node)

# Add edges
workflow.add_edge(START, "cache_check")
workflow.add_conditional_edges("cache_check", route_after_cache, {
    "end": END,
    "manager": "manager",
})
workflow.add_edge("manager", "dispatcher")
workflow.add_conditional_edges("dispatcher", route_after_dispatch, {
    "writer": "writer",
    "inspector": "inspector",
})
workflow.add_edge("inspector", "maturation")
workflow.add_edge("maturation", "suppression")
workflow.add_edge("suppression", "writer")
workflow.add_edge("writer", "cache_store")
workflow.add_edge("cache_store", END)

# Compile the workflow
app = workflow.compile()