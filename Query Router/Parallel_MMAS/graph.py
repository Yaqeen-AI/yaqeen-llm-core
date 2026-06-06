"""
LangGraph Workflow — MMAS Multi-Agent Query Router.

Flow:
  START → cache_check
        → [HIT]  → END (return cached answer)
        → [MISS] → manager (decompose query)
                  → dispatcher (instantiate colonies, parallel workers)
                  → inspector (score chunks, update pheromones)
                  → compressor (fit into context window)
                  → writer (synthesize final answer)
                  → cache_store (save to semantic cache)
                  → END

Direct-answer shortcut:
  ... → dispatcher → writer → END (skips inspector/compressor)
"""

from langgraph.graph import StateGraph, START, END
from state import AgentState

# Import nodes
from manager import manager_node
from dispatcher import dispatcher_node
from inspector import inspector_node
from compressor import compression_node
from writer import writer_node

# Lazy cache singleton
_cache = None


def _get_cache():
    """Lazy-init the SemanticCache — graceful fallback if unavailable."""
    global _cache
    if _cache is None:
        try:
            from cache import SemanticCache
            _cache = SemanticCache()
        except Exception as e:
            print(f"[Cache] Disabled — {e}")
            _cache = False
    return _cache if _cache else None


# ═══════════════════════════════════════════════════════════════════════════════
# Cache nodes
# ═══════════════════════════════════════════════════════════════════════════════

def cache_check_node(state: AgentState):
    """
    Semantic Cache Intercept — check for cache HIT before any processing.
    """
    question = state["question"]
    cache = _get_cache()

    if cache:
        cached_answer = cache.get(question)
        if cached_answer is not None:
            print(f"   [Cache] -> HIT for: '{question[:40]}...'")
            return {
                "cache_hit": True,
                "cached_answer": cached_answer,
                "final_answer": cached_answer,
            }

    print(f"   [Cache] -> MISS for: '{question[:40]}...'")
    return {
        "cache_hit": False,
        "cached_answer": "",
    }


def cache_store_node(state: AgentState):
    """
    Push the final answer back to the Semantic Cache for future hits.
    """
    cache = _get_cache()
    answer = state.get("final_answer", "")
    question = state["question"]

    if cache and answer:
        try:
            cache.set(question, answer)
            print(f"   [Cache] -> Stored answer for: '{question[:40]}...'")
        except Exception:
            pass

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Routing functions
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_cache(state: AgentState):
    """Route based on cache hit/miss."""
    if state.get("cache_hit"):
        return "end"
    return "manager"


def route_after_dispatch(state: AgentState):
    """Skip inspector/compressor for direct_answer (no docs to rerank)."""
    agents = state.get("selected_agents", [])
    if agents == ["direct_answer"]:
        return "writer"
    return "inspector"


# ═══════════════════════════════════════════════════════════════════════════════
# Build the graph
# ═══════════════════════════════════════════════════════════════════════════════

workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("cache_check", cache_check_node)
workflow.add_node("manager", manager_node)
workflow.add_node("dispatcher", dispatcher_node)
workflow.add_node("inspector", inspector_node)
workflow.add_node("compressor", compression_node)
workflow.add_node("writer", writer_node)
workflow.add_node("cache_store", cache_store_node)

# ── Edges ──
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
workflow.add_edge("inspector", "compressor")
workflow.add_edge("compressor", "writer")
workflow.add_edge("writer", "cache_store")
workflow.add_edge("cache_store", END)

# Compile
app = workflow.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_state = {
        "question": "What does the Quran say about fasting and what are the hadith about charity?",
        "current_agent": "",
        "selected_agents": [],
        "retrieved_context": [],
        "reranker_score": 0.0,
        "sub_queries": [],
        "sub_query_agents": {},
        "final_answer": "",
        "messages": [],
        "loop_step": 0,
        "colony_results": {},
        "colony_pheromones": {},
        "inspector_scores": {},
        "pheromone_log": [],
        "cache_hit": False,
        "cached_answer": "",
    }
    final = app.invoke(test_state)
    print(f"\n{'='*60}")
    print(f"Final Answer:\n{final['final_answer']}")