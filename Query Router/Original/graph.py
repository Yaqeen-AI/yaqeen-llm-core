from langgraph.graph import StateGraph, START, END
from state import AgentState

# Import nodes
from router import router_node
from dispatcher import dispatcher_node
from inspector import inspector_node
from compressor import compression_node
from writer import writer_node

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("dispatcher", dispatcher_node)
workflow.add_node("inspector", inspector_node)
workflow.add_node("compressor", compression_node)
workflow.add_node("writer", writer_node)


def route_after_dispatch(state: AgentState):
    """Skip inspector/compressor for direct_answer (no docs to rerank)."""
    agents = state.get("selected_agents", [])
    if agents == ["direct_answer"]:
        return "writer"
    return "inspector"


# ── Graph edges ──────────────────────────────────────────────────────────────
# Flow:
#   START → router (keyword split + route, μs)
#         → dispatcher (parallel agents per sub-query)
#         → inspector (rerank per sub-query) → compressor → writer → END
#
# Direct-answer shortcut:
#   ... → dispatcher → writer → END  (skips inspector/compressor)

workflow.add_edge(START, "router")
workflow.add_edge("router", "dispatcher")
workflow.add_conditional_edges("dispatcher", route_after_dispatch)
workflow.add_edge("inspector", "compressor")
workflow.add_edge("compressor", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

if __name__ == "__main__":
    initial_state = {
        "question": "What does the Quran say about fasting and what are the hadith about charity?",
        "current_agent": "",
        "selected_agents": [],
        "retrieved_context": [],
        "reranker_score": 0.0,
        "sub_queries": [],
        "sub_query_agents": {},
        "final_answer": "",
        "messages": [],
        "loop_step": 0
    }
    final_state = app.invoke(initial_state)
    print(final_state["final_answer"])