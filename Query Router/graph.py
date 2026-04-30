from langgraph.graph import StateGraph, START, END
from state import AgentState

# Import nodes
from supervisor import supervisor_node
from memory import memory_management_node
from manager import manager_node
from workers.quran_agent import quran_agent_node
from workers.hadith_agent import hadith_agent_node
from workers.fiqh_agent import fiqh_agent_node
from workers.direct_answer import direct_answer_node
from inspector import inspector_node
from compressor import compression_node
from writer import writer_node

workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("memory_flush", memory_management_node)
workflow.add_node("manager", manager_node)
workflow.add_node("quran_agent", quran_agent_node)
workflow.add_node("hadith_agent", hadith_agent_node)
workflow.add_node("fiqh_agent", fiqh_agent_node)
workflow.add_node("direct_answer", direct_answer_node)
workflow.add_node("inspector", inspector_node)
workflow.add_node("compressor", compression_node)
workflow.add_node("writer", writer_node)

def route_from_manager(state: AgentState):
    agent = state.get("current_agent", "quran_agent")
    allowed = {"quran_agent", "hadith_agent", "fiqh_agent", "direct_answer"}
    return agent if agent in allowed else "quran_agent"

workflow.add_edge(START, "supervisor")
workflow.add_edge("supervisor", "memory_flush")
workflow.add_edge("memory_flush", "manager")
workflow.add_conditional_edges("manager", route_from_manager)

# 3. Direct all RAG agents to the inspector node
workflow.add_edge("quran_agent", "inspector")
workflow.add_edge("hadith_agent", "inspector")
workflow.add_edge("fiqh_agent", "inspector")

workflow.add_edge("direct_answer", "writer")
workflow.add_edge("inspector", "compressor")
workflow.add_edge("compressor", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

if __name__ == "__main__":
    initial_state = {
        "question": "What does the text say about the vacation policy?",
        "current_agent": "",
        "retrieved_context": [],
        "reranker_score": 0.0,
        "sub_queries": [],
        "final_answer": "",
        "messages": [], 
        "loop_step": 0
    }
    final_state = app.invoke(initial_state)
    print(final_state["final_answer"])