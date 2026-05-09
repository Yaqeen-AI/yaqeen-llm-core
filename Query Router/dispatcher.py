"""
Parallel multi-agent dispatcher with per-sub-query routing.

Reads the `sub_query_agents` map from state and executes the matching
worker agents in parallel via ThreadPoolExecutor.  Each agent receives
the specific sub-query it was routed for (not the full compound question).

Falls back to the original behavior (whole question → selected_agents)
when no sub-query decomposition was performed.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from state import AgentState
from workers.quran_agent import quran_agent_node
from workers.hadith_agent import hadith_agent_node
from workers.fiqh_agent import fiqh_agent_node
from workers.direct_answer import direct_answer_node

AGENT_MAP = {
    "quran_agent": quran_agent_node,
    "hadith_agent": hadith_agent_node,
    "fiqh_agent": fiqh_agent_node,
    "direct_answer": direct_answer_node,
}


def _dispatch_single(fn, state: dict, sub_query: str, agent_name: str) -> list:
    """Run one agent with the question overridden to the sub-query text."""
    modified_state = {**state, "question": sub_query}
    result = fn(modified_state)
    docs = result.get("retrieved_context", [])

    # Tag each document with the originating sub-query for traceability
    for doc in docs:
        doc.metadata["sub_query"] = sub_query

    return docs


def dispatcher_node(state: AgentState):
    """
    Fan-out to multiple agents in parallel, fan-in all documents.

    If sub_query_agents is populated (multi-query mode), each sub-query
    is dispatched to its own set of agents with the sub-query as the
    question text.  Otherwise falls back to dispatching the original
    question to selected_agents.
    """
    sub_query_agents = state.get("sub_query_agents", {})
    original_agents = state.get("selected_agents", [])

    if not sub_query_agents and not original_agents:
        original_agents = ["direct_answer"]

    # ── Build work items: list of (sub_query, agent_name, fn) ──
    work_items = []

    if sub_query_agents:
        # Multi-query mode: dispatch per sub-query
        for sub_query, agents in sub_query_agents.items():
            for agent_name in agents:
                fn = AGENT_MAP.get(agent_name)
                if fn:
                    work_items.append((sub_query, agent_name, fn))
    else:
        # Single-query fallback: dispatch original question to all agents
        question = state["question"]
        for agent_name in original_agents:
            fn = AGENT_MAP.get(agent_name, direct_answer_node)
            work_items.append((question, agent_name, fn))

    # ── Log dispatch plan ──
    unique_agents = sorted(set(name for _, name, _ in work_items))
    num_sub_queries = len(sub_query_agents) if sub_query_agents else 1
    print(f"   [Dispatcher] -> {len(work_items)} tasks across "
          f"{num_sub_queries} sub-quer{'ies' if num_sub_queries > 1 else 'y'}, "
          f"agents: [{', '.join(unique_agents)}]")

    all_docs = []

    if len(work_items) == 1:
        # Single task — no thread overhead
        sub_query, agent_name, fn = work_items[0]
        all_docs = _dispatch_single(fn, state, sub_query, agent_name)
    else:
        # Multiple tasks — parallel execution
        with ThreadPoolExecutor(max_workers=min(len(work_items), 6)) as executor:
            futures = {}
            for sub_query, agent_name, fn in work_items:
                future = executor.submit(
                    _dispatch_single, fn, state, sub_query, agent_name
                )
                futures[future] = (sub_query, agent_name)

            for future in as_completed(futures):
                sub_query, agent_name = futures[future]
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                    sq_preview = sub_query[:40]
                    print(f"   [Dispatcher] -> {agent_name} "
                          f"('{sq_preview}...') returned {len(docs)} docs")
                except Exception as e:
                    print(f"   [Dispatcher] -> {agent_name} FAILED: {e}")

    print(f"   [Dispatcher] -> Total documents collected: {len(all_docs)}")
    return {"retrieved_context": all_docs}
