"""
Dispatcher — Colony Instantiation & Parallel Worker Execution.

For each sub-query from the Manager:
  1. Instantiate an isolated Colony with its own PheromoneTable
  2. Use the colony's probabilistic selection to choose workers
  3. Execute selected workers in parallel via ThreadPoolExecutor
  4. Collect results per-colony for downstream inspection

Also handles periodic inter-colony synchronization.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from state import AgentState
from colony import Colony, is_greeting, tokenize
from synchronizer import synchronize_colonies
from workers.quran_agent import quran_agent_node
from workers.hadith_agent import hadith_agent_node
from workers.fiqh_agent import fiqh_agent_node
from shared_nodes.workers.direct_answer import direct_answer_node

logger = logging.getLogger("mmas.dispatcher")

# ── Worker function registry ──
WORKER_FN_MAP = {
    "quran_agent": quran_agent_node,
    "hadith_agent": hadith_agent_node,
    "fiqh_agent": fiqh_agent_node,
    "direct_answer": direct_answer_node,
}


def _run_worker(fn, query: str) -> list:
    """Execute a single worker and return its documents."""
    state = {"question": query}
    result = fn(state)
    docs = result.get("retrieved_context", [])

    # Tag documents with originating sub-query
    for doc in docs:
        try:
            doc.metadata["sub_query"] = query
        except (AttributeError, TypeError):
            pass

    return docs


def dispatcher_node(state: AgentState):
    """
    Colony Instantiation & Parallel Routing node.

    For each sub-query:
      1. Create a Colony with independent pheromone table
      2. Colony selects workers probabilistically (τ^α · η^β)
      3. Workers execute in parallel
      4. Results tagged by colony for Inspector feedback

    After all colonies execute, performs inter-colony synchronization.

    Sets:
      - colony_results:    {sub_query: {worker: [docs]}}
      - colony_pheromones: {sub_query: pheromone snapshot}
      - sub_query_agents:  {sub_query: [selected_workers]}
      - selected_agents:   merged unique list
      - retrieved_context: all documents aggregated
    """
    sub_queries = state.get("sub_queries", [state["question"]])

    # ── Create colonies ──
    colonies: Dict[str, Colony] = {}
    for i, sq in enumerate(sub_queries):
        colonies[sq] = Colony(sub_query=sq, colony_id=f"colony_{i}")

    # ── Build work items per colony ──
    all_work = []  # (sub_query, worker_name, fn)
    sub_query_agents = {}

    for sq, colony in colonies.items():
        selected = colony.select_workers(min_workers=1)
        sub_query_agents[sq] = selected
        for worker_name in selected:
            fn = WORKER_FN_MAP.get(worker_name)
            if fn:
                all_work.append((sq, worker_name, fn))

    # ── Log dispatch plan ──
    total_tasks = len(all_work)
    unique_workers = sorted(set(name for _, name, _ in all_work))
    print(
        f"   [Dispatcher] -> {total_tasks} tasks across "
        f"{len(colonies)} colon{'ies' if len(colonies) > 1 else 'y'}, "
        f"workers: [{', '.join(unique_workers)}]"
    )

    # ── Execute in parallel ──
    colony_results: Dict[str, Dict[str, list]] = {sq: {} for sq in sub_queries}
    all_docs = []

    if total_tasks <= 1 and all_work:
        # Single task — no thread overhead
        sq, worker_name, fn = all_work[0]
        docs = _run_worker(fn, sq)
        colony_results[sq][worker_name] = docs
        all_docs.extend(docs)
    elif all_work:
        with ThreadPoolExecutor(max_workers=min(total_tasks, 6)) as executor:
            futures = {}
            for sq, worker_name, fn in all_work:
                future = executor.submit(_run_worker, fn, sq)
                futures[future] = (sq, worker_name)

            for future in as_completed(futures):
                sq, worker_name = futures[future]
                try:
                    docs = future.result()
                    colony_results[sq][worker_name] = docs
                    all_docs.extend(docs)
                    print(
                        f"   [Dispatcher] -> {worker_name} "
                        f"('{sq[:35]}...') returned {len(docs)} docs"
                    )
                except Exception as e:
                    logger.error(f"Worker {worker_name} failed: {e}")
                    print(f"   [Dispatcher] -> {worker_name} FAILED: {e}")
                    colony_results[sq][worker_name] = []

    # ── Inter-colony synchronization ──
    if len(colonies) > 1:
        synchronize_colonies(list(colonies.values()))

    # ── Collect pheromone snapshots ──
    colony_pheromones = {
        sq: colony.pheromone.snapshot()
        for sq, colony in colonies.items()
    }

    # ── Merge all selected agents ──
    all_agents = set()
    for agents in sub_query_agents.values():
        all_agents.update(agents)
    domain_agents = all_agents - {"direct_answer"}
    if domain_agents:
        all_agents.discard("direct_answer")

    print(f"   [Dispatcher] -> Total documents collected: {len(all_docs)}")

    return {
        "colony_results": colony_results,
        "colony_pheromones": colony_pheromones,
        "sub_query_agents": sub_query_agents,
        "selected_agents": sorted(all_agents),
        "current_agent": sorted(all_agents)[0] if all_agents else "direct_answer",
        "retrieved_context": all_docs,
    }
