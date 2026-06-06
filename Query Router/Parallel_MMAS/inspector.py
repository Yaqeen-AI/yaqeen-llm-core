"""
Inspector — Evaluates and scores the relevance and quality of retrieved chunks.

Receives all documents from the Dispatcher, groups them by sub-query,
scores each group using cross-encoder reranking, and feeds scores back
to the originating colonies for pheromone updates.

The Inspector's scores drive the MMAS pheromone update loop:
  score → deposit → evaporate → clamp [τ_min, τ_max]
"""

import logging
from typing import Dict, List

from state import AgentState
from colony import Colony
from pheromone import PheromoneTable

logger = logging.getLogger("mmas.inspector")


def _compute_worker_scores(
    colony_results: Dict[str, Dict[str, list]],
    question: str,
) -> Dict[str, Dict[str, float]]:
    """
    Score each worker's output quality per sub-query.

    Strategy:
      1. Try cross-encoder reranking (sentence-transformers)
      2. Fall back to document count heuristic

    Returns: {sub_query: {worker_name: score}}
    """
    # Try to load reranker
    reranker = None
    try:
        from sentence_transformers import CrossEncoder
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
            device=device,
            model_kwargs=model_kwargs,
        )
    except Exception as e:
        logger.warning(f"Reranker unavailable ({e}), using fallback scoring")

    all_scores: Dict[str, Dict[str, float]] = {}

    for sub_query, worker_docs in colony_results.items():
        scores: Dict[str, float] = {}

        for worker_name, docs in worker_docs.items():
            if not docs:
                scores[worker_name] = 0.0
                continue

            # Check for error documents
            error_docs = [d for d in docs if d.metadata.get("error")]
            if error_docs and len(error_docs) == len(docs):
                scores[worker_name] = 0.0
                continue

            valid_docs = [d for d in docs if not d.metadata.get("error")]

            if reranker and valid_docs:
                # Cross-encoder scoring
                pairs = [[sub_query, doc.page_content] for doc in valid_docs]
                try:
                    doc_scores = reranker.predict(pairs, batch_size=16)
                    # Normalize to [0, 1] using sigmoid-like mapping
                    avg_score = sum(doc_scores) / len(doc_scores)
                    # Clamp to [0, 1]
                    normalized = max(0.0, min(1.0, (avg_score + 5) / 10))
                    scores[worker_name] = normalized

                    # Tag individual docs with their rerank scores
                    for doc, score in zip(valid_docs, doc_scores):
                        try:
                            doc.metadata["rerank_score"] = float(score)
                        except (AttributeError, TypeError):
                            pass
                except Exception as e:
                    logger.warning(f"Reranking failed for {worker_name}: {e}")
                    scores[worker_name] = len(valid_docs) / 10.0
            else:
                # Fallback: simple doc count heuristic (more docs = better)
                scores[worker_name] = min(len(valid_docs) / 10.0, 1.0)

        all_scores[sub_query] = scores

    return all_scores


def inspector_node(state: AgentState):
    """
    Inspector node — evaluates retrieved chunks and updates colony pheromones.

    Flow:
      1. Score each worker's output per sub-query
      2. Feed scores back to colony pheromone tables
      3. Enforce MMAS bounds (τ_min, τ_max) during update
      4. Filter/rerank documents by quality
      5. Log pheromone convergence data

    Sets:
      - inspector_scores:   {sub_query: {worker_name: float}}
      - pheromone_log:      list of per-colony pheromone snapshots
      - retrieved_context:  quality-filtered documents
    """
    colony_results = state.get("colony_results", {})
    question = state["question"]

    if not colony_results:
        docs = state.get("retrieved_context", [])
        print(f"   [Inspector] -> No colony results, passing through {len(docs)} docs")
        return {
            "inspector_scores": {},
            "pheromone_log": [],
        }

    # ── Step 1: Score worker outputs ──
    scores = _compute_worker_scores(colony_results, question)

    # ── Step 2 & 3: Pheromone update loop with MMAS enforcement ──
    pheromone_log = []

    for sub_query, worker_scores in scores.items():
        # Reconstruct a temporary colony to update pheromones
        # (In a persistent system, colonies would survive across calls)
        from colony import Colony
        colony = Colony(sub_query=sub_query, colony_id=f"inspect_{sub_query[:20]}")

        log_entry = colony.update_pheromones(worker_scores)
        pheromone_log.append(log_entry)

        # Log MMAS bounding state
        snapshot = colony.pheromone.snapshot()
        logger.info(
            f"[Inspector] Colony '{sub_query[:30]}...' pheromone state: "
            f"τ_min={snapshot['tau_min']:.4f}, τ_max={snapshot['tau_max']:.4f}, "
            f"trails={snapshot['trails']}"
        )
        print(
            f"   [Inspector] -> Colony '{sub_query[:30]}...' scores: "
            f"{', '.join(f'{k}={v:.3f}' for k, v in worker_scores.items())} | "
            f"τ_min={snapshot['tau_min']:.4f}, τ_max={snapshot['tau_max']:.4f}"
        )

    # ── Step 4: Filter documents by quality ──
    docs = state.get("retrieved_context", [])
    approved_docs = []
    threshold = 0.0  # Accept all docs with positive rerank score

    for doc in docs:
        score = doc.metadata.get("rerank_score", None)
        if score is not None and score <= threshold:
            continue  # Skip negatively-scored docs
        approved_docs.append(doc)

    # Sort by rerank score (highest first) if available
    approved_docs.sort(
        key=lambda d: d.metadata.get("rerank_score", 0.0),
        reverse=True,
    )

    print(
        f"   [Inspector] -> {len(docs)} docs in, {len(approved_docs)} approved "
        f"across {len(scores)} sub-quer{'ies' if len(scores) > 1 else 'y'}"
    )

    return {
        "inspector_scores": scores,
        "pheromone_log": pheromone_log,
        "retrieved_context": approved_docs,
    }