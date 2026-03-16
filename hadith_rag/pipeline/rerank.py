# ============================================================
# YaqeenAI — Reranker Module [LOCAL]
# ============================================================
# Reranks retrieved hadith candidates using BAAI/bge-reranker-v2-m3.
# This runs LOCALLY on CPU — only processing ~20 query-document pairs.
#
# DECISION: BGE reranker over Jina reranker because:
# - Explicitly trained/evaluated on Arabic MIRACL benchmark
# - Stronger Arabic morphological robustness
# - Standard in Arabic IR research pipelines
# - ~2-3 sec on CPU for 20 pairs (acceptable latency)

import logging
from typing import Optional

from sentence_transformers import CrossEncoder

from pipeline.config import settings
from pipeline.retrieve import RetrievedHadith

logger = logging.getLogger(__name__)


class HadithReranker:
    """
    Cross-encoder reranker for hadith retrieval results.

    Uses BAAI/bge-reranker-v2-m3 to re-score (query, document) pairs
    and return the top-K most relevant hadiths.

    The cross-encoder sees both query and document together, enabling
    much finer-grained relevance scoring than bi-encoder similarity.
    This is crucial for Arabic text where subtle morphological
    differences change meaning significantly.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL
        logger.info(f"Loading reranker model: {self.model_name}")

        self.model = CrossEncoder(
            self.model_name,
            max_length=512,  # Sufficient for hadith matn (most are <300 tokens)
            device="cpu",    # Explicit CPU — no GPU available locally
        )

        logger.info("Reranker model loaded successfully (CPU mode)")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedHadith],
        top_k: Optional[int] = None,
    ) -> list[RetrievedHadith]:
        """
        Rerank candidate hadiths by cross-encoder relevance score.

        Args:
            query: The user's search query (Arabic or English).
            candidates: List of RetrievedHadith from ChromaDB (typically 20).
            top_k: Number of top results to return (default: RERANK_TOP_K=5).

        Returns:
            Top-K hadiths sorted by relevance (highest first).
        """
        top_k = top_k or settings.RERANK_TOP_K

        if not candidates:
            logger.warning("No candidates to rerank")
            return []

        # Build (query, document) pairs for cross-encoder scoring
        pairs = [(query, hadith.text_ar) for hadith in candidates]

        logger.info(
            f"Reranking {len(pairs)} pairs with {self.model_name} "
            f"(selecting top {top_k})"
        )

        # Score all pairs
        scores = self.model.predict(
            pairs,
            batch_size=len(pairs),  # Process all at once (only ~20 pairs)
            show_progress_bar=False,
        )

        # Attach scores and sort
        scored_hadiths = list(zip(scores, candidates))
        scored_hadiths.sort(key=lambda x: x[0], reverse=True)  # Highest score first

        # Return top-K
        reranked = [hadith for _, hadith in scored_hadiths[:top_k]]

        logger.info(
            f"Reranking complete. Top score: {scored_hadiths[0][0]:.4f}, "
            f"Bottom of top-{top_k}: {scored_hadiths[min(top_k-1, len(scored_hadiths)-1)][0]:.4f}"
        )

        return reranked


# Module-level convenience
_reranker: Optional[HadithReranker] = None


def get_reranker() -> HadithReranker:
    """Get or create the singleton reranker."""
    global _reranker
    if _reranker is None:
        _reranker = HadithReranker()
    return _reranker


def rerank(
    query: str,
    candidates: list[RetrievedHadith],
    top_k: Optional[int] = None,
) -> list[RetrievedHadith]:
    """Convenience function for reranking."""
    return get_reranker().rerank(query=query, candidates=candidates, top_k=top_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick test: just load the model to verify it works
    reranker = HadithReranker()
    print(f"Reranker loaded: {reranker.model_name}")
    print("Model ready for inference on CPU.")
