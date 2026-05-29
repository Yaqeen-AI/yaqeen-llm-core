# ============================================================
# YaqeenAI — Reranker Module [JINA API]
# ============================================================
# Reranks retrieved hadith candidates using Jina's hosted reranker API.

import logging
from typing import Optional

import httpx

from pipeline.config import settings
from pipeline.retrieve import RetrievedHadith

logger = logging.getLogger(__name__)


class HadithReranker:
    """
    Jina API reranker for hadith retrieval results.

    Uses Jina's multilingual reranker to re-score query/document pairs and
    return the top-K most relevant hadiths.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        self.model_name = model_name or settings.RERANKER_MODEL
        self.api_key = api_key or settings.JINA_API_KEY
        self.api_url = api_url or settings.JINA_RERANK_API_URL

        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY is required for reranking. Set it in .env or pass it directly. "
                "Get your key at https://jina.ai/"
            )

        logger.info(f"Jina reranker ready: model={self.model_name}, url={self.api_url}")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedHadith],
        top_k: Optional[int] = None,
    ) -> list[RetrievedHadith]:
        """
        Rerank candidate hadiths by Jina relevance score.

        Args:
            query: The user's search query (Arabic or English).
            candidates: List of retrieved hadith candidates (typically 20).
            top_k: Number of top results to return (default: RERANK_TOP_K=5).

        Returns:
            Top-K hadiths sorted by relevance (highest first).
        """
        top_k = top_k or settings.RERANK_TOP_K
        top_k = min(top_k, len(candidates))

        if not candidates:
            logger.warning("No candidates to rerank")
            return []

        logger.info(
            f"Reranking {len(candidates)} documents with {self.model_name} "
            f"(selecting top {top_k})"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [hadith.text_ar for hadith in candidates],
            "top_n": top_k,
            "return_documents": False,
        }

        with httpx.Client(timeout=60.0) as client:
            response = client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()

        data = response.json()
        results = data.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Unexpected Jina rerank response (missing results): {data}")
        if not results:
            raise ValueError(f"Jina rerank returned no results for {len(candidates)} candidates")

        scored_hadiths: list[tuple[float, RetrievedHadith]] = []
        for item in results:
            index = item.get("index")
            score = item.get("relevance_score")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                raise ValueError(f"Unexpected Jina rerank result index: {item}")
            if not isinstance(score, (int, float)):
                raise ValueError(f"Unexpected Jina rerank result score: {item}")
            scored_hadiths.append((float(score), candidates[index]))

        scored_hadiths.sort(key=lambda x: x[0], reverse=True)
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
    reranker = HadithReranker()
    print(f"Reranker loaded: {reranker.model_name}")
    print("Jina reranker client ready.")
