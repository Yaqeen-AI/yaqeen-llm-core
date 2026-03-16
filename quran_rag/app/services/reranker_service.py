# ============================================================
# YaqeenAI — Reranking Service
# ============================================================
# Cross-encoder reranking — THE #1 accuracy lever in the RAG pipeline.
# Takes (query, candidate) pairs and scores true relevance.
# 
# Impact: +10-20% precision improvement over vector similarity alone.
#
# Models:
#   - Testing:    cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, English-heavy)
#   - Production: BAAI/bge-reranker-v2-m3 (multilingual, SOTA)

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.models.schemas import RetrievalResult


class RerankerService:
    """
    Cross-encoder reranking service.
    
    Unlike bi-encoders (embedding), cross-encoders take the FULL (query, document)
    pair as input and produce a single relevance score. This is far more accurate
    because the model can attend to fine-grained interactions between query and doc.
    
    Trade-off: Much slower (can't be pre-computed), so only used on top-K candidates
    from initial retrieval (typically 20 candidates → reranked to top 5).
    
    Latency:
        - CPU: ~100-200ms per pair → 20 pairs = 2-4 sec
        - GPU: ~10-20ms per pair → 20 pairs = 200-400ms
    """

    _instance: Optional["RerankerService"] = None
    _model: Optional[CrossEncoder] = None

    def __new__(cls) -> "RerankerService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is not None:
            return  # Already initialized

        settings = get_settings()
        self._model_name = settings.reranker_model_name

        logger.info(f"Loading reranker model: {self._model_name}")
        # bge-reranker models benefit from apply_softmax=False — they output
        # raw logits that already have good relative ordering. Sigmoid squashes
        # everything into 0.49-0.51 for irrelevant pairs, hiding discrimination.
        self._model = CrossEncoder(self._model_name, max_length=512)
        # Detect whether the model outputs normalised scores natively
        self._is_bge = "bge-reranker" in self._model_name.lower()
        logger.info(f"Reranker model loaded. BGE mode: {self._is_bge}")

    @property
    def model_name(self) -> str:
        return self._model_name

    def supports_language(self, language_code: str) -> bool:
        """
        Return whether the current reranker is suitable for a language code.

        The default testing model (`ms-marco-MiniLM`) is English-focused and
        degrades Arabic reranking quality, so we disable it for Arabic queries.
        """
        model_name = self._model_name.lower()

        # English-heavy model families.
        if "ms-marco" in model_name and language_code == "ar":
            return False

        # Assume multilingual support for other configured models unless known otherwise.
        return True

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """
        Rerank candidates by cross-encoder relevance scores.
        
        Args:
            query: The user's search query
            candidates: List of RetrievalResult from initial retrieval
            top_k: Number of top results to return after reranking
            
        Returns:
            Reranked list of RetrievalResult (top_k items, highest score first)
        """
        if not candidates:
            return []

        settings = get_settings()
        top_k = top_k or settings.rerank_top_k

        # Build (query, document) pairs for cross-encoder
        pairs = [[query, candidate.text] for candidate in candidates]

        logger.debug(f"Reranking {len(pairs)} candidates...")

        # Score all pairs — raw logits
        raw_scores = self._model.predict(pairs)

        # Score normalisation:
        # - BGE-reranker: outputs logits in roughly [-10, +10].
        #   sigmoid(logit) gives a proper 0→1 probability.  This is the
        #   recommended post-processing per the BGE paper.
        # - ms-marco models: already output near-probability scores.
        #   Applying sigmoid again would over-compress them; skip it.
        if self._is_bge:
            scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=np.float32)))
        else:
            scores = np.array(raw_scores, dtype=np.float32)
            # Clamp to [0, 1] in case model returns values outside that range
            scores = np.clip(scores, 0.0, 1.0)

        # Attach reranking scores and sort
        reranked = []
        for candidate, score in zip(candidates, scores):
            reranked.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    score=float(score),
                    metadata=candidate.metadata,
                    retrieval_method="reranked",
                )
            )

        # Sort by reranking score (descending)
        reranked.sort(key=lambda r: r.score, reverse=True)

        # Return top-k
        return reranked[:top_k]

    def score_pair(self, query: str, document: str) -> float:
        """Score a single (query, document) pair. Useful for debugging."""
        raw = self._model.predict([[query, document]])
        score = float(raw[0]) if hasattr(raw, '__len__') else float(raw)
        if self._is_bge:
            score = float(1.0 / (1.0 + np.exp(-score)))
        return score
