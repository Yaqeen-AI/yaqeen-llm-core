# ============================================================
# YaqeenAI — Embedding Service
# ============================================================
# Handles encoding text into dense vectors.
#
# The service now supports two execution modes:
#   1. Standard SentenceTransformer models
#   2. Jina embeddings v3 through `trust_remote_code=True`
#
# For the full-Quran project the default model is `jinaai/jina-embeddings-v3`,
# which exposes task-specific embeddings:
#   - retrieval.passage for indexed ayah chunks
#   - retrieval.query   for user questions

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


class EmbeddingService:
    """
    Singleton embedding service.
    Loads the model once and reuses for all encode calls.
    
    Jina v3 needs task-aware encoding, while classic retrieval models often rely
    on asymmetric text prefixes.  This wrapper hides that branching so the rest
    of the codebase can simply ask for passage/query embeddings.
    """

    _instance: Optional["EmbeddingService"] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is not None:
            return  # Already initialized

        settings = get_settings()
        self._model_name = settings.embedding_model_name
        self._dimension = settings.embedding_dimension
        self._batch_size = settings.embedding_batch_size
        self._max_length = settings.embedding_max_length
        self._task_passage = settings.embedding_task_passage
        self._task_query = settings.embedding_task_query
        self._prefix_passage = settings.embedding_prefix_passage
        self._prefix_query = settings.embedding_prefix_query
        self._is_jina_v3 = "jina-embeddings-v3" in self._model_name.lower()

        logger.info(f"Loading embedding model: {self._model_name}")
        self._model = SentenceTransformer(
            self._model_name,
            trust_remote_code=self._is_jina_v3,
        )
        actual_dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedding model loaded. Dimension: {actual_dim} "
            f"(configured: {self._dimension})"
        )

        # Auto-correct dimension if model reports different
        if actual_dim != self._dimension:
            logger.warning(
                f"Dimension mismatch! Model reports {actual_dim}, "
                f"config says {self._dimension}. Using model's dimension."
            )
            self._dimension = actual_dim

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_passages(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode document passages into dense vectors.
        
        For instruction-aware models (e5-large), texts should already
        have "passage: " prefix. If using this with raw text, prefix
        will NOT be auto-added (the chunker handles it).
        
        Args:
            texts: List of passage texts (pre-prefixed)
            show_progress: Show tqdm progress bar
            
        Returns:
            numpy array of shape (len(texts), dimension), normalized
        """
        logger.info(f"Encoding {len(texts)} passages (batch_size={self._batch_size})")

        encode_kwargs = self._build_encode_kwargs(
            is_query=False,
            show_progress=show_progress,
        )
        embeddings = self._model.encode(texts, **encode_kwargs)

        logger.info(f"Encoded {len(texts)} passages → shape {embeddings.shape}")
        return embeddings

    def encode_query(self, query: str, add_prefix: bool = True) -> np.ndarray:
        """
        Encode a single query into a dense vector.
        
        Adds "query: " prefix for instruction-aware models.
        
        Args:
            query: Raw user query text
            add_prefix: Whether to add the query prefix
            
        Returns:
            numpy array of shape (dimension,), normalized
        """
        if add_prefix and not self._is_jina_v3:
            query = f"{self._prefix_query}{query}"

        encode_kwargs = self._build_encode_kwargs(is_query=True, show_progress=False)
        embedding = self._model.encode(query, **encode_kwargs)
        return embedding

    def encode_queries_batch(
        self,
        queries: list[str],
        add_prefix: bool = True,
    ) -> np.ndarray:
        """Batch encode multiple queries."""
        if add_prefix and not self._is_jina_v3:
            queries = [f"{self._prefix_query}{q}" for q in queries]

        encode_kwargs = self._build_encode_kwargs(is_query=True, show_progress=False)
        return self._model.encode(queries, **encode_kwargs)

    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        passage_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and passages.
        Since embeddings are L2-normalized, dot product = cosine similarity.
        """
        return np.dot(passage_embeddings, query_embedding)

    def _build_encode_kwargs(self, is_query: bool, show_progress: bool) -> dict:
        """
        Build model-specific kwargs for encode().

        Jina v3 expects explicit retrieval tasks and supports dimensionality
        truncation.  Other sentence-transformer models keep the classic encode
        signature and use text prefixes for asymmetry instead.
        """
        kwargs = {
            "batch_size": self._batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": show_progress,
        }

        if self._is_jina_v3:
            kwargs.update(
                {
                    "task": self._task_query if is_query else self._task_passage,
                    "truncate_dim": self._dimension,
                }
            )

            # The remote-code model exposes this attribute when supported.
            if hasattr(self._model, "max_seq_length"):
                self._model.max_seq_length = self._max_length

        return kwargs
