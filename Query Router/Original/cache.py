"""
Semantic Cache — Global Pheromone Memory.

Intercepts incoming queries and checks for semantic matches.
On HIT: returns cached answer immediately (short-circuits the pipeline).
On MISS: forwards the query to the Manager.

After the Writer produces a final answer, this cache stores it for future hits.

Backed by Qdrant Cloud for semantic similarity search.
"""

import os
import uuid
import hashlib
import logging
import re
from typing import Optional

logger = logging.getLogger("mmas.cache")

# ═══════════════════════════════════════════════════════════════════════════════
# Cache interface
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_COLLECTION = "mmas_query_cache"
SEMANTIC_THRESHOLD = 0.80
EMBED_DIM = 1024       # Jina v3 dimension


class SemanticCache:
    """
    Global Pheromone Memory — semantic cache backed by Qdrant.

    Uses Jina Embeddings v3 for query vectorization and Qdrant cosine
    similarity for semantic matching.  Acts as the first line of defense
    before any colony processing begins.
    """

    def __init__(self):
        self._q = None
        self._jina_key = os.getenv("JINA_API_KEY", "")
        self._init_qdrant()

    def _init_qdrant(self) -> None:
        """Initialize Qdrant client for cache storage."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            url = os.getenv("QDRANT_URL", "")
            key = os.getenv("QDRANT_API_KEY", "")

            if url and key:
                self._q = QdrantClient(url=url, api_key=key)
            else:
                # Local fallback
                _script_dir = os.path.dirname(os.path.abspath(__file__))
                _project_root = os.path.dirname(_script_dir)
                cache_path = os.path.join(_project_root, "qdrant_cache")
                self._q = QdrantClient(path=cache_path)

            if not self._q.collection_exists(CACHE_COLLECTION):
                self._q.create_collection(
                    collection_name=CACHE_COLLECTION,
                    vectors_config=VectorParams(
                        size=EMBED_DIM, distance=Distance.COSINE
                    ),
                )
            logger.info(
                f"Semantic cache ready (collection={CACHE_COLLECTION}, "
                f"threshold={SEMANTIC_THRESHOLD:.0%})"
            )
        except Exception as e:
            logger.warning(f"Semantic cache disabled: {e}")
            self._q = None

    def _normalize(self, query: str) -> str:
        """Arabic-aware normalization for cache keys."""
        text = query.lower().strip()
        text = re.sub(r"[^\w\s\u0600-\u06FF]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _embed(self, text: str) -> Optional[list]:
        """Embed text using Jina Embeddings v3."""
        if not self._jina_key:
            return None
        try:
            import requests
            resp = requests.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self._jina_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "jina-embeddings-v3",
                    "input": [text],
                    "dimensions": EMBED_DIM,
                    "task": "retrieval.query",
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except Exception:
            return None

    # ── Public API ───────────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[str]:
        """
        Check cache for a semantic match.

        Returns the cached answer string on HIT, or None on MISS.
        """
        if not self._q:
            return None

        normalized = self._normalize(query)
        vec = self._embed(normalized)
        if not vec:
            return None

        try:
            hits = self._q.query_points(
                collection_name=CACHE_COLLECTION,
                query=vec,
                limit=1,
                score_threshold=SEMANTIC_THRESHOLD,
                with_payload=True,
            ).points

            if hits:
                answer = hits[0].payload.get("answer", "")
                logger.info(
                    f"Cache HIT (score={hits[0].score:.3f}): "
                    f"'{query[:40]}...' -> cached answer"
                )
                return answer
        except Exception as e:
            logger.debug(f"Cache lookup failed: {e}")

        return None

    def set(self, query: str, answer: str) -> None:
        """
        Store a query-answer pair in the semantic cache.
        """
        if not self._q or not answer:
            return

        normalized = self._normalize(query)
        vec = self._embed(normalized)
        if not vec:
            return

        try:
            from qdrant_client.models import PointStruct
            self._q.upsert(
                collection_name=CACHE_COLLECTION,
                points=[PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "original_query": query,
                        "normalized_query": normalized,
                        "answer": answer,
                    },
                )],
            )
            logger.info(f"Cache SET: '{query[:40]}...'")
        except Exception as e:
            logger.debug(f"Cache store failed: {e}")
