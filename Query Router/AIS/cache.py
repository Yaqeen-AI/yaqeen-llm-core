"""
AIS Memory Cells (Semantic Cache).

Acts as the first line of defense (Antigen Recognition & Memory Cell Activation).
Uses Jina embeddings and Qdrant Cloud or local cache to check for semantic similarity.
"""

import os
import uuid
import logging
from typing import Optional

logger = logging.getLogger("ais.cache")

CACHE_COLLECTION = "ais_memory_cells"
EMBED_DIM = 1024       # Jina v3 default dimension


class MemoryCellsCache:
    """
    Semantic Cache representing Memory Cells in the AIS architecture.
    """

    def __init__(self):
        self._q = None
        self._jina_key = os.getenv("JINA_API_KEY", "")
        
        # Load configurable threshold (default to 0.85 as specified)
        try:
            self.threshold = float(os.getenv("MEMORY_CELL_THRESHOLD", "0.85"))
        except ValueError:
            self.threshold = 0.85
            
        self._init_qdrant()

    def _init_qdrant(self) -> None:
        """Initialize Qdrant client for memory cell storage."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            url = os.getenv("QDRANT_URL", "")
            key = os.getenv("QDRANT_API_KEY", "")

            if url and key:
                self._q = QdrantClient(url=url, api_key=key)
            else:
                # Local fallback inside AIS directory
                _script_dir = os.path.dirname(os.path.abspath(__file__))
                cache_path = os.path.join(_script_dir, "qdrant_cache")
                self._q = QdrantClient(path=cache_path)

            if not self._q.collection_exists(CACHE_COLLECTION):
                self._q.create_collection(
                    collection_name=CACHE_COLLECTION,
                    vectors_config=VectorParams(
                        size=EMBED_DIM, distance=Distance.COSINE
                    ),
                )
            logger.info(
                f"AIS Memory Cells ready (threshold={self.threshold:.2f})"
            )
        except Exception as e:
            logger.warning(f"AIS Memory Cells disabled: {e}")
            self._q = None

    def _embed(self, text: str) -> Optional[list]:
        """Embed query using Jina Embeddings v3."""
        if not self._jina_key:
            logger.warning("JINA_API_KEY not found. Cannot embed antigen.")
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
        except Exception as e:
            logger.error(f"Jina embedding request failed: {e}")
            return None

    def get(self, query: str) -> Optional[str]:
        """
        Query Memory Cells for a semantic hit.
        Returns the cached answer string if cosine similarity exceeds threshold, else None.
        """
        if not self._q:
            return None

        vec = self._embed(query)
        if not vec:
            return None

        try:
            hits = self._q.query_points(
                collection_name=CACHE_COLLECTION,
                query=vec,
                limit=1,
                score_threshold=self.threshold,
                with_payload=True,
            ).points

            if hits:
                answer = hits[0].payload.get("answer", "")
                logger.info(
                    f"Memory Cell HIT (affinity={hits[0].score:.3f}): "
                    f"'{query[:40]}...'"
                )
                print(f"   [Memory Cells] -> HIT! Cosine similarity {hits[0].score:.3f} >= threshold {self.threshold:.2f}")
                return answer
        except Exception as e:
            logger.debug(f"Memory cell lookup failed: {e}")

        print(f"   [Memory Cells] -> MISS. Initiating primary immune response...")
        return None

    def set(self, query: str, answer: str) -> None:
        """
        Store a new Memory Cell (cache update) upon successful response maturation.
        """
        if not self._q or not answer:
            return

        vec = self._embed(query)
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
                        "answer": answer,
                    },
                )],
            )
            logger.info(f"Memory Cell Formed: '{query[:40]}...'")
        except Exception as e:
            logger.debug(f"Failed to save memory cell: {e}")
