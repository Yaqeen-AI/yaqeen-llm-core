# ============================================================
# YaqeenAI — Query Embedding via Jina REST API [LOCAL]
# ============================================================
# Embeds user queries using Jina Embeddings v3 REST API.
# This runs LOCALLY — no GPU needed (cloud API call).
#
# Uses task='retrieval.query' to match against passage embeddings
# built on Colab with task='retrieval.passage'.

import logging
from typing import Optional

import httpx

from pipeline.config import settings

logger = logging.getLogger(__name__)


class JinaQueryEmbedder:
    """
    Embeds queries using Jina Embeddings v3 REST API.

    This is the LOCAL component — uses the cloud API, no GPU needed.
    The Colab/Kaggle notebook handles batch passage embedding with the
    local model on GPU.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self.api_key = api_key or settings.JINA_API_KEY
        self.model = model or settings.JINA_EMBEDDING_MODEL
        self.dimensions = dimensions or settings.JINA_EMBEDDING_DIM
        self.api_url = settings.JINA_API_URL
        self._client = httpx.Client(timeout=30.0)

        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY is required. Set it in .env or pass it directly. "
                "Get your key at https://jina.ai/"
            )

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string using Jina API.

        Args:
            query: The user's search query (Arabic or English).
                   No preprocessing needed — Jina handles raw text.

        Returns:
            List of floats — 1024-dimensional embedding vector.

        Raises:
            httpx.HTTPStatusError: If the API call fails.
            ValueError: If the response is malformed.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "task": "retrieval.query",  # CRITICAL: must be 'retrieval.query' for queries
            "dimensions": self.dimensions,
            "input": [query],
        }

        logger.info(
            f"Embedding query via Jina API: model={self.model}, "
            f"task=retrieval.query, dims={self.dimensions}"
        )

        response = self._client.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()

        if "data" not in data or len(data["data"]) == 0:
            raise ValueError(f"Unexpected Jina API response: {data}")

        embedding = data["data"][0]["embedding"]
        logger.info(f"Query embedded successfully: {len(embedding)} dimensions")

        return embedding

    def embed_batch(self, queries: list[str]) -> list[list[float]]:
        """
        Embed multiple queries in a single API call.

        Args:
            queries: List of query strings.

        Returns:
            List of embedding vectors (same order as input).

        Raises:
            httpx.HTTPStatusError: If the API call fails.
            ValueError: If the response is malformed or count mismatches input.
        """
        if not queries:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "task": "retrieval.query",
            "dimensions": self.dimensions,
            "input": queries,
        }

        response = self._client.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()

        data = response.json()

        if "data" not in data or not isinstance(data["data"], list):
            raise ValueError(f"Unexpected Jina API response (missing 'data'): {data}")

        if len(data["data"]) == 0:
            raise ValueError(
                f"Jina API returned empty 'data' list for {len(queries)} input(s)"
            )

        if len(data["data"]) != len(queries):
            raise ValueError(
                f"Jina API returned {len(data['data'])} embeddings "
                f"but {len(queries)} queries were sent"
            )

        embeddings = []
        for i, item in enumerate(data["data"]):
            if "embedding" not in item or not isinstance(item["embedding"], list):
                raise ValueError(
                    f"Jina API response item {i} missing 'embedding' field: {item}"
                )
            embeddings.append(item["embedding"])

        logger.info(f"Batch embedded {len(embeddings)} queries")

        return embeddings

    def close(self) -> None:
        """Close the persistent HTTP client."""
        self._client.close()


# Module-level convenience instance
_embedder: Optional[JinaQueryEmbedder] = None


def get_embedder() -> JinaQueryEmbedder:
    """Get or create the singleton query embedder."""
    global _embedder
    if _embedder is None:
        _embedder = JinaQueryEmbedder()
    return _embedder


def embed_query(query: str) -> list[float]:
    """Convenience function to embed a single query."""
    return get_embedder().embed_query(query)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    query = sys.argv[1] if len(sys.argv) > 1 else "ما حكم الصلاة"
    vector = embed_query(query)
    print(f"Query: {query}")
    print(f"Embedding dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
