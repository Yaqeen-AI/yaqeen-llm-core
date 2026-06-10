"""
Module-level Qdrant client singletons.

Qdrant local (file) mode locks the storage directory — only one QdrantClient
instance per path is allowed per process. Import from here instead of
instantiating QdrantClient directly so all components share the same handle.
"""

import atexit

from qdrant_client import QdrantClient
from core.config import QDRANT_CACHE_PATH, QDRANT_PATH, FIQH_QDRANT_URL, FIQH_QDRANT_API_KEY

_registry: dict[str, QdrantClient] = {}


def _get(path: str) -> QdrantClient:
    if path not in _registry:
        _registry[path] = QdrantClient(path=path)
    return _registry[path]


def rag_client() -> QdrantClient:
    if FIQH_QDRANT_URL:
        key = "fiqh_web_client"
        if key not in _registry:
            _registry[key] = QdrantClient(
                url=FIQH_QDRANT_URL,
                api_key=FIQH_QDRANT_API_KEY or None,
                timeout=30,
            )
        return _registry[key]
    return _get(QDRANT_PATH)


def cache_client() -> QdrantClient:
    return _get(QDRANT_CACHE_PATH)


def _close_all() -> None:
    for client in list(_registry.values()):
        try:
            client.close()
        except Exception:
            pass
    _registry.clear()


atexit.register(_close_all)
