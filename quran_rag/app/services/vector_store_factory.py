from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from app.models.schemas import ChunkMetadata, DocumentChunk, RetrievalResult


@runtime_checkable
class VectorStoreProtocol(Protocol):
    collection_name: str

    def create_collection(self, dimension: int | None = None) -> None: ...

    def delete_collection(self) -> None: ...

    def collection_info(self) -> dict: ...

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        batch_size: int = 100,
    ) -> int: ...

    def semantic_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 30,
        language_filter: str | None = None,
        surah_filter: int | None = None,
        juz_filter: int | None = None,
        content_type_filter: str | None = None,
        edition_identifier_filter: str | None = None,
    ) -> list[RetrievalResult]: ...

    def get_all_texts_and_ids(
        self,
    ) -> tuple[list[str], list[str], list[str], list[ChunkMetadata]]: ...

    def get_payload_by_chunk_id(self, chunk_id: str) -> dict | None: ...

    def get_by_ayah_ref(
        self,
        *,
        ayah_ref: str,
        language: str | None = None,
        content_type: str | None = None,
        edition_identifier: str | None = None,
    ) -> list[RetrievalResult]: ...

    def get_adjacent_chunks(
        self,
        *,
        surah_number: int,
        ayah_number_in_surah: int,
        language: str,
        content_type: str | None = None,
        edition_identifier: str | None = None,
        window_size: int = 1,
    ) -> list[RetrievalResult]: ...


def build_vector_store() -> VectorStoreProtocol:
    from app.services.chroma_store import ChromaVectorStore

    return ChromaVectorStore()
