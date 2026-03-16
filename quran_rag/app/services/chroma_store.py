from __future__ import annotations

from typing import Optional

import chromadb
import numpy as np
from loguru import logger

from app.core.config import get_settings
from app.models.schemas import ChunkMetadata, DocumentChunk, RetrievalResult
from app.preprocessing.arabic_normalizer import ArabicTextNormalizer


class ChromaVectorStore:
    """
    Persistent Chroma-backed vector store.

    This backend is the default for the full-Quran workflow because it is easy
    to build inside Colab, persist to disk, zip, and move back to the local app
    without running an external database service.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        settings = get_settings()
        self._persist_directory = persist_directory or settings.chroma_persist_directory
        self._collection_name = collection_name or settings.chroma_collection_name
        self._client = chromadb.PersistentClient(path=self._persist_directory)
        self._collection = None
        self._normalizer = ArabicTextNormalizer()

        logger.info(
            "Connected to Chroma persistent store at {}",
            self._persist_directory,
        )

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def create_collection(self, dimension: Optional[int] = None) -> None:
        """
        Create or load the Chroma collection.

        Chroma does not require the embedding dimension to be declared up front.
        The method keeps a `dimension` parameter only because the notebook
        ingestion pipeline passes it in during indexing.
        """
        del dimension
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Chroma collection ready: {}", self._collection_name)

    def delete_collection(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = None
        logger.warning("Deleted Chroma collection '{}'", self._collection_name)

    def collection_info(self) -> dict:
        collection = self._get_collection()
        return {
            "name": self._collection_name,
            "points_count": collection.count(),
            "vectors_count": collection.count(),
            "status": "green",
            "config": f"persist_directory={self._persist_directory}",
        }

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        batch_size: int = 128,
    ) -> int:
        assert len(chunks) == len(embeddings), (
            f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
            f"must have same length"
        )

        collection = self._get_collection()
        total = len(chunks)
        upserted = 0

        for start in range(0, total, batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]

            ids = [chunk.chunk_id for chunk in batch_chunks]
            documents = [chunk.text for chunk in batch_chunks]
            metadatas = []

            for chunk in batch_chunks:
                metadata = chunk.metadata.model_dump(mode="json")
                metadata["chunk_id"] = chunk.chunk_id
                metadata["text_normalized"] = chunk.text_normalized
                metadatas.append(metadata)

            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=batch_embeddings.tolist(),
            )
            upserted += len(ids)

            if upserted % 500 == 0 or upserted >= total:
                logger.info("Upserted {}/{} vectors into Chroma", upserted, total)

        return upserted

    def semantic_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 30,
        language_filter: Optional[str] = None,
        surah_filter: Optional[int] = None,
        juz_filter: Optional[int] = None,
        content_type_filter: Optional[str] = None,
        edition_identifier_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        collection = self._get_collection()
        where = self._build_where(
            language=language_filter,
            surah_number=surah_filter,
            juz=juz_filter,
            content_type=content_type_filter,
            edition_identifier=edition_identifier_filter,
        )

        response = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=document or "",
                    score=self._distance_to_score(distance),
                    metadata=self._metadata_to_chunk_metadata(metadata),
                    retrieval_method="semantic",
                )
            )

        return results

    def get_all_texts_and_ids(
        self,
    ) -> tuple[list[str], list[str], list[str], list[ChunkMetadata]]:
        collection = self._get_collection()
        response = collection.get(include=["documents", "metadatas"])

        ids = response.get("ids", [])
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])

        texts = []
        raw_texts = []
        chunk_ids = []
        canonical_metadatas = []

        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            chunk_ids.append(chunk_id)
            raw_texts.append(document or "")
            texts.append(self._normalizer.normalize_for_bm25_document(document or ""))
            canonical_metadatas.append(self._metadata_to_chunk_metadata(metadata))

        logger.info(
            "Retrieved {} records from Chroma collection '{}'",
            len(chunk_ids),
            self._collection_name,
        )
        return texts, chunk_ids, raw_texts, canonical_metadatas

    def get_payload_by_chunk_id(self, chunk_id: str) -> Optional[dict]:
        collection = self._get_collection()
        response = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        ids = response.get("ids", [])

        if not ids:
            return None

        metadata = (response.get("metadatas") or [{}])[0] or {}
        document = (response.get("documents") or [""])[0] or ""
        return {
            "chunk_id": chunk_id,
            "text": document,
            **metadata,
        }

    def get_by_ayah_ref(
        self,
        *,
        ayah_ref: str,
        language: Optional[str] = None,
        content_type: Optional[str] = None,
        edition_identifier: Optional[str] = None,
    ) -> list[RetrievalResult]:
        collection = self._get_collection()
        response = collection.get(
            where=self._build_where(
                language=language,
                content_type=content_type,
                edition_identifier=edition_identifier,
                ayah_ref=ayah_ref,
            ),
            include=["documents", "metadatas"],
        )

        ids = response.get("ids", [])
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])

        results = []
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=document or "",
                    score=1.0,
                    metadata=self._metadata_to_chunk_metadata(metadata),
                    retrieval_method="exact_ref",
                )
            )

        results.sort(
            key=lambda item: (
                item.metadata.surah_number or 0,
                item.metadata.ayah_number_in_surah or 0,
            )
        )
        return results

    def get_adjacent_chunks(
        self,
        *,
        surah_number: int,
        ayah_number_in_surah: int,
        language: str,
        content_type: Optional[str] = None,
        edition_identifier: Optional[str] = None,
        window_size: int = 1,
    ) -> list[RetrievalResult]:
        collection = self._get_collection()
        lower = max(1, ayah_number_in_surah - window_size)
        upper = ayah_number_in_surah + window_size

        conditions = [
            {"language": {"$eq": language}},
            {"surah_number": {"$eq": surah_number}},
            {"ayah_number_in_surah": {"$gte": lower}},
            {"ayah_number_in_surah": {"$lte": upper}},
        ]
        if content_type:
            conditions.append({"content_type": {"$eq": content_type}})
        if edition_identifier:
            conditions.append({"edition_identifier": {"$eq": edition_identifier}})

        response = collection.get(
            where={"$and": conditions},
            include=["documents", "metadatas"],
        )

        ids = response.get("ids", [])
        documents = response.get("documents", [])
        metadatas = response.get("metadatas", [])

        results = []
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=document or "",
                    score=1.0,
                    metadata=self._metadata_to_chunk_metadata(metadata),
                    retrieval_method="context_window",
                )
            )

        results.sort(
            key=lambda item: (
                item.metadata.surah_number or 0,
                item.metadata.ayah_number_in_surah or 0,
            )
        )
        return results

    def _get_collection(self):
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @staticmethod
    def _distance_to_score(distance: Optional[float]) -> float:
        if distance is None:
            return 0.0
        return float(1.0 - distance)

    @staticmethod
    def _metadata_to_chunk_metadata(metadata: dict) -> ChunkMetadata:
        return ChunkMetadata(
            content_type=metadata.get("content_type", "quran_ayah"),
            language=metadata.get("language", "ar"),
            surah_number=metadata.get("surah_number"),
            surah_name_arabic=metadata.get("surah_name_arabic"),
            surah_name_english=metadata.get("surah_name_english"),
            ayah_number_in_surah=metadata.get("ayah_number_in_surah"),
            ayah_number_global=metadata.get("ayah_number_global"),
            ayah_ref=metadata.get("ayah_ref"),
            juz=metadata.get("juz"),
            manzil=metadata.get("manzil"),
            page=metadata.get("page"),
            ruku=metadata.get("ruku"),
            hizb_quarter=metadata.get("hizb_quarter"),
            sajda=metadata.get("sajda"),
            revelation_type=metadata.get("revelation_type"),
            edition_identifier=metadata.get("edition_identifier"),
            edition_name=metadata.get("edition_name"),
            tafsir_author=metadata.get("tafsir_author"),
            source=metadata.get("source"),
            source_family=metadata.get("source_family"),
            source_url=metadata.get("source_url"),
        )

    @staticmethod
    def _build_where(
        *,
        language: Optional[str] = None,
        surah_number: Optional[int] = None,
        juz: Optional[int] = None,
        content_type: Optional[str] = None,
        edition_identifier: Optional[str] = None,
        ayah_ref: Optional[str] = None,
    ) -> Optional[dict]:
        conditions = []

        if language:
            conditions.append({"language": {"$eq": language}})
        if surah_number is not None:
            conditions.append({"surah_number": {"$eq": surah_number}})
        if juz is not None:
            conditions.append({"juz": {"$eq": juz}})
        if content_type:
            conditions.append({"content_type": {"$eq": content_type}})
        if edition_identifier:
            conditions.append({"edition_identifier": {"$eq": edition_identifier}})
        if ayah_ref:
            conditions.append({"ayah_ref": {"$eq": ayah_ref}})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}
