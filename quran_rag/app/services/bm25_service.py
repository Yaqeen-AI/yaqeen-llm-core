# ============================================================
# YaqeenAI — BM25 Sparse Retrieval Service
# ============================================================
# Keyword-based retrieval using BM25 (Okapi BM25).
# Critical for exact term matching that semantic search misses.
#
# Why BM25 matters for Islamic text:
# - Exact Quran phrase matching (e.g., "آية الكرسي")
# - Hadith narrator names (must be exact)
# - Surah names, Islamic terminology
# - Classical Arabic terms that embedding models may not capture
#
# BM25 Parameters tuned for short Quran ayahs:
# - k1 = 1.5 (moderate term frequency saturation)
# - b  = 0.5 (reduced length normalization — ayahs are short)

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.models.schemas import ChunkMetadata, RetrievalResult
from app.preprocessing.arabic_normalizer import ArabicTextNormalizer


class BM25RetrievalService:
    """
    BM25 sparse retrieval engine for Arabic text.
    
    Lifecycle:
        1. Build index from corpus texts (offline, during ingestion)
        2. Search with normalized queries (online, per request)
    
    The index is held in-memory. For 500K documents this uses ~200-400 MB RAM.
    For production, consider a persistent BM25 index (Elasticsearch/OpenSearch).
    """

    def __init__(self):
        self._settings = get_settings()
        self._normalizer = ArabicTextNormalizer()
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: list[str] = []
        self._raw_texts: list[str] = []  # Original texts for results
        self._normalized_texts: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._metadata_by_chunk_id: dict[str, ChunkMetadata] = {}
        self._is_built = False

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def corpus_size(self) -> int:
        return len(self._chunk_ids)

    def build_index(
        self,
        texts: list[str],
        chunk_ids: list[str],
        raw_texts: Optional[list[str]] = None,
        metadatas: Optional[list[ChunkMetadata]] = None,
    ) -> None:
        """
        Build the BM25 index from a list of texts.
        
        Args:
            texts: Normalized texts (output of normalize_for_bm25)
            chunk_ids: Corresponding chunk IDs (same order)
            raw_texts: Original un-normalized texts for display (optional)
            metadatas: Chunk metadata in the same order as chunk_ids (optional)
        """
        assert len(texts) == len(chunk_ids), "texts and chunk_ids must match"
        if metadatas is not None:
            assert len(metadatas) == len(chunk_ids), "metadatas and chunk_ids must match"

        logger.info(f"Building BM25 index from {len(texts)} documents...")

        # Tokenize for BM25
        self._normalized_texts = texts
        self._tokenized_corpus = [
            self._normalizer.tokenize_document_for_bm25(text) for text in texts
        ]
        self._chunk_ids = chunk_ids
        self._raw_texts = raw_texts if raw_texts else texts
        self._metadata_by_chunk_id = {}
        if metadatas:
            self._metadata_by_chunk_id = {
                chunk_id: metadata for chunk_id, metadata in zip(chunk_ids, metadatas)
            }

        # Build BM25 with tuned parameters for short Quran ayahs
        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self._settings.bm25_k1,
            b=self._settings.bm25_b,
        )
        self._is_built = True

        logger.info(
            f"BM25 index built: {len(texts)} docs, "
            f"k1={self._settings.bm25_k1}, b={self._settings.bm25_b}"
        )

    def build_from_chunks(self, chunks: list) -> None:
        """
        Build BM25 index directly from DocumentChunk objects.
        Convenience method used during ingestion.
        """
        texts = [c.text_normalized for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]
        raw_texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        self.build_index(texts, chunk_ids, raw_texts, metadatas)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        language_filter: Optional[str] = None,
        surah_filter: Optional[int] = None,
        juz_filter: Optional[int] = None,
        content_type_filter: Optional[str] = None,
        edition_identifier_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """
        Search the BM25 index with a query.
        
        Args:
            query: Raw user query (will be normalized internally)
            top_k: Number of results (default: from settings)
            language_filter: Optional language filter ("ar"/"en")
            surah_filter: Optional surah number filter
            juz_filter: Optional juz number filter
            content_type_filter: Optional content type filter
            
        Returns:
            List of RetrievalResult ordered by BM25 score (descending)
        """
        if not self._is_built:
            logger.warning("BM25 index not built yet. Returning empty results.")
            return []

        top_k = top_k or self._settings.bm25_top_k

        # Normalize and tokenize the query
        query_tokens = self._normalizer.tokenize_query_for_bm25(query)

        if not query_tokens:
            logger.warning(f"Query produced no tokens after normalization: '{query}'")
            return []

        effective_content_type = content_type_filter

        # Get BM25 scores for all documents
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue  # Skip zero-score results

            chunk_id = self._chunk_ids[idx]
            metadata = self._metadata_by_chunk_id.get(chunk_id)
            has_filters = any(
                f is not None
                for f in (
                    language_filter,
                    surah_filter,
                    juz_filter,
                    effective_content_type,
                    edition_identifier_filter,
                )
            )
            if metadata is not None:
                if not self._matches_filters(
                    metadata=metadata,
                    language_filter=language_filter,
                    surah_filter=surah_filter,
                    juz_filter=juz_filter,
                    content_type_filter=effective_content_type,
                    edition_identifier_filter=edition_identifier_filter,
                ):
                    continue
            elif has_filters:
                # If filter(s) requested but metadata unavailable, do not leak out-of-scope results.
                continue

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=self._raw_texts[idx],
                    score=score,
                    metadata=metadata if metadata is not None else ChunkMetadata(
                        content_type="quran_ayah",
                        language="ar",
                    ),
                    retrieval_method="bm25",
                )
            )

        logger.debug(f"BM25 search: query='{query[:50]}...' → {len(results)} results")
        return results

    def search_exact_text(
        self,
        query: str,
        top_k: Optional[int] = None,
        language_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        edition_identifier_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """
        Notebook-style exact/phrase text fallback before semantic retrieval.
        """
        if not self._is_built:
            return []

        top_k = top_k or self._settings.bm25_top_k
        normalized_query = self._normalizer.normalize_for_bm25_query(query)
        if not normalized_query:
            return []

        query_tokens = self._normalizer.tokenize_query_for_bm25(query)
        allow_phrase_match = len(query_tokens) >= 3 and len(normalized_query) >= 12
        results = []

        for idx, chunk_id in enumerate(self._chunk_ids):
            metadata = self._metadata_by_chunk_id.get(chunk_id)
            if metadata is not None and not self._matches_filters(
                metadata=metadata,
                language_filter=language_filter,
                content_type_filter=content_type_filter,
                edition_identifier_filter=edition_identifier_filter,
            ):
                continue

            normalized_text = self._normalized_texts[idx]
            is_exact = normalized_query == normalized_text
            is_phrase = allow_phrase_match and f" {normalized_query} " in f" {normalized_text} "
            if not (is_exact or is_phrase):
                continue

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=self._raw_texts[idx],
                    score=1.0 if is_exact else 0.96,
                    metadata=metadata if metadata is not None else ChunkMetadata(
                        content_type="quran_ayah",
                        language="ar",
                    ),
                    retrieval_method="exact_text" if is_exact else "phrase_text",
                )
            )

            if len(results) >= top_k:
                break

        return results

    @staticmethod
    def _matches_filters(
        metadata: ChunkMetadata,
        language_filter: Optional[str] = None,
        surah_filter: Optional[int] = None,
        juz_filter: Optional[int] = None,
        content_type_filter: Optional[str] = None,
        edition_identifier_filter: Optional[str] = None,
    ) -> bool:
        """Check whether chunk metadata satisfies request filters."""
        if language_filter and metadata.language.value != language_filter:
            return False
        if surah_filter is not None and metadata.surah_number != surah_filter:
            return False
        if juz_filter is not None and metadata.juz != juz_filter:
            return False
        if content_type_filter and metadata.content_type.value != content_type_filter:
            return False
        if edition_identifier_filter and metadata.edition_identifier != edition_identifier_filter:
            return False
        return True

    def get_doc_score(self, query: str, doc_index: int) -> float:
        """Get BM25 score for a specific document (for debugging)."""
        if not self._is_built:
            return 0.0
        query_tokens = self._normalizer.tokenize_query_for_bm25(query)
        scores = self._bm25.get_scores(query_tokens)
        return float(scores[doc_index])

    def search_contains(
        self,
        query: str,
        top_k: Optional[int] = None,
        language_filter: Optional[str] = None,
        surah_filter: Optional[int] = None,
        juz_filter: Optional[int] = None,
        content_type_filter: Optional[str] = None,
        edition_identifier_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """
        Substring (contains) fallback search used when BM25 scores are all zero.

        This happens with rare / single-word queries on small test corpora where
        the token appears in too few documents to register a positive BM25 score.

        Strategy:
          1. Normalise the query with BM25-level normalisation.
          2. Scan all corpus texts for substring match.
          3. Score matched docs proportionally to the fraction of query tokens
             they contain (like a soft TF hit) and rank accordingly.

        This is fast on the test corpus (~300 ayahs) and acceptable up to ~50K docs.
        Beyond that, you should use Elasticsearch's `match` query instead.
        """
        if not self._is_built:
            return []

        top_k = top_k or self._settings.bm25_top_k
        query_tokens = self._normalizer.tokenize_query_for_bm25(query)
        effective_content_type = content_type_filter

        if not query_tokens:
            return []

        unique_query_tokens = list(dict.fromkeys(query_tokens))
        scored: list[tuple[int, float]] = []
        for idx, doc_tokens in enumerate(self._tokenized_corpus):
            doc_token_set = set(doc_tokens)
            matched_tokens = [tok for tok in unique_query_tokens if tok in doc_token_set]
            if matched_tokens:
                score = len(matched_tokens) / len(unique_query_tokens)  # [0, 1]

                chunk_id = self._chunk_ids[idx]
                metadata = self._metadata_by_chunk_id.get(chunk_id)
                if metadata is not None and not self._matches_filters(
                    metadata=metadata,
                    language_filter=language_filter,
                    surah_filter=surah_filter,
                    juz_filter=juz_filter,
                    content_type_filter=effective_content_type,
                    edition_identifier_filter=edition_identifier_filter,
                ):
                    continue

                scored.append((idx, score))

        # Sort by score descending, keep top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        results = []
        for idx, score in scored:
            chunk_id = self._chunk_ids[idx]
            metadata = self._metadata_by_chunk_id.get(chunk_id)
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=self._raw_texts[idx],
                    score=score,
                    metadata=metadata if metadata is not None else ChunkMetadata(
                        content_type="quran_ayah",
                        language="ar",
                    ),
                    retrieval_method="bm25_contains",
                )
            )

        logger.debug(
            f"BM25 contains fallback: query='{query[:50]}' → {len(results)} results"
        )
        return results
