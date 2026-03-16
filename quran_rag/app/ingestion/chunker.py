# ============================================================
# YaqeenAI — Quran Chunking Strategy
# ============================================================
# Converts raw Quran API data into canonical DocumentChunks.
#
# Strategy: 1 Ayah = 1 Chunk (as per RAG_EMBEDDING_RERANKING_GENERATION_STRATEGY.md)
# Each chunk contains: original text, normalized text, metadata.

from __future__ import annotations

from loguru import logger

from app.core.config import get_settings
from app.models.schemas import (
    ContentType,
    Language,
    ChunkMetadata,
    DocumentChunk,
    QuranSurah,
)
from app.preprocessing.arabic_normalizer import ArabicTextNormalizer


class QuranChunker:
    """
    Converts QuranSurah objects into DocumentChunks ready for indexing.
    
    Chunking strategy for Quran:
    - 1 ayah = 1 chunk (ayahs are naturally atomic units)
    - Original text preserved for display (with tashkeel)
    - Normalized text for BM25 indexing
    - Prefixed text for embedding (instruction-aware models)
    """

    def __init__(self):
        self._normalizer = ArabicTextNormalizer()
        self._settings = get_settings()

    def chunk_surah(self, surah: QuranSurah) -> list[DocumentChunk]:
        """
        Convert a complete surah into a list of DocumentChunks.
        Each ayah becomes one chunk.
        """
        chunks = []

        edition_id = surah.edition.identifier if surah.edition else "quran-uthmani"
        edition_name = surah.edition.english_name if surah.edition else "Uthmani"
        edition_type = surah.edition.type if surah.edition else "quran"
        edition_language = surah.edition.language if surah.edition else "ar"
        language = Language.ARABIC if edition_language == "ar" else Language.ENGLISH
        content_type = self._resolve_content_type(edition_type, language)

        for ayah in surah.ayahs:
            chunk = self._create_ayah_chunk(
                ayah_text=ayah.text,
                content_type=content_type,
                surah_number=surah.number,
                surah_name_arabic=surah.name,
                surah_name_english=surah.english_name,
                ayah_number_in_surah=ayah.number_in_surah,
                ayah_number_global=ayah.number,
                juz=ayah.juz,
                manzil=ayah.manzil,
                page=ayah.page,
                ruku=ayah.ruku,
                hizb_quarter=ayah.hizb_quarter,
                sajda=ayah.sajda,
                revelation_type=surah.revelation_type,
                edition_identifier=edition_id,
                edition_name=edition_name,
                edition_type=edition_type,
                language=language,
            )
            chunks.append(chunk)

        logger.info(
            f"Chunked surah {surah.number} ({surah.english_name}): "
            f"{len(chunks)} ayah chunks"
        )
        return chunks

    def chunk_multiple_surahs(self, surahs: list[QuranSurah]) -> list[DocumentChunk]:
        """Chunk multiple surahs into a flat list of DocumentChunks."""
        all_chunks = []
        for surah in surahs:
            all_chunks.extend(self.chunk_surah(surah))
        logger.info(f"Total chunks from {len(surahs)} surahs: {len(all_chunks)}")
        return all_chunks

    def _create_ayah_chunk(
        self,
        ayah_text: str,
        content_type: ContentType,
        surah_number: int,
        surah_name_arabic: str,
        surah_name_english: str,
        ayah_number_in_surah: int,
        ayah_number_global: int,
        juz: int,
        manzil: int,
        page: int,
        ruku: int,
        hizb_quarter: int,
        sajda: bool,
        revelation_type: str,
        edition_identifier: str,
        edition_name: str,
        edition_type: str,
        language: Language,
    ) -> DocumentChunk:
        """Create a single DocumentChunk from ayah data."""
        cleaned_text = self._normalizer.normalize_unicode(ayah_text).strip()

        # Generate unique chunk ID
        chunk_id = self._build_chunk_id(
            content_type=content_type,
            edition_identifier=edition_identifier,
            language=language,
            surah_number=surah_number,
            ayah_number_in_surah=ayah_number_in_surah,
        )

        # Normalize text for different uses
        text_normalized = self._normalizer.normalize_for_bm25(cleaned_text)

        # Build rich text for embedding (include surah context)
        context_prefix = self._build_context_prefix(
            content_type=content_type,
            language=language,
            edition_name=edition_name,
            surah_name_arabic=surah_name_arabic,
            surah_name_english=surah_name_english,
            ayah_number_in_surah=ayah_number_in_surah,
        )
        embedding_text = self._normalizer.normalize_for_embedding(cleaned_text)
        text_for_embedding = f"{self._settings.embedding_prefix_passage}{context_prefix}{embedding_text}"

        metadata = ChunkMetadata(
            content_type=content_type,
            language=language,
            surah_number=surah_number,
            surah_name_arabic=surah_name_arabic,
            surah_name_english=surah_name_english,
            ayah_number_in_surah=ayah_number_in_surah,
            ayah_number_global=ayah_number_global,
            ayah_ref=f"{surah_number}:{ayah_number_in_surah}",
            juz=juz,
            manzil=manzil,
            page=page,
            ruku=ruku,
            hizb_quarter=hizb_quarter,
            sajda=sajda,
            revelation_type=revelation_type,
            edition_identifier=edition_identifier,
            edition_name=edition_name,
            source=self._resolve_source_label(content_type, edition_name, edition_type),
            source_url=f"https://api.quranhub.com/v1/ayah/{surah_number}:{ayah_number_in_surah}/{edition_identifier}",
        )

        return DocumentChunk(
            chunk_id=chunk_id,
            text=cleaned_text,  # Original display text after removing invisible controls
            text_normalized=text_normalized,  # For BM25
            text_for_embedding=text_for_embedding,  # For vector embedding
            metadata=metadata,
        )

    @staticmethod
    def _resolve_content_type(edition_type: str, language: Language) -> ContentType:
        if edition_type == "tafsir":
            return ContentType.TAFSIR
        if edition_type == "translation" and language == Language.ENGLISH:
            return ContentType.QURAN_TRANSLATION
        return ContentType.QURAN_AYAH

    @staticmethod
    def _build_chunk_id(
        *,
        content_type: ContentType,
        edition_identifier: str,
        language: Language,
        surah_number: int,
        ayah_number_in_surah: int,
    ) -> str:
        if content_type == ContentType.QURAN_AYAH and edition_identifier == "quran-uthmani":
            return f"quran_{language.value}_{surah_number}_{ayah_number_in_surah}"

        safe_edition = edition_identifier.replace(".", "_").replace("-", "_")
        return (
            f"{content_type.value}_{safe_edition}_{language.value}_"
            f"{surah_number}_{ayah_number_in_surah}"
        )

    @staticmethod
    def _build_context_prefix(
        *,
        content_type: ContentType,
        language: Language,
        edition_name: str,
        surah_name_arabic: str,
        surah_name_english: str,
        ayah_number_in_surah: int,
    ) -> str:
        if content_type == ContentType.TAFSIR:
            return (
                f"تفسير {edition_name} لسورة {surah_name_arabic} آية {ayah_number_in_surah}: "
                if language == Language.ARABIC
                else f"Tafsir {edition_name} for Surah {surah_name_english} Ayah {ayah_number_in_surah}: "
            )

        if content_type == ContentType.QURAN_TRANSLATION:
            return f"Quran translation for Surah {surah_name_english} Ayah {ayah_number_in_surah}: "

        return (
            f"سورة {surah_name_arabic} آية {ayah_number_in_surah}: "
            if language == Language.ARABIC
            else f"Surah {surah_name_english} Ayah {ayah_number_in_surah}: "
        )

    @staticmethod
    def _resolve_source_label(
        content_type: ContentType,
        edition_name: str,
        edition_type: str,
    ) -> str:
        if content_type == ContentType.TAFSIR:
            return f"Tafsir::{edition_name}"
        if edition_type == "translation":
            return f"QuranTranslation::{edition_name}"
        return "Quran"
