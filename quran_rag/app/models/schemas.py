# ============================================================
# YaqeenAI — Domain Models
# ============================================================
# Canonical data models for the entire RAG pipeline.
# These models are used across ingestion, indexing, and retrieval.

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ───────────────────────────────────────────────────────────
# Enums
# ───────────────────────────────────────────────────────────

class ContentType(str, Enum):
    """Type of Islamic content chunk."""
    QURAN_AYAH = "quran_ayah"
    QURAN_TRANSLATION = "quran_translation"
    TAFSIR = "tafsir"
    HADITH = "hadith"
    HADITH_SHARH = "hadith_sharh"


class Language(str, Enum):
    """Supported languages."""
    ARABIC = "ar"
    ENGLISH = "en"


class RevelationType(str, Enum):
    """Quran surah revelation type."""
    MECCAN = "Meccan"
    MEDINAN = "Medinan"


# ───────────────────────────────────────────────────────────
# Quran Data Models (from API)
# ───────────────────────────────────────────────────────────

class QuranEdition(BaseModel):
    """Quran edition/translation metadata."""
    identifier: str
    language: str
    name: str
    english_name: str = Field(alias="englishName", default="")
    format: str = "text"
    type: str = "quran"
    direction: str = "rtl"

    model_config = {"populate_by_name": True}


class QuranAyah(BaseModel):
    """Single ayah from the Quran API."""
    number: int = Field(description="Global ayah number (1-6236)")
    text: str = Field(description="Ayah text")
    number_in_surah: int = Field(alias="numberInSurah", description="Ayah number within surah")
    juz: int = Field(description="Juz number (1-30)")
    manzil: int = Field(description="Manzil number (1-7)")
    page: int = Field(description="Mushaf page number")
    ruku: int = Field(description="Ruku number")
    hizb_quarter: int = Field(alias="hizbQuarter", description="Hizb quarter number")
    sajda: bool = Field(default=False, description="Whether this ayah has a sajda")

    model_config = {"populate_by_name": True}


class QuranSurah(BaseModel):
    """Complete surah data from the Quran API."""
    number: int = Field(description="Surah number (1-114)")
    name: str = Field(description="Arabic name of the surah")
    english_name: str = Field(alias="englishName", default="")
    english_name_translation: str = Field(alias="englishNameTranslation", default="")
    revelation_type: str = Field(alias="revelationType", default="")
    number_of_ayahs: int = Field(alias="numberOfAyahs", default=0)
    ayahs: list[QuranAyah] = Field(default_factory=list)
    edition: Optional[QuranEdition] = None

    model_config = {"populate_by_name": True}


# ───────────────────────────────────────────────────────────
# Canonical Chunk Model (for indexing)
# ───────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    """
    Metadata attached to every chunk in the vector store.
    This is stored in the persistent vector store for filtering and citations.
    """
    content_type: ContentType
    language: Language
    surah_number: Optional[int] = None
    surah_name_arabic: Optional[str] = None
    surah_name_english: Optional[str] = None
    ayah_number_in_surah: Optional[int] = None
    ayah_number_global: Optional[int] = None
    ayah_ref: Optional[str] = None
    juz: Optional[int] = None
    manzil: Optional[int] = None
    page: Optional[int] = None
    ruku: Optional[int] = None
    hizb_quarter: Optional[int] = None
    sajda: Optional[bool] = None
    revelation_type: Optional[str] = None
    edition_identifier: Optional[str] = None
    edition_name: Optional[str] = None
    tafsir_author: Optional[str] = None
    source: Optional[str] = None
    source_family: Optional[str] = None
    source_url: Optional[str] = None


class DocumentChunk(BaseModel):
    """
    The canonical chunk that goes into the vector store.
    This is the universal unit for ALL content types.
    """
    chunk_id: str = Field(description="Unique ID: e.g. quran_ar_2_255 (surah:ayah)")
    text: str = Field(description="Original text (preserved with diacritics for display)")
    text_normalized: str = Field(description="Normalized text for BM25 indexing")
    text_for_embedding: str = Field(description="Text with passage: prefix for embedding")
    metadata: ChunkMetadata


# ───────────────────────────────────────────────────────────
# Retrieval Models
# ───────────────────────────────────────────────────────────

class RetrievalResult(BaseModel):
    """A single retrieval result from any retrieval method."""
    chunk_id: str
    text: str
    score: float
    metadata: ChunkMetadata
    retrieval_method: str = Field(description="How this was retrieved: semantic/bm25/hybrid/reranked")


class RetrievalRequest(BaseModel):
    """Input to the retrieval pipeline."""
    query: str = Field(description="User's search query (Arabic or English)")
    language: Optional[Language] = Field(default=None, description="Preferred language filter")
    content_type_filter: Optional[ContentType] = Field(
        default=None,
        description="Restrict retrieval to one content type such as quran_ayah or tafsir",
    )
    edition_identifier_filter: Optional[str] = Field(
        default=None,
        description="Restrict retrieval to a specific edition identifier",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    use_reranking: bool = Field(default=True, description="Whether to apply reranking")
    use_hybrid: bool = Field(default=True, description="Whether to use hybrid (semantic+BM25)")
    surah_filter: Optional[int] = Field(
        default=None,
        ge=1,
        le=114,
        description="Filter by surah number",
    )
    juz_filter: Optional[int] = Field(
        default=None,
        ge=1,
        le=30,
        description="Filter by juz number",
    )


class RetrievalResponse(BaseModel):
    """Output from the retrieval pipeline."""
    query: str
    results: list[RetrievalResult]
    total_candidates_semantic: int = 0
    total_candidates_bm25: int = 0
    total_after_fusion: int = 0
    total_after_reranking: int = 0
    pipeline_steps: list[str] = Field(
        default_factory=list,
        description="Log of pipeline steps executed"
    )
    latency_ms: float = 0.0


# ───────────────────────────────────────────────────────────
# Generation Models
# ───────────────────────────────────────────────────────────

class AnswerCitation(BaseModel):
    """Compact citation payload returned with generated answers."""
    chunk_id: str
    surah_number: Optional[int] = None
    ayah_number_in_surah: Optional[int] = None
    ayah_ref: Optional[str] = None
    surah_name_english: Optional[str] = None
    content_type: Optional[ContentType] = None
    edition_identifier: Optional[str] = None
    edition_name: Optional[str] = None
    score: float
    text: str


class AnswerRequest(BaseModel):
    """High-level answer request that sits on top of retrieval."""
    query: str = Field(description="User question")
    language: Optional[Language] = Field(default=None, description="Preferred answer language")
    content_type_filter: Optional[ContentType] = Field(default=None)
    edition_identifier_filter: Optional[str] = Field(default=None)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of retrieved chunks")
    context_window: int = Field(
        default=1,
        ge=0,
        le=5,
        description="How many neighboring ayahs to include on each side of a hit",
    )
    surah_filter: Optional[int] = Field(
        default=None,
        ge=1,
        le=114,
        description="Optional surah filter",
    )
    juz_filter: Optional[int] = Field(
        default=None,
        ge=1,
        le=30,
        description="Optional juz filter",
    )
    use_hybrid: bool = Field(default=True)
    use_reranking: bool = Field(default=True)


class AnswerResponse(BaseModel):
    """Generated answer plus the retrieval trace used to build it."""
    query: str
    answer: str
    model_name: str
    citations: list[AnswerCitation] = Field(default_factory=list)
    retrieval: RetrievalResponse
    prompt_preview: str = ""
