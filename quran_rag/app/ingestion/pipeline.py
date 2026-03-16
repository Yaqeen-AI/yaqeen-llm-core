# ============================================================
# YaqeenAI — Ingestion Pipeline
# ============================================================
# End-to-end ingestion: API fetch → chunk → embed → index
# This module is used by the Colab notebook to populate the Chroma store.

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from app.core.config import get_settings
from app.ingestion.quran_api_client import QuranApiClient
from app.ingestion.chunker import QuranChunker
from app.models.schemas import DocumentChunk
from app.services.embedding_service import EmbeddingService
from app.services.bm25_service import BM25RetrievalService
from app.services.vector_store_factory import build_vector_store

ARABIC_QURAN_EDITIONS = ["quran-uthmani"]
ARABIC_TAFSIR_EDITIONS = ["ar.tabari", "ar.muyassar", "ar.mukhtasar"]
DEFAULT_INGESTION_EDITIONS = (
    ARABIC_QURAN_EDITIONS + ARABIC_TAFSIR_EDITIONS
)


async def fetch_complete_quran_data(
    edition: str = "quran-uthmani",
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
) -> list:
    """
    Fetch the entire Quran in one request and cache the response on disk.

    This is the preferred path for the Colab ingestion notebook because it is
    faster, produces a single cache artifact, and avoids 114 sequential calls.
    """
    cache_path = Path(cache_dir) if cache_dir else Path("data/cache")
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"complete_quran_{edition}.json"

    if cache_file.exists() and not force_refresh:
        logger.info("Loading complete Quran from cache: {}", cache_file)
        with open(cache_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        from app.models.schemas import QuranSurah

        return [QuranSurah(**item) for item in raw_data]

    async with QuranApiClient() as client:
        surahs = await client.get_complete_quran(edition)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(
            [surah.model_dump(by_alias=True) for surah in surahs],
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Cached complete Quran to {}", cache_file)
    return surahs


async def fetch_selected_editions_data(
    editions: list[str],
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
) -> dict[str, list]:
    """
    Fetch and cache the exact edition set used by the notebook build.

    The returned mapping is keyed by edition identifier so the notebook can
    report per-edition counts and the ingestion pipeline can merge them into one
    multilingual Quran+tafsir corpus.
    """
    edition_to_surahs: dict[str, list] = {}

    for edition in editions:
        edition_to_surahs[edition] = await fetch_complete_quran_data(
            edition=edition,
            cache_dir=cache_dir,
            force_refresh=force_refresh,
        )

    return edition_to_surahs


def run_ingestion_pipeline(
    editions: Optional[list[str]] = None,
    cache_dir: str = "data/cache",
    rebuild_collection: bool = False,
) -> tuple[list[DocumentChunk], BM25RetrievalService]:
    """
    Full corpus ingestion pipeline used by the Colab notebook:
        1. Fetch Arabic Quran + selected Arabic tafsir editions
        2. Chunk into DocumentChunks (1 ayah = 1 chunk)
        3. Embed the merged corpus
        4. Index into the persisted Chroma vector store
        5. Build BM25 index
    """
    settings = get_settings()
    editions = editions or DEFAULT_INGESTION_EDITIONS

    logger.info("=" * 60)
    logger.info("YAQEEN AI — QURAN INGESTION PIPELINE")
    logger.info("Scope: arabic_quran + selected_tafsir_editions")
    logger.info(f"Editions: {editions}")
    logger.info(f"Embedding model: {settings.embedding_model_name}")
    logger.info("=" * 60)

    # ─── Step 1: Fetch data ───
    logger.info("Step 1: Fetching Quran data...")
    edition_to_surahs = asyncio.run(fetch_selected_editions_data(editions, cache_dir))
    surahs = [surah for edition_surahs in edition_to_surahs.values() for surah in edition_surahs]
    total_ayahs = sum(len(s.ayahs) for s in surahs)
    logger.info(
        "Fetched {} edition layers with {} total surah objects and {} total ayah chunks before chunking",
        len(edition_to_surahs),
        len(surahs),
        total_ayahs,
    )

    # ─── Step 2: Chunk ───
    logger.info("Step 2: Chunking ayahs...")
    chunker = QuranChunker()
    chunks = chunker.chunk_multiple_surahs(surahs)
    logger.info(f"Created {len(chunks)} chunks")

    # ─── Step 3: Embed ───
    logger.info("Step 3: Embedding chunks...")
    embedding_service = EmbeddingService()

    # Extract texts for embedding (already prefixed by chunker)
    texts_for_embedding = [c.text_for_embedding for c in chunks]
    embeddings = embedding_service.encode_passages(texts_for_embedding)
    logger.info(f"Embedded {len(chunks)} chunks → shape {embeddings.shape}")

    # Save embeddings checkpoint
    checkpoint_dir = Path("data/embeddings")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.save(checkpoint_dir / "quran_embeddings.npy", embeddings)
    logger.info(f"Saved embeddings to {checkpoint_dir / 'quran_embeddings.npy'}")

    # ─── Step 4: Index into vector store ───
    logger.info("Step 4: Indexing into vector store...")
    vector_store = build_vector_store()

    if rebuild_collection:
        try:
            vector_store.delete_collection()
        except Exception:
            pass

    vector_store.create_collection(dimension=embedding_service.dimension)
    upserted = vector_store.upsert_chunks(chunks, embeddings)
    logger.info("Indexed {} chunks into Chroma", upserted)

    # ─── Step 5: Build BM25 index ───
    logger.info("Step 5: Building BM25 index...")
    bm25_service = BM25RetrievalService()
    bm25_service.build_from_chunks(chunks)
    logger.info(f"BM25 index ready: {bm25_service.corpus_size} documents")

    # ─── Report ───
    info = vector_store.collection_info()
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE!")
    logger.info(f"Collection: {info['name']}")
    logger.info(f"Points: {info['points_count']}")
    logger.info(f"BM25 docs: {bm25_service.corpus_size}")
    logger.info("=" * 60)

    return chunks, bm25_service
