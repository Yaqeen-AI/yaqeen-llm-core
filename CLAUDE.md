# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working with this repo.

## Project Overview

Two independent Arabic RAG systems for Islamic texts, no shared code:

- **`hadith_rag/`** — 155K authenticated hadiths from Dorar Al-Seniyyah. Dense (Jina v3 + ChromaDB) + sparse (TF-IDF char n-gram) retrieval, BGE reranking, Gemini generation with citation grounding.
- **`quran_rag/`** — Quranic ayah themes with tafsir. Dense (Jina v3 local + Qdrant) + sparse (custom BM25) retrieval, RRF fusion, ayah-overlap deduplication.

Both FastAPI apps serving Arabic responses. All content Arabic Islamic scholarship — hadith grading, Quran tafsir, narrator chains.

## Development Commands

### Hadith RAG

```bash
cd hadith_rag
python -m venv .venv && source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# API server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# CLI pipeline test
python -m pipeline.rag_pipeline "ما صحة حديث من غشنا فليس منا"

# Build TF-IDF sparse index (requires data/processed_hadiths.jsonl)
python -m retrieval.build_tfidf_index
```

### Quran RAG

```bash
cd quran_rag
pip install -r requirements.txt

# Requires Qdrant running (use docker-compose)
docker compose up -d qdrant

# API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or with Docker (GPU)
docker compose up --build

# Run ingestion pipeline standalone
python -m app.ingestion.local_pipeline [surah_number] [db_path]
```

### Environment Variables

**Hadith RAG** — `.env` in `hadith_rag/`:
- `JINA_API_KEY` (required) — query-time embeddings via Jina REST API
- `GEMINI_API_KEY` (required) — generation via Google Gemini

**Quran RAG** — `.env` in `quran_rag/` or project root:
- No API keys needed for local Jina model embedding
- Qdrant runs locally (default `localhost:6333`)

## Architecture

### Hadith RAG Pipeline (`hadith_rag/`)

```
Query → QueryPreprocessor (classify: greeting/stats/metadata/narrator/general)
      → Early exit for greetings, out-of-scope, dataset stats
      → HybridRetriever (Dense ChromaDB + TF-IDF sparse + RRF fusion)
      → Canonical group deduplication
      → HadithReranker (BGE cross-encoder, 20→5)
      → HadithGenerator (Gemini, query-type-aware prompts)
      → Citation grounding verification
      → RAGResponse with grade audit
```

- **Config**: `pipeline/config.py` — dataclass `Settings`, loaded from env/`.env`. Contains hadith grade classification (sahih/hasan/daif/mawdu) with Arabic pattern matching and `GradeAudit`.
- **Orchestrator**: `pipeline/rag_pipeline.py` — `HadithRAGPipeline` wires all stages. Entry point for API and CLI.
- **API**: `api/app.py` — FastAPI with Pydantic models. Global `_pipeline` init in lifespan.
- **Embedding**: Query-time only via Jina REST API (`pipeline/embed_query.py`). Bulk embedding done on Colab GPU.
- **Generation**: Gemini (`pipeline/generate.py`), Groq commented out for quick rollback.

### Quran RAG Pipeline (`quran_rag/`)

```
POST /api/v1/ingest → ThemeLoader (SQLite) → QuranApiClient (fetch ayahs + tafsir)
                    → AdaptiveThematicChunker → JinaEmbedder (local model)
                    → QdrantVectorStore (upsert with BM25 fit)

POST /api/v1/search → JinaEmbedder.embed_query → QdrantVectorStore.hybrid_search
                    → Dense cosine + BM25 sparse → RRF fusion
                    → Ayah-overlap deduplication → results
```

- **Config**: `app/core/config.py` — pydantic-settings `BaseSettings` with `get_settings()` LRU cache.
- **Orchestrator**: `app/ingestion/local_pipeline.py` — `QuranicRAGPipeline` handles ingest and search.
- **API**: `app/api/retrieval_router.py` — mounted at `/api/v1`. Pipeline must init via `/ingest` before `/search` works.
- **Vector Store**: `app/services/vector_store.py` — `QdrantVectorStore` with in-memory hybrid search. Custom `ArabicBM25Vectorizer` with morphological stemming and dual tokenization (aggressive + light).
- **Embedding**: Local Jina v3 model via `app/ingestion/embedder.py` (GPU when available, fp16).
- **Data Source**: `ayah-themes-clean.db` SQLite with Quranic themes, loaded by `app/ingestion/theme_loader.py`.

### Key Differences Between Systems

| Aspect | Hadith RAG | Quran RAG |
|--------|-----------|-----------|
| Vector DB | ChromaDB (file-persisted) | Qdrant (Docker service) |
| Embedding | Jina REST API (query-time) | Jina local model (GPU) |
| Sparse retrieval | TF-IDF char n-gram (sklearn) | Custom BM25 with Arabic stemmer |
| LLM generation | Yes (Gemini) | No (retrieval-only) |
| Index build | Colab GPU notebooks | Local via `/ingest` endpoint |
| Config style | dataclass + `os.getenv` | pydantic-settings `BaseSettings` |

### Data Flow Notes

- Hadith ChromaDB index pre-built on Google Colab (GPU), downloaded as artifacts. Local runtime CPU-only.
- Quran RAG ingestion fetches ayah text + tafsir from external Quran API (`api.quranhub.com`), caches locally.
- Both systems use RRF (Reciprocal Rank Fusion) merging dense and sparse results.
- Arabic text normalization (tashkeel stripping, alef/taa normalization) critical throughout both pipelines.

## No Test Suite

No automated tests in repo. `.gitignore` excludes `tests/` directory.