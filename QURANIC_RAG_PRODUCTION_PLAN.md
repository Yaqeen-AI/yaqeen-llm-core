# Quranic RAG: Notebook → Production Migration Plan

This document describes how to migrate the `quranic_rag_colab.ipynb` notebook into a production-grade, locally-runnable codebase — using the `hadith_rag/` project as the architectural reference.

---

## Target Architecture

```
quran_rag/
├── main.py                          # Uvicorn entry point
├── requirements.txt                 # Pinned dependencies
├── Dockerfile                       # Container build
├── docker-compose.yml               # Qdrant + API orchestration
├── .env.example                     # Required env vars template
│
├── api/
│   ├── app.py                       # FastAPI server + lifespan
│   └── models.py                    # Pydantic request/response schemas
│
├── pipeline/
│   ├── config.py                    # Settings (env-backed, single source of truth)
│   ├── arabic_normalizer.py         # TextNormalizer (from notebook cell 12)
│   ├── rag_pipeline.py              # Orchestrator: preprocess → retrieve → rerank → generate
│   ├── rerank.py                    # Cross-encoder reranking (BGE reranker)
│   └── generate.py                  # LLM answer generation + citation grounding
│
├── retrieval/
│   ├── query_preprocessor.py        # Query classification + normalization
│   ├── query_expander.py            # LLM-driven morphological expansion
│   ├── dense_retriever.py           # ChromaDB/Qdrant dense vector search
│   ├── tfidf_service.py             # TF-IDF char n-gram sparse retrieval
│   ├── hybrid_retriever.py          # RRF fusion + multi-query orchestration
│   └── build_tfidf_index.py         # One-time script to build/rebuild sparse index
│
├── ingestion/
│   ├── theme_loader.py              # SQLite theme loading
│   ├── quran_fetcher.py             # Async API fetcher + cache
│   ├── chunker.py                   # Adaptive thematic chunker
│   ├── embedder.py                  # Jina embedding (local model or API)
│   └── ingest_pipeline.py           # Full ingestion orchestrator
│
├── data/
│   ├── ayah-themes-clean.db         # Source database
│   ├── tfidf_index.pkl              # Serialized sparse index
│   └── dataset_stats.json           # Pre-computed corpus statistics
│
└── chroma_db/                       # Persistent ChromaDB storage
    └── quranic_themes/
```

---

## Migration Steps

### Step 1: Extract Configuration (`pipeline/config.py`)

**What changes from notebook:**

- Replace frozen dataclasses with environment-backed `Settings` class
- Single source of truth — no more hardcoded values in class bodies overriding config
- All paths relative to project root, not `/content/`

```python
import os
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class Settings:
    # Paths
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    CHROMA_PERSIST_DIR: str = "chroma_db/quranic_themes"
    CHROMA_COLLECTION_NAME: str = "quranic_themes"
    DATABASE_PATH: str = "data/ayah-themes-clean.db"
    TFIDF_INDEX_PATH: str = "data/tfidf_index.pkl"

    # Embedding — choose local model OR API
    EMBEDDING_MODE: str = os.getenv("EMBEDDING_MODE", "local")
    JINA_EMBEDDING_MODEL: str = "jina-embeddings-v3"
    JINA_EMBEDDING_DIM: int = 1024       # Full dimensions for production
    LOCAL_MODEL_NAME: str = "jinaai/jina-embeddings-v3"
    LOCAL_EMBEDDING_DIM: int = 512       # Matryoshka for local GPU

    # Reranker
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANK_TOP_K: int = 5

    # Retrieval
    DENSE_TOP_K: int = 30
    SPARSE_TOP_K: int = 30
    RETRIEVAL_TOP_K: int = 20    # After RRF, before rerank
    RRF_K: int = 60
    RRF_DENSE_WEIGHT: float = 0.45
    RRF_SPARSE_WEIGHT: float = 0.55

    # TF-IDF
    TFIDF_MAX_FEATURES: int = 300_000
    TFIDF_NGRAM_RANGE: tuple = (3, 5)

    # Generation
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemma-3-27b-it"
    GENERATION_TEMPERATURE: float = 0.3

    # QuranHub API
    QURAN_API_BASE_URL: str = "https://api.quranhub.com/v1"
    QURAN_API_EDITIONS: tuple = ("ar.jalalayn", "ar.muyassar", "ar.sabouni")
    QURAN_API_CONCURRENT: int = 5
    QURAN_API_TIMEOUT: int = 30

    # Chunking
    CHUNK_ATOMIC_MAX: int = 3
    CHUNK_LIGHT_MAX: int = 8
    CHUNK_MEDIUM_MAX: int = 15

    # Caching
    EMBEDDING_CACHE_SIZE: int = 1000

settings = Settings()
```

---

### Step 2: Replace Sparse Retrieval (`retrieval/tfidf_service.py`)

**Why:** The notebook's custom BM25 with aggressive stemmer has known bugs (over-stripping Arabic roots). TF-IDF with char n-grams is proven in hadith_rag and avoids stemming entirely.

**Port from hadith_rag pattern:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

class TfidfService:
    def __init__(self, settings):
        self.settings = settings
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=settings.TFIDF_NGRAM_RANGE,
            max_features=settings.TFIDF_MAX_FEATURES,
            min_df=3,
            max_df=0.85,
            sublinear_tf=True,
            norm='l2',
        )
        self.doc_matrix = None
        self.doc_ids = []

    def build_index(self, documents: list[str], doc_ids: list[str]):
        """Build TF-IDF matrix from corpus documents."""
        self.doc_ids = doc_ids
        self.doc_matrix = self.vectorizer.fit_transform(documents)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'doc_matrix': self.doc_matrix,
                'doc_ids': self.doc_ids,
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.doc_matrix = data['doc_matrix']
            self.doc_ids = data['doc_ids']

    def search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        query_vec = self.vectorizer.transform([query])
        scores = (self.doc_matrix @ query_vec.T).toarray().flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]
```

**Advantages over notebook BM25:**

- No stemmer bugs — char n-grams capture roots automatically
- Vectorized NumPy search — orders of magnitude faster than dict iteration
- Persistent index — build once, load instantly on restart
- Proven on 155K hadith documents in production

---

### Step 3: Add Query Preprocessing (`retrieval/query_preprocessor.py`)

**Port from hadith_rag, adapted for Quranic context:**

```python
class QueryType(Enum):
    VERSE_LOOKUP = "verse_lookup"       # "بسم الله الرحمن الرحيم"
    TOPIC = "topic"                     # "ما هو التوحيد"
    TAFSIR_REQUEST = "tafsir_request"   # "تفسير سورة البقرة آية 255"
    THEME_SEARCH = "theme_search"       # "قصص الأنبياء في القرآن"
    SURAH_INFO = "surah_info"           # "ما هي سورة الكهف"
    GREETING = "greeting"
    OUT_OF_SCOPE = "out_of_scope"
    GENERAL = "general"
```

**Stages:**

1. Language detection + greeting handling
2. Out-of-scope detection (programming, entertainment, etc.)
3. Arabic normalization (tashkil, alef variants)
4. Query type classification (regex + keyword patterns)
5. Surah/ayah reference extraction ("البقرة 255" → filter surah=2, ayah=255)
6. Short query expansion (single-word topics → enriched terms)
7. LLM-driven query expansion (morphological variants + reformulations)

---

### Step 4: Replace In-Memory Store with ChromaDB (`retrieval/dense_retriever.py`)

**Why:** ChromaDB provides persistent HNSW indexing, metadata filtering, and handles the full Quran corpus efficiently.

```python
import chromadb

class DenseRetriever:
    def __init__(self, settings):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 32,
                "hnsw:search_ef": 150,
            }
        )

    def search(self, query_embedding, top_k=30, where_filter=None):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,  # e.g., {"surah_number": 2}
        )
        return results
```

**Key differences from notebook:**

- Persistent storage — survives restarts
- HNSW index — O(log n) search vs. O(n) brute force
- Native metadata filtering — no Python-side filtering loop
- Scales to full Quran (6,236 ayahs, ~5,000+ chunks)

---

### Step 5: Add Hybrid Retriever (`retrieval/hybrid_retriever.py`)

**Orchestrates dense + sparse + multi-query + RRF fusion:**

```python
class HybridRetriever:
    def __init__(self, dense_retriever, tfidf_service, embedder, settings):
        self.dense = dense_retriever
        self.sparse = tfidf_service
        self.embedder = embedder
        self.settings = settings

    def search(self, preprocessed_query) -> list[RetrievalResult]:
        # 1. Dense retrieval
        query_vec = self.embedder.embed_query(preprocessed_query.dense_query)
        dense_results = self.dense.search(query_vec, top_k=self.settings.DENSE_TOP_K)

        # 2. Multi-query sparse retrieval
        all_sparse = []
        for variant in preprocessed_query.multi_queries:
            sparse_results = self.sparse.search(variant, top_k=self.settings.SPARSE_TOP_K)
            all_sparse.append(sparse_results)

        # 3. Fuse sparse variants via RRF
        fused_sparse = self._rrf_fuse_sparse(all_sparse)

        # 4. Fuse dense + sparse via RRF
        combined = self._rrf_fuse(dense_results, fused_sparse)

        return combined[:self.settings.RETRIEVAL_TOP_K]
```

---

### Step 6: Add Embedding Abstraction (`ingestion/embedder.py`)

**Dual mode: local GPU (for batch ingestion) or API (for query-time):**

```python
class EmbedderFactory:
    @staticmethod
    def create(settings) -> Embedder:
        if settings.EMBEDDING_MODE == "api":
            return JinaAPIEmbedder(
                api_key=settings.JINA_API_KEY,
                model=settings.JINA_EMBEDDING_MODEL,
                dimensions=settings.JINA_EMBEDDING_DIM,
            )
        else:
            return JinaLocalEmbedder(
                model_name=settings.LOCAL_MODEL_NAME,
                dimensions=settings.LOCAL_EMBEDDING_DIM,
            )
```

**Benefits:**

- Local development: use API embeddings (no GPU needed, ~100ms/query)
- Batch ingestion: use local model on GPU (faster for thousands of chunks)
- Same interface, same vectors, interchangeable

---

### Step 7: Build FastAPI Server (`api/app.py`)

**Port from hadith_rag pattern:**

```
POST /query       # Full RAG: preprocess → retrieve → rerank → generate
POST /search      # Retrieval-only (no generation)
GET  /health      # Component status checks
GET  /stats       # Index statistics
GET  /            # Web UI
```

**Response includes timing breakdown:**

```json
{
  "answer": "...",
  "citations": [...],
  "hadiths": [...],
  "timing": {
    "preprocess_ms": 5,
    "dense_ms": 45,
    "sparse_ms": 30,
    "fusion_ms": 1,
    "rerank_ms": 2100,
    "generate_ms": 3500,
    "total_ms": 5681
  }
}
```

---

### Step 8: Add Generation Layer (`pipeline/generate.py`)

**Query-type-aware system prompts:**

| Query Type       | System Prompt Focus                                                         |
| ---------------- | --------------------------------------------------------------------------- |
| `topic`          | Cite relevant ayahs with full tafsir. Order by relevance.                   |
| `verse_lookup`   | Return exact verse with all available tafsirs.                              |
| `tafsir_request` | Deep tafsir comparison across editions (Jalalayn vs. Muyassar vs. Sabouni). |
| `theme_search`   | List all matching themes with ayah ranges and keywords.                     |
| `general`        | Standard citation rules + tafsir respect.                                   |

**Citation grounding:** Post-generation verification that all cited surah:ayah references exist in retrieved context.

---

### Step 9: Ingestion Pipeline (`ingestion/ingest_pipeline.py`)

**One-time batch job, not part of the serving path:**

```bash
# Full ingestion (all surahs)
python -m ingestion.ingest_pipeline

# Single surah
python -m ingestion.ingest_pipeline --surah 2

# Rebuild TF-IDF index only (after ChromaDB is populated)
python -m retrieval.build_tfidf_index
```

**Steps:**

1. Load themes from SQLite
2. Fetch ayahs + tafsir from QuranHub API (with file cache)
3. Chunk themes adaptively
4. Embed chunks (local GPU or API)
5. Upsert into ChromaDB
6. Build TF-IDF index from ChromaDB documents
7. Save TF-IDF index to disk

---

### Step 10: Docker Setup

**`docker-compose.yml`:**

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./chroma_db:/app/chroma_db
    depends_on:
      - qdrant # Optional: if migrating to Qdrant server later

  # Optional: Qdrant server for large-scale deployment
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
```

**`Dockerfile`:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Local Development Setup

### Prerequisites

```bash
# Python 3.11+
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables (`.env`)

```bash
# Required
JINA_API_KEY=jina_xxxxx          # For API-mode embeddings
GEMINI_API_KEY=xxxxx             # For generation

# Optional
EMBEDDING_MODE=api               # "api" (no GPU) or "local" (GPU required)
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### First Run

```bash
# 1. Place database
cp ayah-themes-clean.db data/

# 2. Run ingestion (one-time, ~30 min for full Quran)
python -m ingestion.ingest_pipeline

# 3. Start server
python main.py
# → API at http://localhost:8000
# → Web UI at http://localhost:8000/
```

### Development Workflow

```bash
# Run with auto-reload
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# Test a query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "ما هو التوحيد؟"}'

# Search only (no generation)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "الصبر على البلاء", "top_k": 5}'
```

---

## Migration Checklist

| #   | Task                                        | Source                     | Target                             | Status |
| --- | ------------------------------------------- | -------------------------- | ---------------------------------- | ------ |
| 1   | Extract config to env-backed Settings       | Notebook cells 3, 8        | `pipeline/config.py`               |        |
| 2   | Extract TextNormalizer                      | Notebook cell 12           | `pipeline/arabic_normalizer.py`    |        |
| 3   | Extract data models                         | Notebook cell 10           | `api/models.py`                    |        |
| 4   | Extract QuranFetcher + CachedQuranFetcher   | Notebook cell 14           | `ingestion/quran_fetcher.py`       |        |
| 5   | Extract ThemeLoader                         | Notebook cell 22           | `ingestion/theme_loader.py`        |        |
| 6   | Extract AdaptiveThematicChunker             | Notebook cell 16           | `ingestion/chunker.py`             |        |
| 7   | Extract JinaEmbedder + add API mode         | Notebook cell 18           | `ingestion/embedder.py`            |        |
| 8   | **Replace** BM25 with TF-IDF char n-grams   | Notebook cell 20           | `retrieval/tfidf_service.py`       |        |
| 9   | **Replace** SimpleVectorStore with ChromaDB | Notebook cell 20           | `retrieval/dense_retriever.py`     |        |
| 10  | **New**: Query preprocessor                 | —                          | `retrieval/query_preprocessor.py`  |        |
| 11  | **New**: Query expander                     | —                          | `retrieval/query_expander.py`      |        |
| 12  | **New**: Hybrid retriever                   | Notebook cell 20 (partial) | `retrieval/hybrid_retriever.py`    |        |
| 13  | **New**: Reranker (upgrade to BGE)          | Notebook cell 20 (partial) | `pipeline/rerank.py`               |        |
| 14  | **New**: Generation layer                   | —                          | `pipeline/generate.py`             |        |
| 15  | **New**: FastAPI server                     | —                          | `api/app.py`                       |        |
| 16  | **New**: Ingestion pipeline script          | Notebook cells 22-26       | `ingestion/ingest_pipeline.py`     |        |
| 17  | **New**: Docker setup                       | —                          | `Dockerfile`, `docker-compose.yml` |        |
| 18  | Port cross-encoder reranking                | Notebook cell 20           | `pipeline/rerank.py`               |        |
| 19  | Build TF-IDF index script                   | —                          | `retrieval/build_tfidf_index.py`   |        |
| 20  | Write `.env.example`                        | —                          | `.env.example`                     |        |

---

## Key Differences: Notebook vs. Production

| Aspect              | Notebook (Current)           | Production (Target)             |
| ------------------- | ---------------------------- | ------------------------------- |
| Embedding           | Local Jina model on T4 GPU   | Jina API (no GPU) or local      |
| Vector store        | In-memory dict               | ChromaDB with HNSW              |
| Sparse retrieval    | Custom BM25 + stemmer        | TF-IDF char n-grams (sklearn)   |
| Query preprocessing | None                         | Classification + expansion      |
| Generation          | None (copy-paste to ChatGPT) | Integrated LLM with prompting   |
| Persistence         | Ephemeral (lost on restart)  | Disk-backed (ChromaDB + pickle) |
| Dense search        | O(n) brute force             | O(log n) HNSW                   |
| Reranker            | mmarco-mMiniLM               | BGE-reranker-v2-m3              |
| Deployment          | Google Colab                 | Docker / bare metal             |
| Config              | Frozen dataclasses           | Environment variables           |
| API                 | None                         | FastAPI with health/stats       |
