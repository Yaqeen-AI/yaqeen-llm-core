# YaqeenAI — Hadith RAG System

> Retrieval-Augmented Generation system for 155,000+ authenticated hadiths from the Dorar Al-Seniyyah dataset.  
> Hybrid dense + sparse retrieval · BGE cross-encoder reranking · Groq (Llama 3.3 70B) generation · Citation-grounded Arabic answers.

---

## Architecture

```
User Query
    │
    ▼
Query Preprocessor  (normalize · classify · expand)
    │
    ├──────────────────────────┐
    ▼                          ▼
Dense Retrieval            Sparse Retrieval
Jina v3 + ChromaDB         TF-IDF char n-gram
(top-30)                   (top-30)
    │                          │
    └──────────┬───────────────┘
               ▼
       RRF Fusion (k=60)
               │
               ▼
     Canonical Deduplication
               │
               ▼
    BGE Reranker (20 → top-5)
               │
               ▼
  Groq / Llama 3.3 70B Generation
               │
               ▼
   Citation Grounding Verification
               │
               ▼
     Structured Arabic Response
```

### Models

| Component | Model |
|-----------|-------|
| Embedding (query-time) | `jinaai/jina-embeddings-v3` · 1024-dim · via REST API |
| Reranker | `BAAI/bge-reranker-v2-m3` · CPU cross-encoder |
| Generation | `llama-3.3-70b-versatile` via Groq API (free) |

---

## Project Structure

```
hadith_rag/
├── api/
│   ├── app.py                  ← FastAPI server (query · search · health · stats · UI)
│   └── ui.html                 ← Single-page web UI
├── ingestion/
│   └── pipeline.py             ← Raw JSON → cleaned JSONL
├── retrieval/
│   ├── tfidf_service.py        ← TF-IDF char n-gram sparse index
│   ├── build_tfidf_index.py    ← Index builder
│   ├── hybrid_retriever.py     ← Dense + Sparse + RRF fusion
│   └── query_preprocessor.py  ← Normalize · classify · expand
├── pipeline/
│   ├── config.py               ← Centralised settings (.env-backed)
│   ├── arabic_normalizer.py    ← Strip tashkeel / tatweel / prefix
│   ├── embed_query.py          ← Jina REST API embedding
│   ├── retrieve.py             ← ChromaDB dense retrieval
│   ├── rerank.py               ← BGE cross-encoder reranking
│   ├── generate.py             ← Groq generation + citation grounding
│   └── rag_pipeline.py         ← Full pipeline orchestrator (CLI entry point)
├── notebooks/
│   ├── 01_data_prep.ipynb      ← [Colab] Clean + normalise dataset
│   └── 02_index_build.ipynb    ← [Colab] Embed 155K hadiths + build ChromaDB
├── data/
│   ├── dorar_hadith_full_dataset.json   ← Raw dataset (155,502 records)
│   ├── processed_hadiths.jsonl          ← Output of ingestion pipeline
│   ├── ingestion_stats.json             ← Ingestion run statistics
│   └── tfidf_index.pkl                  ← Built locally (~400 MB)
├── chroma_db/hadith_chroma_db/          ← Downloaded from Colab (~700 MB)
├── requirements.txt
├── .env                        ← API keys (not committed)
├── STEPS.md
├── STRATEGY.md
└── DESIGN_DECISIONS.md
```

---

## Quick Start

### 1 · Install dependencies

```powershell
cd hadith_rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2 · Set API keys

Create `.env` in `hadith_rag/`:

```env
JINA_API_KEY=jina_xxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxx
```

- **Jina AI** (free) → https://jina.ai/
- **Groq** (free, no credit card) → https://console.groq.com/

### 3 · Run ingestion

```powershell
python -m ingestion.pipeline
```

### 4 · Build ChromaDB on Colab

The 155K-hadith embedding requires a GPU. Use Google Colab (free T4):

1. Upload `data/dorar_hadith_full_dataset.json` to `MyDrive/hadith_data/`
2. Run `notebooks/01_data_prep.ipynb`
3. Run `notebooks/02_index_build.ipynb` — saves `hadith_chroma_db.zip` to Drive
4. Download and extract to `chroma_db/hadith_chroma_db/`

### 5 · Build TF-IDF sparse index

```powershell
python -m retrieval.build_tfidf_index
```

### 6 · Start the API + Web UI

```powershell
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

| URL | Description |
|-----|-------------|
| http://localhost:8000 | Web UI |
| http://localhost:8000/docs | Swagger API docs |
| http://localhost:8000/health | Health check |

### 7 · Run CLI pipeline

```powershell
python -m pipeline.rag_pipeline "ما صحة حديث من غشنا فليس منا"
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/query` | Full RAG (retrieve → rerank → generate) |
| `POST` | `/search` | Retrieval-only, no generation |
| `GET` | `/health` | Component health check |
| `GET` | `/stats` | Index statistics |

### Query with grade filter

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "أحاديث الصيام", "grade_filter": ["sahih", "hasan"]}'
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JINA_API_KEY` | *(required)* | Jina AI API key |
| `GROQ_API_KEY` | *(required)* | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `CHROMA_PERSIST_DIR` | `chroma_db/hadith_chroma_db` | ChromaDB path |
| `CHROMA_COLLECTION_NAME` | `hadith_collection` | Collection name |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model |
| `RETRIEVAL_TOP_K` | `20` | Fused candidates before reranking |
| `RERANK_TOP_K` | `5` | Final results after reranking |
| `DENSE_TOP_K` | `30` | Dense retrieval depth |
| `SPARSE_TOP_K` | `30` | TF-IDF retrieval depth |
| `RRF_K` | `60` | RRF fusion constant |
| `TFIDF_INDEX_PATH` | `data/tfidf_index.pkl` | TF-IDF index path |
| `TFIDF_MAX_FEATURES` | `300000` | TF-IDF vocabulary cap |
| `EMBEDDING_CACHE_SIZE` | `1000` | LRU cache size |

---

## Dataset

- **Source**: [Dorar Al-Seniyyah](https://dorar.net/)
- **Size**: 155,502 records · single JSON file
- **Grades**: Detected from `ruling` field — `sahih` · `hasan` · `daif` · `mawdu` · `unknown`

---

## Documentation

| File | Contents |
|------|----------|
| [STEPS.md](STEPS.md) | Step-by-step execution guide |
| [STRATEGY.md](STRATEGY.md) | Architecture and design decisions |
| [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) | Technology choice justifications |
