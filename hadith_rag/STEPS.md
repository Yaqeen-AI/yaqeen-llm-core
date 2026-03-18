# 📋 YaqeenAI Hadith RAG — Setup Steps

> **Architecture:** Preprocessing + Embedding on **Google Colab (GPU)** → Download artifacts → **Local runtime (CPU)**

---

## Prerequisites

| Requirement | Details |
|---|---|
| Google Colab | Free tier with T4 GPU runtime |
| Google Drive | `dorar_hadith_full_dataset.json` uploaded to Drive |
| Python 3.11+ | Local machine with `venv` |
| API Keys | Jina AI (embedding at query time) + Groq (LLM generation) |

---

## Phase 0 — Local Environment Setup

```bash
cd hadith_rag
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Create `.env` in `hadith_rag/`:

```env
JINA_API_KEY=jina_xxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
```

---

## Phase 1 — Data Preprocessing (Colab)

> **Notebook:** `notebooks/01_data_prep.ipynb`

**Upload** `dorar_hadith_full_dataset.json` to your Google Drive at:
```
MyDrive/colabnotebooks/yaqeen_hadith_rag/data/
```

**Open** `01_data_prep.ipynb` in Google Colab and run all cells.

**What it does:**
1. Loads the raw JSON dataset from Google Drive
2. Inspects and converts to DataFrame
3. Detects hadith grade (sahih/hasan/daif/mawdu) from the `ruling` field
4. Generates stable `doc_id` for each hadith
5. Normalizes Arabic text (strips tashkeel, normalizes alef/taa)
6. Computes `canonical_group_id` for near-duplicate dedup
7. **Builds metadata-enriched `embedding_text`** — includes narrator, source, and grade alongside the matn so semantic search matches metadata queries too
8. **Computes `dataset_stats.json`** — pre-computed counts (total, by grade, by narrator, by source) for instant stats answers
9. Exports `cleaned_hadiths.parquet` + `dataset_stats.json` to Drive

**Outputs saved to Drive:**
- `cleaned_hadiths.parquet` (~50 MB)
- `dataset_stats.json` (~5 KB)

---

## Phase 2 — Embedding & Index Build (Colab GPU)

> **Notebook:** `notebooks/02_index_build.ipynb`

**Open** `02_index_build.ipynb` in Google Colab **(T4 GPU runtime required)** and run all cells.

**What it does:**
1. Loads `cleaned_hadiths.parquet` from Drive
2. Loads Jina v3 embedding model on GPU (fp16)
3. Batch-embeds all `embedding_text` strings (metadata-enriched) → 1024-dim vectors
4. Creates ChromaDB collection with HNSW index (cosine, ef=200, M=32)
5. Bulk inserts all documents + embeddings + metadata into ChromaDB
6. **Builds TF-IDF sparse index** (char n-grams 3-5, 150K features) on Colab
7. Runs verification queries (dense + filtered + TF-IDF)
8. Zips all artifacts and triggers download

**Outputs zipped for download:**
- `hadith_chroma_db/` — ChromaDB vector store
- `tfidf_index.pkl` — TF-IDF sparse index
- `dataset_stats.json` — pre-computed dataset statistics

---

## Phase 3 — Download Artifacts to Local Machine

After Colab notebook 02 finishes, a zip file `hadith_rag_artifacts.zip` is downloaded.

Extract it into your local project:

```
hadith_rag/
├── chroma_db/
│   └── hadith_chroma_db/    ← from zip
├── data/
│   ├── tfidf_index.pkl      ← from zip
│   └── dataset_stats.json   ← from zip
```

**Verify files exist:**
```bash
ls chroma_db/hadith_chroma_db/chroma.sqlite3
ls data/tfidf_index.pkl
ls data/dataset_stats.json
```

---

## Phase 4 — Run the Local Server

```bash
cd hadith_rag
.venv\Scripts\activate

# Quick test via CLI
python -m pipeline.rag_pipeline

# Start the FastAPI server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open the UI: **http://localhost:8000**

---

## Phase 5 — Verify Everything Works

Test these query types:

| Query | Expected Behavior |
|---|---|
| `السلام عليكم` | Greeting → instant response (no retrieval) |
| `كم عدد الأحاديث` | Dataset stats → instant response from `dataset_stats.json` |
| `كم حديث صحيح` | Grade-specific stats → instant count |
| `أحاديث عن الصبر` | Topic search → hybrid retrieval + rerank + LLM |
| `ما صحة حديث من غشنا فليس منا` | Metadata query → retrieval + metadata-focused LLM |
| `أحاديث رواها أبو هريرة` | Narrator search → retrieval + narrator-focused LLM |
| `how to code in python` | Out-of-scope → polite rejection (no retrieval) |

---

## File Structure Reference

```
hadith_rag/
├── notebooks/
│   ├── 01_data_prep.ipynb        # Colab: preprocessing + stats
│   └── 02_index_build.ipynb      # Colab: embedding + ChromaDB + TF-IDF
├── pipeline/
│   ├── config.py                 # Central configuration
│   ├── rag_pipeline.py           # Pipeline orchestrator
│   ├── embed_query.py            # Jina API query embedding
│   ├── retrieve.py               # ChromaDB retrieval
│   ├── rerank.py                 # BGE reranker
│   ├── generate.py               # Groq LLM generation
│   └── arabic_normalizer.py      # Arabic text normalization
├── retrieval/
│   ├── hybrid_retriever.py       # Dense + Sparse + RRF fusion
│   ├── query_preprocessor.py     # Query classification & routing
│   ├── tfidf_service.py          # TF-IDF search service
│   └── bm25_service.py           # BM25 search service
├── api/
│   ├── app.py                    # FastAPI server
│   └── ui.html                   # Web interface
├── data/
│   ├── dataset_stats.json        # Pre-computed statistics
│   └── tfidf_index.pkl           # TF-IDF sparse index
├── chroma_db/
│   └── hadith_chroma_db/         # ChromaDB vector store
├── requirements.txt
├── STEPS.md                      # ← You are here
└── STRATEGY.md                   # Architecture & design decisions
```
