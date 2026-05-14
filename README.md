<div align="center">

# الموسوعة الفقهية الكويتية — بحث ذكي
### FiqhRAG · Islamic Jurisprudence AI Search

![Python](https://img.shields.io/badge/Python-3.11%2B%20%7C%203.14%20tested-3776AB?style=flat-square&logo=python&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?style=flat-square)
![Jina](https://img.shields.io/badge/Jina_AI-Embeddings_v3-000000?style=flat-square)
![Redis](https://img.shields.io/badge/Redis-Two--Tier_Cache-DC382D?style=flat-square&logo=redis&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Retriever-7B2FBE?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-1C7C54?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini_2.0_Flash-Google_API-4285F4?style=flat-square&logo=google&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=flat-square)

*Ask any Islamic jurisprudence question in Arabic. Get a grounded, cited, madhab-attributed answer from 46 volumes of classical scholarship.*

</div>

---

## Overview

FiqhRAG is a full Arabic end-to-end Retrieval-Augmented Generation system built on the **Kuwaiti Fiqh Encyclopedia** (الموسوعة الفقهية الكويتية) — the most comprehensive modern reference in Islamic jurisprudence, covering all four Sunni madhabs across 46 volumes.

Every answer is:
- **Grounded** in specific passages from the encyclopedia
- **Cited** with standardized academic references (`م.ف.ك — جX، صY`)
- **Madhab-aware** — Hanafi, Maliki, Shafi'i, and Hanbali positions are detected and labeled per passage
- **Topic-filtered** — queries are routed to the relevant Fiqh topic slice before search, reducing the effective corpus from 100k to ~10–25k chunks
- **Citation-validated** — post-generation check flags any invented reference numbers
- **Cached** — two-tier cache (Redis exact + Qdrant semantic) eliminates redundant API calls
- **Query-router ready** — the LangGraph retrieval graph returns `list[NodeWithScore]` documents for downstream routing

---

## Architecture

```
Arabic Query
     │
     ▼
┌─────────────────────────────────────────────────┐
│                Two-Tier Cache                   │
│  Tier 1: Redis SHA-256 hash  →  O(1) exact hit  │
│  Tier 2: Jina v3 cosine ≥ 0.80 → semantic hit   │
└────────────────────┬────────────────────────────┘
                     │ miss
                     ▼
┌─────────────────────────────────────────────────┐
│         LangGraph: extract_filter_node          │
│  detect_mazhabs()    →  mazhab pre-filter       │
│  detect_fiqh_topic() →  topic pre-filter        │
│  (narrows corpus: 100k → ~10–25k chunks)        │
└────────────────────┬────────────────────────────┘
                     │
           ┌─────────┴──────────┐
           ▼                    ▼
   ┌──────────────┐    ┌────────────────┐
   │  Jina v3     │    │   BM25 Okapi   │
   │  Dense       │    │   Feature-hash │
   │  1024-dim    │    │   2048-dim     │
   └──────┬───────┘    └──────┬─────────┘
          │ (parallel)        │
          └─────────┬─────────┘
                    ▼
         ┌─────────────────────┐
         │  Qdrant Prefetch    │
         │  + Metadata Filter  │
         │  (topic + mazhab)   │
         │  (20 per modality)  │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Reciprocal Rank    │
         │  Fusion  (RRF)      │
         └──────────┬──────────┘
                    ▼
         ┌─────────────────────┐
         │  Jina Reranker v2   │
         │  Multilingual       │
         │  → Top 10 results   │
         │  + Mazhab tags      │
         │  + Topic tags       │
         └──────────┬──────────┘
                    ▼
         ┌──────────────────────────┐
         │  LangGraph StateGraph    │
         │  FiqhRAGState            │
         │  { query,                │
         │    mazhab_filter,        │
         │    topic_filter,         │
         │    documents }           │
         │  → list[NodeWithScore]   │  ◄─ consumed by query router
         └──────────┬───────────────┘
                    ▼
         ┌─────────────────────┐
         │  Gemini 2.0 Flash   │
         │  (CustomLLM)        │
         │  Arabic answer      │
         │  + inline [n] refs  │
         │  + citation check   │
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Populate Cache     │
         │  (Tier 1 + Tier 2)  │
         └─────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Vector DB** | [Qdrant](https://qdrant.tech) (local) | Stores named dense vectors (`dense` + `bm25_dense`) + metadata indexes |
| **Dense Embeddings** | [Jina Embeddings v3](https://jina.ai) · 1024-dim · `BaseEmbedding` | Semantic similarity |
| **Keyword Vectors** | Custom BM25 Okapi · feature-hashed · 2048-dim | Probabilistic keyword ranking as dense vector |
| **Hybrid Fusion** | Reciprocal Rank Fusion via Qdrant | Merges both result lists |
| **Reranker** | [Jina Reranker v2](https://jina.ai) multilingual · `BaseNodePostprocessor` | Cross-encoder reranking |
| **Retriever** | [LlamaIndex](https://llamaindex.ai) `BaseRetriever` | Standard `NodeWithScore` document interface |
| **LLM** | [LlamaIndex](https://llamaindex.ai) `CustomLLM` wrapping Gemini | Settings-registered LLM for LlamaIndex compatibility |
| **Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` | Filter extraction + retrieval graph |
| **Cache Tier 1** | [Redis](https://redis.io) · SHA-256 · LRU eviction | Exact-match, microsecond lookup |
| **Cache Tier 2** | Qdrant · Jina v3 · cosine ≥ 0.80 | Semantic-match, paraphrase hits |
| **Generation** | Gemini 2.0 Flash via [Google AI](https://ai.google.dev) | Arabic answer synthesis via API |
| **REST API** | [FastAPI](https://fastapi.tiangolo.com) | Retrieval-only endpoint for query router integration |
| **Web UI** | [Gradio](https://gradio.app) | Arabic RTL chat interface |
| **Arabic NLP** | Custom normalization + mazhab + topic detection | NFKC, diacritics, scholar patterns, Fiqh topic routing |

---

## Project Structure

```
FiqhRAG/
│
├── core/                         Library — imported by all entry points
│   ├── arabic_utils.py           Normalization · mazhab detection · Fiqh topic detection · citation format
│   ├── bm25.py                   Custom BM25 Okapi + feature-hashed dense vectors (GPU optional)
│   ├── cache.py                  Two-tier cache (Redis exact + Qdrant semantic)
│   ├── config.py                 All settings — edit this file
│   ├── embeddings.py             JinaEmbedding (LlamaIndex BaseEmbedding) — shared across all callers
│   ├── generator.py              GeminiLLM (LlamaIndex CustomLLM) + Arabic answer generation + citation validator
│   ├── graph.py                  LangGraph StateGraph — filter extraction + retrieval pipeline
│   ├── http.py                   Shared requests.Session — TCP connection pooling for Jina API calls
│   ├── llamaindex_retriever.py   LlamaIndex BaseRetriever adapter + Result↔NodeWithScore converters
│   ├── qdrant_singleton.py       Shared Qdrant client (prevents file-lock conflicts)
│   ├── reranker.py               JinaReranker (LlamaIndex BaseNodePostprocessor)
│   ├── retriever.py              Hybrid search (BM25 + Qdrant RRF) + mazhab/topic filter + fallback logic
│   └── schema.py                 TypedDicts: QdrantPayload · NodeMetadata (single source of truth for field names)
│
├── scripts/                      One-time / admin tools
│   ├── ingest.py                 Build the vector index (run once, ~90 min on free Jina tier)
│   ├── enrich_payloads.py        Backfill mazhab + fiqh_topic tags on existing index (run once, <1 min)
│   └── smoke_test.py             RAG evaluation — hit rate, rerank scores, latency across 20 questions
│
├── data/                         Generated artifacts  (gitignored)
│   ├── bm25_corpus.pkl           Fitted BM25 model (16,971 documents)
│   └── embed_checkpoint.pkl      Ingestion resume checkpoint
│
├── fiqh_data/                    Source data — 45 JSONL volumes (vol 2–46)
├── qdrant_storage/               Qdrant RAG index        (gitignored, auto-created)
├── qdrant_cache/                 Qdrant semantic cache   (gitignored, auto-created)
│
├── app.py                        ▶  Web UI      — python app.py          (port 7860)
├── main.py                       ▶  CLI          — python main.py
├── api.py                        ▶  REST API     — uvicorn api:app        (port 8000)
│
├── .env                          API keys  (not committed)
└── requirements.txt              Python dependencies
```

---

## Setup

### Prerequisites

| Requirement | Details |
|---|---|
| **Python** | 3.11 or higher (3.14 fully supported and tested) |
| **[Jina AI API key](https://jina.ai)** | Free tier — used for embeddings + reranker |
| **[Google AI API key](https://ai.google.dev)** | Used for Gemini 2.0 Flash generation |
| **[Redis](https://redis.io/download)** | Optional — Tier 1 cache degrades gracefully if absent |
| **NVIDIA GPU + CUDA** | Optional — enables GPU-accelerated BM25 encoding (see GPU section below) |

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env
```

`.env` contents:
```
JINA_API_KEY=your_jina_key_here
GOOGLE_API_KEY=your_google_key_here
GEMINI_MODEL=gemini-2.0-flash
```

### 3 — Build the index (run once)

Loads all 46 volumes, builds BM25 corpus, embeds with Jina v3, and indexes into Qdrant.

```bash
python -m scripts.ingest
```

> **Note:** Jina free tier is rate-limited to ~6 requests/min. With 16,971 chunks at batch size 32, this takes approximately 90 minutes. The pipeline is **resumable** — if interrupted, re-run the same command and it picks up from the checkpoint automatically.

### 4 — Enrich existing index (run once)

Backfills `mazhabs` and `fiqh_topic` metadata on every Qdrant point. No re-embedding — pure local operation, completes in under a minute. **Required** for topic-based corpus filtering to take effect.

```bash
python -m scripts.enrich_payloads
```

Sample output:
```
Enriching 16,971 points in 'fiqh_rag' with mazhab + fiqh_topic tags...
Done — 16,971 points updated.

Mazhab mention counts:
  حنبلي       3,241  (19.1%)
  شافعي       2,987  (17.6%)
  حنفي        2,814  (16.6%)
  مالكي       2,603  (15.3%)
  جمهور       1,120   (6.6%)
```

### 5 — Run

**Web UI** (recommended):
```bash
python app.py
# Opens at http://localhost:7860
```

**CLI:**
```bash
python main.py
```

**REST API:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## Fiqh Topic Filtering

Each chunk is tagged at ingestion time with its dominant Fiqh topic using keyword pattern matching on the Arabic text. At query time, the same detector runs on the user's question and applies a Qdrant pre-filter before the vector search — so Qdrant scans only the relevant topic slice.

**Topic categories and typical corpus share:**

| Topic | Arabic | ~Corpus share |
|---|---|---|
| Sales & contracts | بيوع | ~13% |
| Partnerships & debts | الشركات والديون | ~12% |
| Prayer | صلاة | ~21% |
| Pilgrimage | حج | ~17% |
| Purification | طهارة | ~9% |
| Fasting | صيام | ~4% |
| Zakat | زكاة | ~4% |
| Marriage | نكاح | ~3% |
| Divorce | طلاق | ~3% |
| Inheritance | ميراث | ~2% |
| Crimes & penalties | جنايات | ~5% |

For a 100k-chunk corpus, a prayer query searches ~21k chunks instead of 100k — a 5× reduction with zero accuracy loss.

**Tie handling:** when two topics score equally (e.g. "السواك في الصلاة"), both are passed to Qdrant as a union filter — still a meaningful corpus reduction. If the collection has no `fiqh_topic` field yet (before `enrich_payloads.py` is run), the retriever falls back to full-corpus search automatically and logs a warning.

**Threshold:** configured via `_MIN_TOPIC_SCORE` in `core/arabic_utils.py` (default: 1). Increase to 2 to require stronger signal before filtering is applied.

---

## Query Router Integration

The LangGraph graph is designed to be embedded in a larger query router pipeline. The retrieval node returns standard LlamaIndex `NodeWithScore` documents — not a generated answer — so the router decides what to do next.

```python
from core.graph import fiqh_graph
from core.llamaindex_retriever import nodes_to_results

# Invoke the retrieval graph
result = fiqh_graph.invoke({"query": "ما حكم الوضوء بالماء المستعمل؟"})

# List of NodeWithScore — each node carries full metadata
docs = result["documents"]   # list[NodeWithScore]

# Metadata available on each doc:
# doc.node.text          — Arabic passage text
# doc.score              — rerank score (primary relevance signal)
# doc.node.metadata keys:
#   volume_id            — e.g. "Volume 40"
#   book_page            — e.g. "Page 359"
#   chunk_page           — e.g. "1 of 45"
#   source_url           — original URL
#   mazhabs              — detected madhabs e.g. ["حنفي", "شافعي"]
#   fiqh_topic           — dominant topic e.g. "طهارة"
#   qdrant_score         — pre-rerank fusion score
#   rerank_score         — post-rerank score (same as doc.score)
#   short_ref            — formatted citation string "م.ف.ك — ج40، ص359"
#   rank                 — 0-based rank after reranking
```

To pass documents to the generator:
```python
from core.llamaindex_retriever import nodes_to_results
from core.generator import generate_answer

results = nodes_to_results(docs)
answer = generate_answer(query, results)
```

**REST API** (retrieval only, no generation):
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "ما حكم الوضوء بالماء المستعمل؟"}'
```

---

## GPU Acceleration (Optional)

BM25 dense vector encoding supports GPU acceleration via PyTorch. The system auto-detects GPU availability — no config change needed.

**Status:** `BM25_USE_GPU` in `core/config.py` is set automatically at startup:
- `True` when torch is installed and CUDA is available → GPU path
- `False` otherwise → NumPy CPU fallback (fully functional, slightly slower during ingestion)

### Installing torch on Python 3.14

PyTorch stable releases do not yet publish wheels for Python 3.14. Use the nightly build instead:

```bash
pip uninstall torch -y
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126
```

---

## Configuration

All settings live in `core/config.py`:

| Setting | Default | Description |
|---|---|---|
| `JINA_EMBED_MODEL` | `jina-embeddings-v3` | Embedding model |
| `JINA_RERANK_MODEL` | `jina-reranker-v2-base-multilingual` | Reranker model |
| `EMBED_DIM` | `1024` | Embedding dimensions |
| `GOOGLE_API_KEY` | env var | Google AI API key for Gemini |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name (override via env var) |
| `TOP_K_FETCH` | `20` | Candidates fetched before reranking |
| `TOP_K_FINAL` | `10` | Results returned after reranking |
| `BM25_K1` | `1.5` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 length normalization |
| `BM25_DENSE_DIM` | `2048` | Feature-hash vector size |
| `BM25_USE_GPU` | auto-detected | GPU BM25 encoding (True when torch + CUDA present) |
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_MAX_MEMORY` | `256mb` | Redis memory cap (LRU eviction) |
| `SEMANTIC_THRESHOLD` | `0.80` | Min cosine score for Tier 2 cache hit |

---

## Two-Tier Cache

Repeated or paraphrased questions are served from cache, skipping retrieval and LLM generation entirely.

```
Query → normalize → SHA-256 hash → Redis GET
                                        │
                              hit ──────┘  ← microseconds, free
                                        │
                              miss ─────▼
                                   Jina v3 embed → Qdrant cosine search (≥ 0.80)
                                        │
                              hit ──────┘  ← ~200ms, one Jina API call
                                   (promotes answer to Redis for next time)
                                        │
                              miss ─────▼
                                   Full pipeline (retrieve → rerank → generate)
                                   → store in both tiers
```

Both tiers degrade gracefully — if Redis is not running, Tier 1 is skipped and the system falls back to Tier 2 only.

---

## Arabic Normalization

Applied before BM25 fitting and query encoding. Jina embeddings receive the original text — it handles Arabic natively.

| Transform | Details |
|---|---|
| **NFKC** | Expands `ﷺ` → `صلى الله عليه وسلم`, decomposes Arabic ligatures and presentation forms |
| Alef variants `أ إ آ ٱ` | → `ا` |
| Ya variant `ى` | → `ي` |
| Waw/Ya hamza `ؤ ئ` | → `و` / `ي` |
| Tatweel `ـ` | removed |
| Tashkeel (diacritics) | removed |
| Eastern numerals `٠–٩` | → `0–9` |

---

## Madhab Detection & Citation

Each retrieved passage is automatically tagged with the Islamic schools of thought it references, using compiled regex patterns for school names and prominent scholar names.

**Detected schools:** حنفي · مالكي · شافعي · حنبلي · جمهور

**Citation format:** `م.ف.ك — جX، صY` (Mausuah Fiqhiyyah Kuwaitiyah, Volume X, Page Y)

**Citation validation:** After generation, every `[n]` inline reference is checked against the retrieved chunk list. Out-of-range references trigger an automatic warning in the response.

---

## Data

**Source:** Kuwaiti Fiqh Encyclopedia (الموسوعة الفقهية الكويتية)
- 46 volumes · 16,971 indexed passages
- Covers all four Sunni madhabs
- Classical and contemporary rulings

Each JSONL record:
```json
{
  "volume_id":   "Volume 6",
  "book_page":   "Page 71",
  "chunk_page":  "71 of 381",
  "chunk_text":  "...(Arabic text)...",
  "source_url":  "https://chat.aifiqh.com/..."
}
```

After `enrich_payloads.py`, each Qdrant point additionally carries:
```json
{
  "mazhabs":    ["حنفي", "شافعي"],
  "fiqh_topic": "طهارة"
}
```

The full payload schema is defined in `core/schema.py` (`QdrantPayload` TypedDict).

---

## Inspiration

Inspired by [AIFiqh](https://aifiqh.com) — a specialized Islamic jurisprudence AI platform. This project extends their RAG concept with Qdrant hybrid search, Jina v3 embeddings, custom BM25 Okapi with feature-hashed dense vectors, a two-tier response cache, madhab-aware context, Fiqh topic filtering, LlamaIndex retriever/LLM abstractions, LangGraph orchestration, and Gemini 2.0 Flash generation.
