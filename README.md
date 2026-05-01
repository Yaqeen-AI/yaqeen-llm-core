<div align="center">

# الموسوعة الفقهية الكويتية — بحث ذكي
### FiqhRAG · Islamic Jurisprudence AI Search

![Python](https://img.shields.io/badge/Python-3.11%2B%20%7C%203.14%20tested-3776AB?style=flat-square&logo=python&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?style=flat-square)
![Jina](https://img.shields.io/badge/Jina_AI-Embeddings_v3-000000?style=flat-square)
![Redis](https://img.shields.io/badge/Redis-Two--Tier_Cache-DC382D?style=flat-square&logo=redis&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00?style=flat-square)
![LM Studio](https://img.shields.io/badge/LM_Studio-Gemma_4-7C3AED?style=flat-square)

*Ask any Islamic jurisprudence question in Arabic. Get a grounded, cited, madhab-attributed answer from 46 volumes of classical scholarship — powered entirely by local AI.*

</div>

---

## Overview

FiqhRAG is a full Arabic end-to-end Retrieval-Augmented Generation system built on the **Kuwaiti Fiqh Encyclopedia** (الموسوعة الفقهية الكويتية) — the most comprehensive modern reference in Islamic jurisprudence, covering all four Sunni madhabs across 46 volumes.

Every answer is:
- **Grounded** in specific passages from the encyclopedia
- **Cited** with standardized academic references (`م.ف.ك — جX، صY`)
- **Madhab-aware** — Hanafi, Maliki, Shafi'i, and Hanbali positions are detected and labeled per passage
- **Citation-validated** — post-generation check flags any invented reference numbers
- **Cached** — two-tier cache (Redis exact + Qdrant semantic) eliminates redundant LLM calls
- **Generated locally** — no data leaves your machine

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
│             Arabic Normalization                │
│  NFKC · alef · tashkeel · tatweel               │
│  ya/waw hamza · Eastern numerals (٠–٩ → 0–9)    │
└──────────────────┬──────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
 ┌──────────────┐    ┌────────────────┐
 │  Jina v3     │    │   BM25 Okapi   │
 │  Dense       │    │   Feature-hash │
 │  1024-dim    │    │   2048-dim     │
 └──────┬───────┘    └──────┬─────────┘
        │                   │
        └─────────┬─────────┘
                  ▼
       ┌─────────────────────┐
       │   Qdrant Prefetch   │
       │   (50 per modality) │
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
       └──────────┬──────────┘
                  ▼
       ┌─────────────────────┐
       │   Gemma 4           │
       │   (LM Studio)       │
       │   Arabic answer     │
       │   + inline [n] refs │
       │   + citation check  │
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
| **Vector DB** | [Qdrant](https://qdrant.tech) (local) | Stores named dense vectors (`dense` + `bm25_dense`) |
| **Dense Embeddings** | [Jina Embeddings v3](https://jina.ai) · 1024-dim | Semantic similarity |
| **Keyword Vectors** | Custom BM25 Okapi · feature-hashed · 2048-dim | Probabilistic keyword ranking as dense vector |
| **Hybrid Fusion** | Reciprocal Rank Fusion via Qdrant | Merges both result lists |
| **Reranker** | [Jina Reranker v2](https://jina.ai) multilingual | Cross-encoder reranking |
| **Cache Tier 1** | [Redis](https://redis.io) · SHA-256 · LRU eviction | Exact-match, microsecond lookup |
| **Cache Tier 2** | Qdrant · Jina v3 · cosine ≥ 0.80 | Semantic-match, paraphrase hits |
| **Generation** | Gemma 4 via [LM Studio](https://lmstudio.ai) | Local Arabic answer synthesis |
| **Web UI** | [Gradio](https://gradio.app) | Arabic RTL chat interface |
| **Arabic NLP** | Custom normalization + mazhab detection | NFKC, diacritics, scholar patterns |

---

## Project Structure

```
FiqhRAG/
│
├── core/                         Library — imported by all entry points
│   ├── arabic_utils.py           Normalization + mazhab detection + citation format
│   ├── bm25.py                   Custom BM25 Okapi + feature-hashed dense vectors (GPU optional)
│   ├── cache.py                  Two-tier cache (Redis exact + Qdrant semantic)
│   ├── config.py                 All settings — edit this file
│   ├── generator.py              Gemma 4 via LM Studio + citation validator
│   ├── qdrant_singleton.py       Shared Qdrant client (prevents file-lock conflicts)
│   └── retriever.py              Hybrid search + Jina reranker + mazhab tagging
│
├── scripts/                      One-time / admin tools
│   ├── ingest.py                 Build the vector index (run once, ~90 min)
│   ├── enrich_payloads.py        Add mazhab tags to existing index (run once, <1 min)
│   └── verify_integration.py    Diagnostic — verify all components are working
│
├── data/                         Generated artifacts  (gitignored)
│   ├── bm25_corpus.pkl           Fitted BM25 model (16,971 documents)
│   └── embed_checkpoint.pkl      Ingestion resume checkpoint
│
├── fiqh_data/                    Source data — 45 JSONL volumes (vol 2–46)
├── qdrant_storage/               Qdrant RAG index        (gitignored, auto-created)
├── qdrant_cache/                 Qdrant semantic cache   (gitignored, auto-created)
│
├── app.py                        ▶  Web UI   — python app.py  or  start.bat
├── main.py                       ▶  CLI      — python main.py
├── start.bat                     Windows launcher (kills stale locks before starting)
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
| **[LM Studio](https://lmstudio.ai)** | Load **Gemma 4**, start local server on port `1234` |
| **[Jina AI API key](https://jina.ai)** | Free tier — used for embeddings + reranker |
| **[Redis](https://redis.io/download)** | Optional — Tier 1 cache degrades gracefully if absent |
| **NVIDIA GPU + CUDA** | Optional — enables GPU-accelerated BM25 encoding (see GPU section below) |

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Configure environment

```bash
# Copy the example and add your Jina API key
cp .env.example .env
```

`.env` contents:
```
JINA_API_KEY=your_key_here
```

### 3 — Set your LM Studio model name

Open `core/config.py` and match the model name shown in LM Studio → Model tab:

```python
LM_STUDIO_MODEL = "gemma-4-26b-a4b"   # ← change to match yours
```

### 4 — Build the index (run once)

Loads all 46 volumes, builds BM25 corpus, embeds with Jina v3, and indexes into Qdrant.

```bash
python scripts/ingest.py
```

> **Note:** Jina free tier is rate-limited to ~6 requests/min. With 16,971 chunks at batch size 32, this takes approximately 90 minutes. The pipeline is **resumable** — if interrupted, re-run the same command and it picks up from the checkpoint automatically. A corrupted checkpoint is detected and discarded, restarting cleanly.

### 5 — Enrich existing index with mazhab tags (run once)

Backfills mazhab detection into every Qdrant point. No re-embedding — pure local operation, completes in under a minute.

```bash
python scripts/enrich_payloads.py
```

Sample output:
```
Enriching 16,971 points...
Done — 16,971 points updated.

Mazhab mention counts:
  حنبلي       3,241  (19.1%)
  شافعي       2,987  (17.6%)
  حنفي        2,814  (16.6%)
  مالكي       2,603  (15.3%)
  جمهور       1,120   (6.6%)
```

### 6 — Verify integration

Run the diagnostic script to confirm all components are wired correctly:

```bash
python scripts/verify_integration.py
```

Expected output:
```
[OK] Config loaded: BM25_USE_GPU=True, BM25_DENSE_DIM=2048
[OK] BM25Okapi imported
[OK] FiqhRetriever initialized
     - BM25 corpus size: 16971 documents
[OK] Dense BM25 query encoding works
[OK] Qdrant collection 'fiqh' exists  (16971 points)
[OK] Gradio app (app.py) syntax valid
[OK] Torch available: CUDA=True
     - Device: NVIDIA GeForce RTX 3080 Ti
```

### 7 — Run

**Web UI** (recommended):
```bash
python app.py
# Opens at http://localhost:7860
```

**CLI:**
```bash
python main.py
```

---

## GPU Acceleration (Optional)

BM25 dense vector encoding supports GPU acceleration via PyTorch. The system auto-detects GPU availability — no config change needed.

**Status:** `BM25_USE_GPU` in `core/config.py` is set automatically at startup:
- `True` when torch is installed and CUDA is available → GPU path
- `False` otherwise → NumPy CPU fallback (fully functional, slightly slower during ingestion)

### Installing torch on Python 3.14

PyTorch stable releases do not yet publish wheels for Python 3.14. Use the nightly build instead. The CUDA 12.6 wheel is backward compatible with CUDA 13.x drivers.

```bash
# First remove any existing (incompatible) torch
pip uninstall torch -y

# Install nightly for Python 3.14 + CUDA 12.6 (works on CUDA 13.x drivers)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126
```

Check your CUDA version with `nvidia-smi` and substitute the correct tag if needed (`cu121`, `cu124`, `cu126`).

---

## Configuration

All settings live in `core/config.py`:

| Setting | Default | Description |
|---|---|---|
| `JINA_EMBED_MODEL` | `jina-embeddings-v3` | Embedding model |
| `JINA_RERANK_MODEL` | `jina-reranker-v2-base-multilingual` | Reranker model |
| `EMBED_DIM` | `1024` | Embedding dimensions |
| `LM_STUDIO_MODEL` | `gemma-4-26b-a4b` | Model loaded in LM Studio |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio server |
| `TOP_K_FETCH` | `50` | Candidates before reranking |
| `TOP_K_FINAL` | `10` | Results after reranking |
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
  "mazhabs": ["حنفي", "شافعي"]
}
```

---

## Performance & Latency

### Baseline (smoke test — 10 questions)

| Stage | Range | Notes |
|---|---|---|
| **Retrieval** | 3–14 s | Jina embed API + Qdrant local + Jina rerank API |
| **Generation** | 53–107 s | Gemma 4 26B on CPU via LM Studio |
| **Cache hit (Tier 1)** | < 1 ms | Redis exact match |
| **Cache hit (Tier 2)** | ~200 ms | Qdrant semantic search + 1 Jina embed call |

### Optimizations applied

| # | What | Where | Impact |
|---|---|---|---|
| **Connection pooling** | `requests.Session()` reuses TCP connections to Jina API | `core/retriever.py` | −0.5–1 s/query |
| **TOP_K_FETCH 50→30** | Fewer candidates sent to reranker (latency ∝ n) | `core/config.py` | −1–3 s/query |
| **MAX_OUTPUT_TOKENS 2048→1024** | Fiqh answers rarely exceed 1k tokens; CPU inference time ∝ tokens | `core/config.py` | −15–30 s/query |
| **Streaming generation** | Tokens yielded to UI as they arrive; user reads while model generates | `core/generator.py`, `app.py` | First token in ~2 s (was 53–107 s wait) |
| **Gradio queue tuned** | `max_size=20` prevents silent drops under load | `app.py` | Multi-user fairness |
| **Request timeouts** | 15 s embed / 20 s rerank; prevents silent hangs | `core/retriever.py` | Reliability |

### Remaining roadmap (not yet implemented)

| Improvement | Expected gain | Effort |
|---|---|---|
| Async Jina calls (`aiohttp`) — overlap embed + rerank latency | −1–3 s | Medium |
| Pass embedding from cache layer to retriever — avoid double-embed on miss | −3–7 s | Medium |
| GPU inference in LM Studio (RTX 3080 Ti) | −40–80 s generation | Low (config only) |
| Qdrant server mode — removes single-process file lock constraint | Enables multi-process deploys | Medium |
| Batch embedding during ingestion already uses `EMBED_BATCH_SIZE=32` | — | Done |

### How to get GPU generation

The biggest single gain is offloading Gemma 4 to your RTX 3080 Ti in LM Studio:

1. Open LM Studio → select your Gemma 4 model
2. In **Load Settings**, set **GPU Layers** to the maximum the model allows
3. Restart the server — generation drops from 53–107 s to ~5–15 s

---

## Inspiration

Inspired by [AIFiqh](https://aifiqh.com) — a specialized Islamic jurisprudence AI platform. This project extends their RAG concept with Qdrant hybrid search, Jina v3 embeddings, custom BM25 Okapi with feature-hashed dense vectors, a two-tier response cache, madhab-aware context, and a fully local generation pipeline via LM Studio.
