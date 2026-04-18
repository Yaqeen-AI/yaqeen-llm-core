<div align="center">

# الموسوعة الفقهية الكويتية — بحث ذكي
### FiqhRAG · Islamic Jurisprudence AI Search

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
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
│                Two-Tier Cache                    │
│  Tier 1: Redis SHA-256 hash  →  O(1) exact hit  │
│  Tier 2: Jina v3 cosine ≥ 0.80 → semantic hit  │
└────────────────────┬────────────────────────────┘
                     │ miss
                     ▼
┌─────────────────────────────────────────────────┐
│             Arabic Normalization                 │
│  NFKC · alef · tashkeel · tatweel               │
│  ya/waw hamza · Eastern numerals (٠–٩ → 0–9)   │
└──────────────────┬──────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         ▼                    ▼
 ┌──────────────┐    ┌────────────────┐
 │  Jina v3     │    │   TF-IDF       │
 │  Dense       │    │   Sparse       │
 │  1024-dim    │    │   char (3–5)   │
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
| **Vector DB** | [Qdrant](https://qdrant.tech) (local) | Stores dense + sparse vectors |
| **Dense Embeddings** | [Jina Embeddings v3](https://jina.ai) · 1024-dim | Semantic similarity |
| **Sparse Vectors** | scikit-learn TF-IDF · char n-grams (3–5) | Keyword matching |
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
│   ├── cache.py                  Two-tier cache (Redis exact + Qdrant semantic)
│   ├── config.py                 All settings — edit this file
│   ├── generator.py              Gemma 4 via LM Studio + citation validator
│   ├── qdrant_singleton.py       Shared Qdrant client (prevents file-lock conflicts)
│   └── retriever.py              Hybrid search + Jina reranker + mazhab tagging
│
├── scripts/                      One-time / admin tools
│   ├── ingest.py                 Build the vector index (run once)
│   └── enrich_payloads.py        Add mazhab tags to existing index (run once)
│
├── data/                         Generated artifacts  (gitignored)
│   ├── tfidf_model.pkl           Fitted TF-IDF model
│   └── embed_checkpoint.pkl      Ingestion resume checkpoint
│
├── fiqh_data/                    Source data — 46 JSONL volumes
├── qdrant_storage/               Qdrant RAG index        (gitignored, auto-created)
├── qdrant_cache/                 Qdrant semantic cache   (gitignored, auto-created)
│
├── app.py                        ▶  Web UI   — python app.py  or  start.bat
├── main.py                       ▶  CLI      — python main.py
├── start.bat                     Windows launcher (kills stale locks before starting)
│
├── .env                          API keys  (not committed)
├── .env.example                  Key template
└── requirements.txt              Python dependencies
```

---

## Setup

### Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.11 or higher |
| [LM Studio](https://lmstudio.ai) | Load **Gemma 4**, start local server on port `1234` |
| [Jina AI API key](https://jina.ai) | Free tier — used for embeddings + reranker |
| [Redis](https://redis.io/download) | Running locally on port `6379` (optional — Tier 1 cache degrades gracefully if absent) |

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env
# Add your Jina API key to .env
```

### 3 — Set your LM Studio model name

Open `core/config.py` and match the model name shown in LM Studio → Model tab:

```python
LM_STUDIO_MODEL = "gemma-3-27b-it"   # ← change to match yours
```

### 4 — Build the index (run once)

Loads all 46 volumes, fits TF-IDF, embeds with Jina v3, and indexes into Qdrant.

```bash
python scripts/ingest.py
```

> **Note:** Jina free tier is rate-limited to ~6 requests/min. With 16,971 chunks at batch size 32, this takes approximately 90 minutes. The pipeline is resumable — if interrupted, re-run the same command and it picks up from the checkpoint.

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

### 6 — Run

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

## Configuration

All settings live in `core/config.py`:

| Setting | Default | Description |
|---|---|---|
| `JINA_EMBED_MODEL` | `jina-embeddings-v3` | Embedding model |
| `JINA_RERANK_MODEL` | `jina-reranker-v2-base-multilingual` | Reranker model |
| `EMBED_DIM` | `1024` | Embedding dimensions |
| `LM_STUDIO_MODEL` | `gemma-3-27b-it` | Model loaded in LM Studio |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio server |
| `TOP_K_FETCH` | `50` | Candidates before reranking |
| `TOP_K_FINAL` | `10` | Results after reranking |
| `TFIDF_MAX_FEATURES` | `65536` | TF-IDF vocabulary size |
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_MAX_MEMORY` | `256mb` | Redis memory cap (LRU eviction) |
| `SEMANTIC_THRESHOLD` | `0.80` | Min cosine score for Tier 2 cache hit |
| `BATCH_DELAY` | `10.0s` | Delay between Jina API calls (ingest) |

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

Applied before TF-IDF fitting and query encoding. Jina embeddings receive the original text — it handles Arabic natively.

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

## Inspiration

Inspired by [AIFiqh](https://aifiqh.com) — a specialized Islamic jurisprudence AI platform. This project extends their RAG concept with Qdrant hybrid search, Jina v3 embeddings, character-level TF-IDF for Arabic morphology, a two-tier response cache, madhab-aware context, and a fully local generation pipeline via LM Studio.
