"""
One-time ingestion pipeline.

    python ingest.py

Steps:
  1. Load all JSONL chunks from fiqh_data/
  2. Normalize Arabic text → build BM25 corpus → dense hashed vectors
  3. Embed all chunks with Jina Embeddings v3 → dense vectors
  4. Upsert both into Qdrant (named vectors: "dense" + "bm25_dense")
"""

import json
import os
import pathlib
import pickle
import sys
import time

# Ensure the project root is on sys.path so `core.*` imports resolve
# regardless of which directory the script is run from.
_PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import requests
from tqdm import tqdm

from llama_index.core.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PayloadSchemaType, PointStruct, VectorParams,
)

from core.config import (
    COLLECTION_NAME, DATA_DIR, EMBED_BATCH_SIZE, EMBED_DIM,
    JINA_API_KEY, QDRANT_PATH,
    BM25_PATH, BM25_K1, BM25_B, BM25_DENSE_DIM, BM25_USE_GPU, UPSERT_BATCH_SIZE,
)
from core.bm25 import BM25Okapi
from core.embeddings import JinaEmbedding

_embed_model = JinaEmbedding()

EMBED_CHECKPOINT = "embed_checkpoint.pkl"   # resume file for embeddings
BATCH_DELAY      = 10.0                      # seconds between Jina API calls (free tier ~6 RPM)
MAX_RETRIES      = 6                         # retries on 429 / 5xx
RETRY_BASE       = 5.0                       # exponential backoff base (seconds)
from core.arabic_utils import detect_mazhabs, detect_fiqh_topic, normalize_corpus
from core.schema import QdrantPayload


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_records() -> list[dict]:
    files = sorted(DATA_DIR.glob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl files found in {DATA_DIR}")
    records = []
    for path in tqdm(files, desc="Loading volumes"):
        with open(str(path), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def records_to_documents(records: list[dict]) -> list[Document]:
    """Wrap raw JSONL records as LlamaIndex Document objects."""
    return [
        Document(
            text=rec["chunk_text"],
            metadata={k: v for k, v in rec.items() if k != "chunk_text"},
            id_=str(idx),
        )
        for idx, rec in enumerate(records)
    ]


# ---------------------------------------------------------------------------
# 2. BM25 dense vectors
# ---------------------------------------------------------------------------

def simple_tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for Arabic text."""
    return text.split()


def build_bm25_corpus(texts: list[str]) -> BM25Okapi:
    """Build BM25 corpus from normalized texts."""
    print(f"Building BM25 corpus on {len(texts):,} documents...")
    tokenized = [simple_tokenize(t) for t in texts]
    bm25 = BM25Okapi(
        tokenized,
        k1=BM25_K1,
        b=BM25_B,
        dense_dim=BM25_DENSE_DIM,
        use_gpu=BM25_USE_GPU,
    )
    tmp_path = BM25_PATH.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(bm25, f)
    os.replace(tmp_path, BM25_PATH)
    avg_tokens = sum(len(t) for t in tokenized) // len(tokenized) if tokenized else 0
    print(f"BM25 corpus built — {bm25.corpus_size} documents, ~{avg_tokens} avg tokens/doc")
    return bm25


# ---------------------------------------------------------------------------
# 3. Jina dense embeddings
# ---------------------------------------------------------------------------

MAX_CHARS = 6000  # Jina v3 max ~8192 tokens; Arabic ~4 chars/token → 6000 chars is safe


def _embed_batch_with_retry(batch: list[str]) -> list[list[float]]:
    """Embed one batch via JinaEmbedding with exponential backoff on 429/5xx."""
    safe_batch = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in batch]

    for attempt in range(MAX_RETRIES):
        try:
            return _embed_model._call_jina(safe_batch, task="retrieval.passage")
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status in (401, 403):
                sys.exit(f"Jina authentication failed ({status}). Check JINA_API_KEY in .env.")
            if status in (429, 500, 502, 503, 504):
                wait = RETRY_BASE * (2 ** attempt)
                print(f"\n  [{status}] rate limited — waiting {wait:.0f}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            elif status == 400:
                safe_batch = [t[:2000] for t in safe_batch]
                print(f"\n  [400] bad request — truncating to 2000 chars and retrying")
            else:
                raise
        except Exception:
            raise
    sys.exit("Max retries exceeded on Jina API.")


def jina_embed(texts: list[str]) -> list[list[float]]:
    """Embed all texts with checkpoint/resume support."""
    if not JINA_API_KEY:
        sys.exit("JINA_API_KEY not set — add it to .env")

    # Load checkpoint if exists
    checkpoint_path = BM25_PATH.parent / EMBED_CHECKPOINT
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "rb") as f:
                all_vecs = pickle.load(f)
            start_batch = len(all_vecs) // EMBED_BATCH_SIZE
            print(f"Resuming from batch {start_batch} ({len(all_vecs):,} vectors already done)")
        except Exception as exc:
            print(f"[WARN] Checkpoint corrupt, starting fresh: {exc}")
            all_vecs = []
            start_batch = 0
    else:
        all_vecs = []
        start_batch = 0

    batches = range(0, len(texts), EMBED_BATCH_SIZE)

    for i in tqdm(batches, desc="Jina v3 embedding", initial=start_batch, total=len(batches)):
        if i < start_batch * EMBED_BATCH_SIZE:
            continue
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vecs = _embed_batch_with_retry(batch)
        all_vecs.extend(vecs)
        time.sleep(BATCH_DELAY)

        # Save checkpoint every 50 batches
        if (i // EMBED_BATCH_SIZE) % 50 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump(all_vecs, f)

    # Save final checkpoint then clean up
    with open(checkpoint_path, "wb") as f:
        pickle.dump(all_vecs, f)

    return all_vecs


# ---------------------------------------------------------------------------
# 4. Qdrant upsert
# ---------------------------------------------------------------------------

def setup_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        print(f"Dropping existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            "bm25_dense": VectorParams(size=BM25_DENSE_DIM, distance=Distance.COSINE),
        },
    )
    for field in ("mazhabs", "volume_id", "fiqh_topic"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    print(f"Collection '{COLLECTION_NAME}' created with payload indexes on mazhabs + volume_id + fiqh_topic.")


def upsert(client: QdrantClient, records: list[dict], dense_vecs: list, bm25) -> None:
    for i in tqdm(range(0, len(records), UPSERT_BATCH_SIZE), desc="Upserting to Qdrant"):
        batch = records[i : i + UPSERT_BATCH_SIZE]
        points = [
            PointStruct(
                id=i + j,
                vector={
                    "dense": dense_vecs[i + j],
                    "bm25_dense": bm25.dense_vector_for_doc(bm25.corpus[i + j]),
                },
                payload=QdrantPayload(
                    chunk_text=rec["chunk_text"],
                    volume_id=rec["volume_id"],
                    book_page=rec["book_page"],
                    chunk_page=rec["chunk_page"],
                    source_url=rec.get("source_url", ""),
                    mazhabs=detect_mazhabs(rec["chunk_text"]),
                    fiqh_topic=(_t[0] if (_t := detect_fiqh_topic(rec["chunk_text"])) and len(_t) == 1 else ""),
                ),
            )
            for j, rec in enumerate(batch)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== FiqhRAG Ingestion Pipeline ===\n")

    records = load_records()
    print(f"Loaded {len(records):,} chunks from {DATA_DIR}\n")

    texts = [r["chunk_text"] for r in records]

    print("Normalizing Arabic text...")
    normalized = [normalize_corpus(t) for t in tqdm(texts, desc="Normalizing")]
    bm25 = build_bm25_corpus(normalized)

    print(f"\nEmbedding {len(texts):,} chunks with Jina v3...")
    dense_vecs = jina_embed(texts)   # original text for Jina (handles diacritics natively)

    client = QdrantClient(path=QDRANT_PATH)
    setup_collection(client)
    upsert(client, records, dense_vecs, bm25)

    info = client.get_collection(COLLECTION_NAME)
    print(f"\nDone! {info.points_count:,} points indexed.")


if __name__ == "__main__":
    main()
