"""
One-time ingestion pipeline.

    python ingest.py

Steps:
  1. Load all JSONL chunks from fiqh_data/
  2. Normalize Arabic text → fit TF-IDF → sparse vectors
  3. Embed all chunks with Jina Embeddings v3 → dense vectors
  4. Upsert both into Qdrant (named vectors: "dense" + "sparse")
"""

import json
import pickle
import sys
import time

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PointStruct, SparseIndexParams,
    SparseVector, SparseVectorParams, VectorParams,
)

from core.config import (
    COLLECTION_NAME, DATA_DIR, EMBED_BATCH_SIZE, EMBED_DIM,
    JINA_API_KEY, JINA_EMBED_MODEL, QDRANT_PATH,
    TFIDF_MAX_FEATURES, TFIDF_PATH, UPSERT_BATCH_SIZE,
)

EMBED_CHECKPOINT = "embed_checkpoint.pkl"   # resume file for embeddings
BATCH_DELAY      = 10.0                      # seconds between Jina API calls (free tier ~6 RPM)
MAX_RETRIES      = 6                         # retries on 429 / 5xx
RETRY_BASE       = 5.0                       # exponential backoff base (seconds)
from core.arabic_utils import detect_mazhabs, normalize_corpus


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_records() -> list[dict]:
    files = sorted(DATA_DIR.glob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl files found in {DATA_DIR}")
    records = []
    for path in tqdm(files, desc="Loading volumes"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# 2. TF-IDF sparse vectors
# ---------------------------------------------------------------------------

def fit_tfidf(texts: list[str]):
    print(f"Fitting TF-IDF on {len(texts):,} documents...")
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        strip_accents=None,
    )
    matrix = tfidf.fit_transform(texts)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    print(f"TF-IDF fitted — vocabulary: {len(tfidf.vocabulary_):,} terms")
    return matrix


def row_to_sparse(matrix, idx: int) -> SparseVector:
    row = matrix[idx].tocoo()
    return SparseVector(indices=row.col.tolist(), values=row.data.tolist())


# ---------------------------------------------------------------------------
# 3. Jina dense embeddings
# ---------------------------------------------------------------------------

MAX_CHARS = 6000  # Jina v3 max ~8192 tokens; Arabic ~4 chars/token → 6000 chars is safe


def _embed_batch_with_retry(batch: list[str], headers: dict) -> list[list[float]]:
    """Embed one batch with exponential backoff on 429/5xx and truncation on 400."""
    # Truncate any oversized texts upfront
    safe_batch = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in batch]

    for attempt in range(MAX_RETRIES):
        resp = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers=headers,
            json={"model": JINA_EMBED_MODEL, "input": safe_batch,
                  "dimensions": EMBED_DIM, "task": "retrieval.passage"},
        )
        if resp.status_code == 200:
            data = sorted(resp.json()["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in data]
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = RETRY_BASE * (2 ** attempt)
            print(f"\n  [{resp.status_code}] rate limited — waiting {wait:.0f}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
        elif resp.status_code == 400:
            # Further truncate and retry once
            safe_batch = [t[:2000] for t in safe_batch]
            print(f"\n  [400] bad request — truncating to 2000 chars and retrying")
        else:
            resp.raise_for_status()
    sys.exit("Max retries exceeded on Jina API.")


def jina_embed(texts: list[str]) -> list[list[float]]:
    """Embed all texts with checkpoint/resume support."""
    if not JINA_API_KEY:
        sys.exit("JINA_API_KEY not set — add it to .env")

    # Load checkpoint if exists
    checkpoint_path = TFIDF_PATH.parent / EMBED_CHECKPOINT
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            all_vecs = pickle.load(f)
        start_batch = len(all_vecs) // EMBED_BATCH_SIZE
        print(f"Resuming from batch {start_batch} ({len(all_vecs):,} vectors already done)")
    else:
        all_vecs = []
        start_batch = 0

    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    batches = range(0, len(texts), EMBED_BATCH_SIZE)

    for i in tqdm(batches, desc="Jina v3 embedding", initial=start_batch, total=len(batches)):
        if i < start_batch * EMBED_BATCH_SIZE:
            continue
        batch = texts[i : i + EMBED_BATCH_SIZE]
        vecs = _embed_batch_with_retry(batch, headers)
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
        vectors_config={"dense": VectorParams(size=EMBED_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))},
    )
    print(f"Collection '{COLLECTION_NAME}' created.")


def upsert(client: QdrantClient, records: list[dict], dense_vecs: list, sparse_matrix) -> None:
    for i in tqdm(range(0, len(records), UPSERT_BATCH_SIZE), desc="Upserting to Qdrant"):
        batch = records[i : i + UPSERT_BATCH_SIZE]
        points = [
            PointStruct(
                id=i + j,
                vector={"dense": dense_vecs[i + j], "sparse": row_to_sparse(sparse_matrix, i + j)},
                payload={
                    "chunk_text": rec["chunk_text"],
                    "volume_id":  rec["volume_id"],
                    "book_page":  rec["book_page"],
                    "chunk_page": rec["chunk_page"],
                    "source_url": rec.get("source_url", ""),
                    "mazhabs":    detect_mazhabs(rec["chunk_text"]),
                },
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

    # Skip normalization + TF-IDF if already fitted
    if TFIDF_PATH.exists():
        print(f"TF-IDF already fitted at {TFIDF_PATH} — skipping.")
        import pickle as _pkl
        with open(TFIDF_PATH, "rb") as f:
            tfidf = _pkl.load(f)
        normalized = [normalize_corpus(t) for t in tqdm(texts, desc="Normalizing")]
        sparse_matrix = tfidf.transform(normalized)
    else:
        print("Normalizing Arabic text...")
        normalized = [normalize_corpus(t) for t in tqdm(texts, desc="Normalizing")]
        sparse_matrix = fit_tfidf(normalized)

    print(f"\nEmbedding {len(texts):,} chunks with Jina v3...")
    dense_vecs = jina_embed(texts)   # original text for Jina (handles diacritics natively)

    client = QdrantClient(path=QDRANT_PATH)
    setup_collection(client)
    upsert(client, records, dense_vecs, sparse_matrix)

    info = client.get_collection(COLLECTION_NAME)
    print(f"\nDone! {info.points_count:,} points indexed.")


if __name__ == "__main__":
    main()
