import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Jina AI (embeddings + reranker) ---
JINA_API_KEY      = os.getenv("JINA_API_KEY", "")
JINA_EMBED_MODEL  = "jina-embeddings-v3"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"
EMBED_DIM         = 1024

# --- LM Studio (local Gemma 4) ---
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL    = "gemma-4-26b-a4b"   # name shown in LM Studio → Model tab
LM_STUDIO_API_KEY  = "lm-studio"        # LM Studio ignores this, but openai SDK requires it

# --- Qdrant (local file storage) ---
QDRANT_PATH     = str(Path(__file__).parent.parent / "qdrant_storage")
COLLECTION_NAME = "fiqh"

# --- Data paths ---
DATA_DIR        = Path(__file__).parent.parent / "fiqh_data"
DATA_ARTIFACTS  = Path(__file__).parent.parent / "data"
BM25_PATH       = DATA_ARTIFACTS / "bm25_corpus.pkl"

# --- Search tuning ---
EMBED_BATCH_SIZE  = 32   # chunks per Jina API call during ingestion
UPSERT_BATCH_SIZE = 256  # points per Qdrant upsert
TOP_K_FETCH       = 30   # candidate pool before reranking  (was 50; reranker latency ∝ n)
TOP_K_FINAL       = 10   # results returned after reranking

# --- Generation ---
MAX_OUTPUT_TOKENS = 1024  # max LLM output tokens (was 2048; fiqh answers rarely exceed 1k)

# --- BM25 sparse vectors ---
# BM25 parameters: k1 (term frequency saturation), b (length normalization)
BM25_K1           = 1.5
BM25_B            = 0.75

# --- BM25 dense vectors ---
# Dense BM25 uses feature hashing into a fixed-size vector so Qdrant can search it
# as a normal dense named vector. GPU acceleration is optional (torch if installed).
BM25_DENSE_DIM    = 2048


def _gpu_available() -> bool:
    try:
        import torch  # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:
        return False


BM25_USE_GPU = _gpu_available()

# --- Two-Tier Cache ---
REDIS_HOST        = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT        = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB          = int(os.getenv("REDIS_DB", 0))
REDIS_MAX_MEMORY  = os.getenv("REDIS_MAX_MEMORY", "256mb")

QDRANT_CACHE_PATH  = str(Path(__file__).parent.parent / "qdrant_cache")
CACHE_COLLECTION   = "fiqh_query_cache"
SEMANTIC_THRESHOLD = 0.80
