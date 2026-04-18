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
LM_STUDIO_MODEL    = "gemma-3-27b-it"   # name shown in LM Studio → Model tab
LM_STUDIO_API_KEY  = "lm-studio"        # LM Studio ignores this, but openai SDK requires it

# --- Qdrant (local file storage) ---
QDRANT_PATH     = str(Path(__file__).parent.parent / "qdrant_storage")
COLLECTION_NAME = "fiqh"

# --- Data paths ---
DATA_DIR        = Path(__file__).parent.parent / "fiqh_data"
DATA_ARTIFACTS  = Path(__file__).parent.parent / "data"
TFIDF_PATH      = DATA_ARTIFACTS / "tfidf_model.pkl"

# --- Search tuning ---
EMBED_BATCH_SIZE  = 32   # chunks per Jina API call during ingestion
UPSERT_BATCH_SIZE = 256  # points per Qdrant upsert
TOP_K_FETCH       = 50   # candidate pool before reranking
TOP_K_FINAL       = 10   # results returned after reranking

# --- TF-IDF ---
TFIDF_MAX_FEATURES = 65536

# --- Two-Tier Cache ---
REDIS_HOST        = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT        = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB          = int(os.getenv("REDIS_DB", 0))
REDIS_MAX_MEMORY  = os.getenv("REDIS_MAX_MEMORY", "256mb")

QDRANT_CACHE_PATH  = str(Path(__file__).parent.parent / "qdrant_cache")
CACHE_COLLECTION   = "fiqh_query_cache"
SEMANTIC_THRESHOLD = 0.80
