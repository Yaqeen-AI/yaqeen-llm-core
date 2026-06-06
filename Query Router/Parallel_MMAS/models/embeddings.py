"""Embedding model singleton — lazy, GPU-aware."""
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 32, "normalize_embeddings": False},
    )
except Exception as e:
    print(f"[embeddings] WARNING: Failed to load embedding model ({e}); embeddings unavailable.")
    embeddings = None