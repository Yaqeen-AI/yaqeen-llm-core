"""Cross-encoder reranker singleton — lazy, GPU-aware."""
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
except Exception:
    device = "cpu"
    model_kwargs = {}

try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length=512,
        device=device,
        model_kwargs=model_kwargs,
    )
except Exception as e:
    print(f"[reranker] WARNING: Failed to load reranker ({e}); reranker unavailable.")
    reranker = None