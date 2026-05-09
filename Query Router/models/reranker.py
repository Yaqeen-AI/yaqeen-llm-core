import torch
from sentence_transformers import CrossEncoder

# Optimize: bind to cuda explicitly and load weights in fp16 to utilize Tensor Cores
device = "cuda" if torch.cuda.is_available() else "cpu"
model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2", 
    max_length=512,
    device=device,
    model_kwargs=model_kwargs
)