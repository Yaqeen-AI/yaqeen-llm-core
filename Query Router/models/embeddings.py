import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

# Optimize: explicit device binding and fast batch encoding
device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={"batch_size": 32, "normalize_embeddings": False}
)