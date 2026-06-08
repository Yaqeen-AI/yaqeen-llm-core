import os
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
except ImportError:
    print("qdrant_client is not installed.")
    exit(1)

cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qdrant_cache")
print(f"Connecting to Qdrant at: {cache_path}")
q = QdrantClient(path=cache_path)

if q.collection_exists("mmas_query_cache"):
    q.delete_collection("mmas_query_cache")
    print("Flushed mmas_query_cache (deleted collection).")
    q.create_collection(
        collection_name="mmas_query_cache",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    print("Recreated empty mmas_query_cache collection.")
else:
    print("Cache collection 'mmas_query_cache' does not exist.")
