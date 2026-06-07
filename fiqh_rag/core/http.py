"""
Shared requests.Session for all Jina API calls.

A single session pools TCP connections to api.jina.ai across JinaEmbedding
and JinaReranker, saving ~50ms per call from connection reuse.
"""

import requests

jina_session = requests.Session()
