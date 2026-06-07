"""
Central metadata schema for the FiqhRAG pipeline.

QdrantPayload  — fields stored in / read from every Qdrant point.
NodeMetadata   — fields that live in a LlamaIndex TextNode.metadata dict.

These TypedDicts are the single source of truth for field names.
They carry zero runtime cost — pure type annotations for IDE support
and catching field-name typos at development time.
"""

from typing import TypedDict


class QdrantPayload(TypedDict):
    """Payload stored on every Qdrant point (written by ingest.py / enrich_payloads.py)."""
    chunk_text:  str
    volume_id:   str
    book_page:   str
    chunk_page:  str
    source_url:  str
    mazhabs:     list[str]   # detected Islamic schools of law
    fiqh_topic:  str         # dominant Fiqh topic category (empty string = unclassified)


class NodeMetadata(TypedDict):
    """Metadata dict inside every LlamaIndex TextNode (set by result_to_node())."""
    volume_id:    str
    book_page:    str
    chunk_page:   str
    source_url:   str
    mazhabs:      list[str]
    fiqh_topic:   str
    qdrant_score: float
    rerank_score: float
    short_ref:    str
    rank:         int
