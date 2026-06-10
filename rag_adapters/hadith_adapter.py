from __future__ import annotations

import os
# Disable CUDA to prevent device/driver mismatch errors with local PyTorch/fastembed models
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from typing import Any

from orchestrator.models import Citation, RagSource, RetrievedDocument, SourceRetrievalConfig
from rag_adapters.base import RagAdapter



class HadithAdapter(RagAdapter):
    def __init__(self, retriever: Any | None = None) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            from hadith_rag.retrieval.retriever import HadithRagConfig, get_hadith_retriever

            cfg = HadithRagConfig(
                jina_api_key=os.getenv("JINA_API_KEY", ""),
                qdrant_url=os.getenv("HADITH_QDRANT_URL", os.getenv("QDRANT_URL", "http://localhost:6333")),
                qdrant_api_key=os.getenv("HADITH_QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", "")),
                hadith_collection_name=os.getenv("HADITH_COLLECTION_NAME", "hadiths"),
                enable_hybrid=False,  # fastembed not installed; flip to True when available
            )
            self._retriever = get_hadith_retriever(cfg)
        return self._retriever

    async def retrieve(self, query: str, config: SourceRetrievalConfig) -> list[RetrievedDocument]:
        _ensure_setup(self.retriever)
        # Force dense mode — fastembed (required for hybrid) is not installed
        mode = "dense" if not getattr(self.retriever.cfg, "enable_hybrid", False) else config.mode
        filters = dict(config.filters)
        nodes = await self.retriever.search(
            query=query,
            mode=mode,
            top_k=config.top_k,
            similarity_top_k=config.similarity_top_k,
            rerank_top_n=config.rerank_top_n,
            skip_rerank=config.skip_rerank,
            book=filters.get("book"),
            grade=filters.get("grade"),
            min_grade=filters.get("min_grade"),
            rawi=filters.get("rawi"),
            category=filters.get("category"),
            subcategory_name=filters.get("subcategory_name"),
            mohadeth=filters.get("mohadeth"),
            has_explanation=filters.get("has_explanation"),
            canonical_group_id=filters.get("canonical_group_id"),
            prioritize_sahihayn=bool(filters.get("prioritize_sahihayn", True)),
            dedup_canonical=bool(filters.get("dedup_canonical", True)),
        )
        return [_node_to_document(node, index) for index, node in enumerate(nodes)]

def _node_to_document(node: Any, index: int) -> RetrievedDocument:
    metadata = dict(getattr(node.node, "metadata", {}) or {})
    label = _hadith_label(metadata, index)
    
    text = getattr(node.node, "text", "")
    explanation = _metadata_text(metadata, "explanation", "sharh", "شرح")
    if explanation and "الشرح:" not in text:
        text = f"{text}\n\nالشرح:\n{explanation}"
        
    return RetrievedDocument(
        id=str(getattr(node.node, "id_", "") or metadata.get("hadith_tag") or metadata.get("hadith_id") or f"hadith-{index}"),
        source=RagSource.HADITH,
        text=text,
        score=float(node.score or 0.0),
        citation=Citation(source=RagSource.HADITH, label=label, metadata=metadata),
        metadata=metadata,
    )

def _hadith_label(metadata: dict[str, Any], index: int) -> str:
    book = metadata.get("book") or metadata.get("masdar") or metadata.get("source") or "Hadith"
    number = metadata.get("numberOrPage") or metadata.get("hadith_number") or metadata.get("safha_raqam") or metadata.get("page") or index + 1
    return f"[{book} {number}]"


def _metadata_text(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value:
            return str(value).strip()
    return ""


def _ensure_setup(retriever: Any) -> None:
    if getattr(retriever, "_index", None) is None and hasattr(retriever, "setup"):
        retriever.setup()
