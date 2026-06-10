from __future__ import annotations
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import asyncio
import sys
from pathlib import Path
from typing import Any

from orchestrator.models import Citation, RagSource, RetrievedDocument, SourceRetrievalConfig
from rag_adapters.base import RagAdapter
class FiqhAdapter(RagAdapter):
    def __init__(self, retriever: Any | None = None) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            fiqh_root = Path(__file__).resolve().parents[1] / "fiqh_rag"
            if str(fiqh_root) not in sys.path:
                sys.path.append(str(fiqh_root))
            from fiqh_rag.core.retriever import FiqhLlamaRetriever

            self._retriever = FiqhLlamaRetriever()
        return self._retriever

    async def retrieve(self, query: str, config: SourceRetrievalConfig) -> list[RetrievedDocument]:
        filters = dict(config.filters)
        top_k_fetch = config.similarity_top_k
        top_k_final = config.top_k
        if hasattr(self.retriever, "_top_k_fetch"):
            self.retriever._top_k_fetch = top_k_fetch
        if hasattr(self.retriever, "_top_k_final"):
            self.retriever._top_k_final = top_k_final
            if hasattr(self.retriever, "_reranker"):
                self.retriever._reranker.top_n = top_k_final

        nodes = await asyncio.to_thread(
            self.retriever.retrieve_with_filter,
            query,
            filters.get("mazhab_filter"),
            filters.get("topic_filter") or filters.get("category"),
        )
        return [_node_to_document(node, index) for index, node in enumerate(nodes[: config.top_k])]


def _node_to_document(node: Any, index: int) -> RetrievedDocument:
    metadata = dict(getattr(node.node, "metadata", {}) or {})
    label = metadata.get("short_ref") or _fiqh_label(metadata, index)
    return RetrievedDocument(
        id=str(getattr(node.node, "id_", "") or f"fiqh-{index}"),
        source=RagSource.FIQH,
        text=getattr(node.node, "text", ""),
        score=float(node.score or metadata.get("rerank_score") or metadata.get("qdrant_score") or 0.0),
        citation=Citation(source=RagSource.FIQH, label=f"[{label}]", metadata=metadata),
        metadata=metadata,
    )


def _fiqh_label(metadata: dict[str, Any], index: int) -> str:
    volume = metadata.get("volume_id")
    page = metadata.get("book_page")
    topic = metadata.get("fiqh_topic")
    
    label_parts = ["الموسوعة الفقهية"]
    if volume:
        label_parts.append(volume)
    if page:
        label_parts.append(page)
    if topic:
        label_parts.append(f"({topic})")
        
    if len(label_parts) > 1:
        return " - ".join(label_parts)
        
    return f"Fiqh Evidence {index + 1}"
