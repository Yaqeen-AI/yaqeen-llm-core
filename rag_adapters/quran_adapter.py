from __future__ import annotations

import logging
import os
# Disable CUDA to prevent device/driver mismatch errors with local PyTorch/fastembed models
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from typing import Any

from orchestrator.models import Citation, RagSource, RetrievedDocument, SourceRetrievalConfig
from rag_adapters.base import RagAdapter

logger = logging.getLogger(__name__)


class QuranAdapter(RagAdapter):
    def __init__(self, retriever: Any | None = None) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            from quran_rag.retriever import get_quran_retriever

            self._retriever = get_quran_retriever()
        return self._retriever

    async def retrieve(self, query: str, config: SourceRetrievalConfig) -> list[RetrievedDocument]:
        _ensure_setup(self.retriever)
        filters = dict(config.filters)

        # Consume (and discard) parent/child preference keys — we do NOT filter
        # by is_parent in Qdrant because:
        #   1. Qdrant's bool payload index isn't reachable via LlamaIndex's
        #      MetadataFilter (which only accepts int|float|str|list[...]).
        #   2. Parent/summary chunks naturally score far lower on semantic
        #      similarity for leaf-level queries, so the filter is unnecessary.
        filters.pop("prefer_child", None)
        filters.pop("is_parent", None)

        # Parent context is fetched and attached automatically when enabled.
        include_parent_context: bool = bool(filters.pop("include_parent_context", True))

        nodes = await self.retriever.search(
            query=query,
            mode=config.mode,
            similarity_top_k=config.similarity_top_k,
            rerank_top_n=config.rerank_top_n,
            skip_rerank=config.skip_rerank,
            include_parent_context=include_parent_context,
            **_allowed(filters, {"surah", "revelation_type", "juz", "ruku", "hizb_quarter", "edition", "ayah_text"}),
        )
        return [_node_to_document(node, index) for index, node in enumerate(nodes[: config.top_k])]


def _node_to_document(node: Any, index: int) -> RetrievedDocument:
    metadata = dict(getattr(node.node, "metadata", {}) or {})
    label = _quran_label(metadata, index)

    text = getattr(node.node, "text", "")

    return RetrievedDocument(
        id=str(getattr(node.node, "id_", "") or metadata.get("chunk_id") or f"quran-{index}"),
        source=RagSource.QURAN,
        text=text,
        score=float(node.score or 0.0),
        citation=Citation(source=RagSource.QURAN, label=label, metadata=metadata),
        metadata=metadata,
    )


def _quran_label(metadata: dict[str, Any], index: int) -> str:
    surah = metadata.get("surah_name_arabic") or metadata.get("surah_name_english") or metadata.get("surah") or metadata.get("surah_number")
    ayah = metadata.get("ayah_range") or metadata.get("ayah_from") or metadata.get("ayah") or metadata.get("verse")
    if surah and ayah:
        return f"[Quran {surah} {ayah}]"
    if surah:
        return f"[Quran Surah {surah}]"
    return f"[Quran Evidence {index + 1}]"


def _ensure_setup(retriever: Any) -> None:
    if getattr(retriever, "_index", None) is None and hasattr(retriever, "setup"):
        retriever.setup()


def _allowed(values: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key in keys and value is not None}
