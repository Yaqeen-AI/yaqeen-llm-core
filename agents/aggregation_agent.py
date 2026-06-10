from __future__ import annotations

import hashlib

from orchestrator.models import AggregatedEvidence, RagSource, RetrievedDocument

# Small score boost for child Quran chunks that carry actual ayah text.
_CHILD_AYAH_BOOST = 0.05


class AggregationAgent:
    async def run(self, documents: list[RetrievedDocument], final_top_k: int = 8) -> AggregatedEvidence:
        if not documents:
            return AggregatedEvidence()

        max_by_source = {}
        for document in documents:
            max_by_source[document.source] = max(max_by_source.get(document.source, 0.0), document.score or 0.0)

        deduped: dict[str, RetrievedDocument] = {}
        for document in documents:
            signature = _document_signature(document)
            source_max = max_by_source.get(document.source) or 1.0
            normalized = max(0.0, min(1.0, (document.score or 0.0) / source_max))

            # Prioritize child Quran chunks that contain actual ayah text over
            # parent/summary chunks, which lack direct verse quotations.
            if (
                document.source == RagSource.QURAN
                and not document.metadata.get("is_parent", False)
                and document.metadata.get("ayah_text")
            ):
                normalized = min(1.0, normalized + _CHILD_AYAH_BOOST)

            candidate = document.model_copy(update={"normalized_score": normalized})
            existing = deduped.get(signature)
            if existing is None or candidate.normalized_score > existing.normalized_score:
                deduped[signature] = candidate

        ranked = sorted(
            deduped.values(),
            key=lambda doc: (doc.normalized_score, doc.score),
            reverse=True,
        )
        return AggregatedEvidence(documents=_balanced_top_k(ranked, final_top_k))


def _document_signature(document: RetrievedDocument) -> str:
    citation = document.citation.label if document.citation else ""
    raw = f"{document.source}:{citation}:{document.text[:300]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _balanced_top_k(ranked: list[RetrievedDocument], final_top_k: int) -> list[RetrievedDocument]:
    sources = []
    seen_sources: set[RagSource] = set()
    for document in ranked:
        if document.source in seen_sources:
            continue
        sources.append(document.source)
        seen_sources.add(document.source)

    if len(sources) <= 1:
        return ranked[:final_top_k]

    selected: list[RetrievedDocument] = []
    selected_ids: set[str] = set()
    per_source_floor = 2 if final_top_k >= len(sources) * 2 else 1

    for source in sources:
        source_docs = [document for document in ranked if document.source == source]
        for document in source_docs[:per_source_floor]:
            selected.append(document)
            selected_ids.add(document.id)

    for document in ranked:
        if len(selected) >= final_top_k:
            break
        if document.id in selected_ids:
            continue
        selected.append(document)
        selected_ids.add(document.id)

    return selected[:final_top_k]
