from __future__ import annotations

from orchestrator.models import Citation, RagSource, RetrievedDocument


class CitationAgent:
    async def run(self, documents: list[RetrievedDocument]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[str] = set()
        for document in documents:
            citation = document.citation or _fallback_citation(document)
            key = f"{citation.source}:{citation.label}"
            if key in seen:
                continue
            citations.append(citation)
            seen.add(key)
        return citations


def _fallback_citation(document: RetrievedDocument) -> Citation:
    labels = {
        RagSource.QURAN: "Quran",
        RagSource.HADITH: "Hadith",
        RagSource.FIQH: "Fiqh",
    }
    return Citation(source=document.source, label=f"[{labels[document.source]}]", metadata=document.metadata)

