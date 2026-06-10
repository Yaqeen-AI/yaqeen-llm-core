from __future__ import annotations

import json
import os

from agents.base import Agent
from orchestrator.models import QueryRewrite, QueryUnderstanding, RagSelection, RagSource

SYSTEM_PROMPT = """You are a RAG Router for the Yaqeen AI Islamic Assistant. 
Determine the required knowledge sources for the user query depending on domain characteristics.
Must return a valid JSON object.

Output schema elements:
- "selected_sources": list containing any combination of ["quran", "hadith", "fiqh"]. If off-topic, return an empty list.
- "route_type": "single_source" if 1 source, "multi_source" if multiple, or "out_of_scope".
- "confidence": "high", "medium", or "low".
- "reason": A brief 1-sentence explanation of why these sources were selected.

Guidelines:
- If asking for verses / tafsir -> "quran"
- If asking for authentic narrations -> "hadith"
- If asking for rulings (halal/haram/worship) -> "fiqh"
- If cross-domain (e.g. Ruling with Quranic evidence) -> multiple sources.
Never answer the query itself."""


class RagSelectorAgent(Agent):
    async def run(self, query: str, understanding: QueryUnderstanding, rewrite: QueryRewrite) -> RagSelection:
        deterministic = _deterministic_selection(query, understanding, rewrite)
        if deterministic.confidence in {"high", "medium"} or not _use_llm_router():
            return deterministic

        try:
            return await self._structured_json(
                SYSTEM_PROMPT,
                json.dumps(
                    {
                        "query": query,
                        "understanding": understanding.model_dump(),
                        "rewrite": rewrite.model_dump(),
                    },
                    ensure_ascii=False,
                ),
                RagSelection,
            )
        except Exception:
            return deterministic


def _deterministic_selection(query: str, understanding: QueryUnderstanding, rewrite: QueryRewrite) -> RagSelection:
    sources: list[RagSource] = []
    domain_map = {
        "quran": [RagSource.QURAN],
        "hadith": [RagSource.HADITH],
        "fiqh": [RagSource.FIQH],
        "multi": [],
    }
    sources.extend(domain_map.get(understanding.domain, []))

    text = " ".join(
        [
            query,
            understanding.intent,
            understanding.evidence_need,
            rewrite.normalized_query,
            rewrite.rewritten_query,
            rewrite.expanded_query,
        ]
    ).casefold()
    if understanding.domain == "multi":
        if "mixed" == understanding.evidence_need:
            sources.extend([RagSource.QURAN, RagSource.HADITH])
        if any(word in text for word in ("قرآن", "آية", "ايات", "آيات", "تفسير", "quran", "ayah")):
            sources.append(RagSource.QURAN)
        if any(word in text for word in ("حديث", "أحاديث", "احاديث", "رواه", "hadith")):
            sources.append(RagSource.HADITH)
        if any(word in text for word in ("حكم", "فقه", "يجوز", "حرام", "حلال", "fiqh", "ruling")):
            sources.append(RagSource.FIQH)

    if not sources and understanding.domain == "out_of_scope":
        return RagSelection(selected_sources=[], route_type="out_of_scope", confidence="high", reason="Query is outside configured Islamic sources.")
    if not sources:
        sources = [RagSource.QURAN, RagSource.HADITH, RagSource.FIQH]

    deduped = _dedupe_sources(sources)
    return RagSelection(
        selected_sources=deduped,
        route_type="single_source" if len(deduped) == 1 else "multi_source",
        confidence="high" if understanding.confidence >= 0.7 and understanding.domain != "unknown" else "medium",
        reason="Deterministic route from query understanding and source-specific rewrite.",
    )


def _dedupe_sources(sources: list[RagSource]) -> list[RagSource]:
    seen: set[RagSource] = set()
    result: list[RagSource] = []
    for source in sources:
        if source in seen:
            continue
        result.append(source)
        seen.add(source)
    return result


def _use_llm_router() -> bool:
    return os.getenv("YAQEEN_USE_LLM_ROUTER", "").strip().lower() in {"1", "true", "yes"}
