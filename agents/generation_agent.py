from __future__ import annotations

import json
import os
import re
from typing import Any

from agents.base import Agent
from orchestrator.models import AggregatedEvidence, GeneratedAnswer, QueryRewrite

BASE_SYSTEM_PROMPT = """You are Yaqeen AI, a highly knowledgeable and grounded Islamic assistant.
Synthesize a faithful, structured answer based strictly on the provided evidence snippets array.
Respond in the language of the original query unless otherwise instructed.

Grounding constraints:
1. Do not invent rulings, hadiths, verses, narrators, grades, or references. If the evidence is insufficient,
   clearly say what is missing.
2. Quranic verses: never truncate or paraphrase ayahs when quoting them. Quote the full ayah text exactly as
   provided. Do not fabricate any ayah text.
3. Thematic Quran summaries are context only. Do not treat a theme summary as a direct ayah quote.
4. Citations: include inline citations using the exact citation_label from evidence items.
5. Never expose internal metadata, retrieval settings, scores, cache state, or routing details.
6. You may use provided compact metadata to understand context, ordering, edition, topic, ayah range, narrator,
   source, grade, and fiqh topic. Do not print metadata as raw JSON.
7. Respect evidence source boundaries:
   - Only write a Quran section or say "الأدلة من القرآن" when at least one evidence item has source="quran".
   - If a hadith or fiqh source quotes a Quranic ayah internally, cite it as that hadith/fiqh evidence item, not
     as Quran evidence.
   - Do not force coverage of Quran, Hadith, and Fiqh. Use only the source types that genuinely support the answer.
   - Exception: if the user explicitly asks for a source type and evidence items from that source type are present,
     you must use and cite at least one relevant item from that source type.

Answer style:
- Choose the answer shape dynamically from the user's question and the retrieved evidence.
- Do not use a fixed template. Do not always start with "الخلاصة" or "Summary".
- Use headings only when they make the answer easier to read. If the answer is simple, write a direct paragraph.
- If several sources are genuinely used, you may group by those used sources only. If only one or two citations are needed, keep it compact.
- The first sentence should answer the user's actual question directly when the evidence supports it.
- Keep the tone natural and scholarly, not mechanical.

Hadith-specific requirements:
- For every Hadith evidence item used, include a compact structured entry:
  - "النص أو معناه": quote the exact hadith text if it is clearly present; otherwise give a cautious meaning summary
    and make clear it is a meaning, not a verbatim quote.
  - "شرح مختصر": summarize the provided explanation when present after "الشرح:". If no explanation is provided,
    derive only a conservative, obvious explanation from the hadith text itself.
  - "الدلالة": explain how this hadith answers the user.
  - End the entry with its citation label.
- If multiple versions of the same hadith are retrieved, collapse duplicates and mention the strongest/clearest
  citation labels together.
- Do not include raw metadata fields or empty metadata objects.

Json Structure: Return a valid JSON object with:
- "answer": the structured text response.
- "citations": an array of citation objects used in your response. Use the exact source and label from evidence.
- "follow_up_questions": 2 to 3 closely related follow-up questions grounded in the retrieved source domains."""

_MAX_TEXT_CHARS = int(os.getenv("YAQEEN_GENERATION_DOC_CHAR_LIMIT", "4200"))
_QURAN_METADATA_KEYS = {
    "surah",
    "surah_name_arabic",
    "surah_name_english",
    "ayah_from",
    "ayah_to",
    "ayah_range",
    "theme",
    "edition",
    "revelation_type",
    "juz",
    "page",
    "is_parent",
    "chunk_id",
    "parent_chunk_id",
    "ayah_text",
}
_HADITH_METADATA_KEYS = {
    "book",
    "masdar",
    "source",
    "numberOrPage",
    "hadith_number",
    "safha_raqam",
    "rawi",
    "mohadeth",
    "grade",
    "ruling",
    "category",
    "subcategory_name",
    "hasExplanation",
    "explanation",
}
_FIQH_METADATA_KEYS = {
    "short_ref",
    "volume_id",
    "book_page",
    "fiqh_topic",
    "mazhabs",
    "chapter",
    "section",
}


class GenerationAgent(Agent):
    async def run(
        self,
        original_query: str,
        rewrite: QueryRewrite,
        evidence: AggregatedEvidence,
    ) -> GeneratedAnswer:
        if not evidence.documents:
            return GeneratedAnswer(
                answer=(
                    "لم أجد في المقاطع المسترجعة دليلاً كافياً للإجابة بثقة. "
                    "جرّب صياغة السؤال بذكر السورة أو الآية أو نص الحديث أو الباب الفقهي المقصود."
                    if _looks_arabic(original_query)
                    else "I could not find enough retrieved evidence to answer confidently. Try adding the surah, ayah, hadith text, or fiqh topic."
                ),
                citations=[],
                follow_up_questions=[],
            )

        evidence_items = []
        for doc in evidence.documents:
            label = doc.citation.label if doc.citation else ""
            metadata = _compact_metadata(str(doc.source), doc.metadata)
            item: dict = {
                "citation_label": label,
                "source": doc.source,
                "metadata": metadata,
                "text": f"{label}\n{_truncate(doc.text, _MAX_TEXT_CHARS)}" if label else _truncate(doc.text, _MAX_TEXT_CHARS),
            }
            if doc.citation:
                item["citation"] = doc.citation.model_dump()
            evidence_items.append(item)

        source_counts = _source_counts(evidence_items)
        requested_sources = _requested_sources(original_query)
        payload = {
            "original_query": original_query,
            "rewritten_query": rewrite.rewritten_query,
            "source_queries": rewrite.source_queries,
            "source_counts": source_counts,
            "requested_sources": requested_sources,
            "answer_format": {
                "adaptive_structure": True,
                "avoid_fixed_template": True,
                "hide_metadata": True,
                "use_metadata_for_context": True,
                "hadith_entries_need_explanation": "hadith" in source_counts,
                "quran_metadata_available": "quran" in source_counts,
                "fiqh_metadata_available": "fiqh" in source_counts,
            },
            "evidence": evidence_items,
        }
        raw_generated = await self.llm.generate_json(
            _build_system_prompt(original_query, evidence_items, requested_sources),
            json.dumps(payload, ensure_ascii=False),
        )
        generated = GeneratedAnswer.model_validate(_coerce_generated_payload(raw_generated, evidence_items))
        return generated.model_copy(update={"answer": _clean_generated_answer(generated.answer, source_counts)})


def _build_system_prompt(original_query: str, evidence_items: list[dict[str, Any]], requested_sources: list[str] | None = None) -> str:
    sources = _source_counts(evidence_items)
    requested_sources = requested_sources or _requested_sources(original_query)
    additions: list[str] = []
    required_available = [source for source in requested_sources if source in sources]
    required_missing = [source for source in requested_sources if source not in sources]
    if required_available:
        additions.append(
            "Coverage requirement: the user explicitly requested these source types and evidence is available for them: "
            + ", ".join(required_available)
            + ". Use and cite at least one relevant evidence item from each of these source types."
        )
    if required_missing:
        additions.append(
            "Missing requested source warning: the user requested these source types, but no evidence items from them are available: "
            + ", ".join(required_missing)
            + ". State this limitation briefly instead of inventing evidence."
        )
    if "quran" in sources:
        additions.append(
            """Quran generation policy:
- Use metadata fields such as surah_name_arabic, ayah_range, theme, edition, and ayah_text to group and order evidence.
- If ayah_text is present, quote it exactly when needed. If text contains tafsir only, label it as tafsir/meaning.
- When multiple chunks share a surah/theme, synthesize them together instead of treating each chunk as unrelated.
- Mention tafsir edition naturally only when it matters (e.g. concise Jalalayn vs detailed Ibn Kathir context)."""
        )
    else:
        additions.append('No evidence item has source="quran"; do not create a Quran evidence section.')
    if "hadith" in sources:
        additions.append(
            """Hadith generation policy:
- Use metadata fields such as book/source, number/page, rawi, mohadeth, grade/ruling, and explanation.
- Prefer stronger/clearer narrations when several hadith chunks overlap.
- If explanation metadata is present, summarize it under a short explanation; do not paste long raw explanation text."""
        )
    if "fiqh" in sources:
        additions.append(
            """Fiqh generation policy:
- Use metadata fields such as fiqh_topic, mazhabs, volume/page, and short_ref to attribute legal discussion.
- Distinguish direct textual evidence from juristic explanation and conditions."""
        )
    if _looks_arabic(original_query):
        additions.append("Use natural Arabic prose. Add Arabic headings only when they help; do not force a fixed heading set.")
    else:
        additions.append("Use natural English prose. Add English headings only when they help; do not force a fixed heading set.")

    return f"{BASE_SYSTEM_PROMPT}\n\nDynamic context for this answer:\n" + "\n\n".join(additions)


def _compact_metadata(source: str, metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "quran": _QURAN_METADATA_KEYS,
        "hadith": _HADITH_METADATA_KEYS,
        "fiqh": _FIQH_METADATA_KEYS,
    }.get(source, set())
    compact: dict[str, Any] = {}
    for key in allowed:
        value = metadata.get(key)
        if value in (None, "", [], {}):
            continue
        if isinstance(value, str):
            compact[key] = _truncate(value, 700)
        else:
            compact[key] = value
    return compact


def _coerce_generated_payload(payload: dict[str, Any], evidence_items: list[dict[str, Any]]) -> dict[str, Any]:
    coerced = dict(payload or {})
    coerced["answer"] = str(coerced.get("answer") or "").strip()

    if not isinstance(coerced.get("follow_up_questions"), list):
        value = coerced.get("follow_up_questions")
        coerced["follow_up_questions"] = [str(value)] if value else []

    evidence_citations_by_label = {
        item.get("citation_label"): item.get("citation")
        for item in evidence_items
        if item.get("citation_label") and isinstance(item.get("citation"), dict)
    }
    citations: list[dict[str, Any]] = []
    for citation in coerced.get("citations") or []:
        if isinstance(citation, dict):
            label = citation.get("label")
            if label in evidence_citations_by_label:
                citations.append(evidence_citations_by_label[label])
            elif citation.get("source") and label:
                citations.append(citation)
        elif isinstance(citation, str):
            matched = evidence_citations_by_label.get(citation)
            if matched:
                citations.append(matched)
    coerced["citations"] = citations
    return coerced


def _source_counts(evidence_items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in evidence_items:
        source = str(item.get("source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return counts


def _truncate(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip() + "..."


def _looks_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06FF" for char in text)


def _requested_sources(query: str) -> list[str]:
    text = (query or "").casefold()
    sources: list[str] = []
    if any(term in text for term in ("قرآن", "القرآن", "آية", "ايات", "آيات", "quran", "ayah", "verse")):
        sources.append("quran")
    if any(term in text for term in ("حديث", "الأحاديث", "احاديث", "السنة", "sunnah", "hadith")):
        sources.append("hadith")
    if any(term in text for term in ("حكم", "فقه", "الفقه", "قول الفقهاء", "أقوال الفقهاء", "ruling", "fiqh")):
        sources.append("fiqh")
    return _dedupe_strings(sources)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        result.append(value)
        seen.add(value)
    return result


def _clean_generated_answer(answer: str, source_counts: dict[str, int]) -> str:
    cleaned = re.sub(r"^\s*(?:الخلاصة|خلاصة|Summary)\s*[:：]\s*", "", answer or "").strip()
    if "quran" not in source_counts:
        cleaned = re.sub(
            r"(?:^|\n\n)الأدلة من القرآن\s*[:：]\s*.*?(?=\n\n(?:الأحاديث|البيان الفقهي|الدليل|الحكم|التفصيل|Hadith|Fiqh)\s*[:：]|\Z)",
            "",
            cleaned,
            flags=re.DOTALL,
        ).strip()
    return cleaned
