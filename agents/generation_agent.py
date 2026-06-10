from __future__ import annotations

import json
import os
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

Answer structure:
- Use short Arabic section headings when the response is Arabic, or short English headings when the query is English.
- Prefer this shape when relevant:
  1. "الخلاصة" / "Summary": direct answer in 2-4 sentences.
  2. Evidence sections grouped by source: "الأدلة من القرآن", "الأحاديث", "البيان الفقهي".
  3. "وجه الدلالة" / "How the evidence supports the answer": concise synthesis.
- Do not make a decorative or verbose outline for very small answers; keep it readable and structured.

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
- "citations": an array of citation objects used in your response; include source and label only.
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
        payload = {
            "original_query": original_query,
            "rewritten_query": rewrite.rewritten_query,
            "source_queries": rewrite.source_queries,
            "source_counts": source_counts,
            "answer_format": {
                "structured": True,
                "hide_metadata": True,
                "use_metadata_for_context": True,
                "hadith_entries_need_explanation": "hadith" in source_counts,
                "quran_metadata_available": "quran" in source_counts,
                "fiqh_metadata_available": "fiqh" in source_counts,
            },
            "evidence": evidence_items,
        }
        return await self._structured_json(
            _build_system_prompt(original_query, evidence_items),
            json.dumps(payload, ensure_ascii=False),
            GeneratedAnswer,
        )


def _build_system_prompt(original_query: str, evidence_items: list[dict[str, Any]]) -> str:
    sources = _source_counts(evidence_items)
    additions: list[str] = []
    if "quran" in sources:
        additions.append(
            """Quran generation policy:
- Use metadata fields such as surah_name_arabic, ayah_range, theme, edition, and ayah_text to group and order evidence.
- If ayah_text is present, quote it exactly when needed. If text contains tafsir only, label it as tafsir/meaning.
- When multiple chunks share a surah/theme, synthesize them together instead of treating each chunk as unrelated.
- Mention tafsir edition naturally only when it matters (e.g. concise Jalalayn vs detailed Ibn Kathir context)."""
        )
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
        additions.append("Use Arabic headings and concise Arabic prose. Avoid English unless the evidence label is English.")
    else:
        additions.append("Use English headings and concise English prose unless the evidence itself must be quoted in Arabic.")

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
