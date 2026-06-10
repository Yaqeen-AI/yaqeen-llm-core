from __future__ import annotations

import json
import re

from agents.base import Agent
from orchestrator.models import QueryUnderstanding

SYSTEM_PROMPT = """You are an expert Islamic query analyst for a Quran/Hadith/Fiqh RAG system.
Analyze the user's question as a retrieval problem. Return one valid JSON object matching the schema.
Do NOT answer the question.

Your output will directly control retrieval depth, query rewriting, metadata filters, and final answer style.
Be precise, conservative, and repeatable. The same query should produce the same analysis.

Core fields:
1. "language": "ar", "en", "mixed", or "unknown".
2. "intent": concise label such as "tafsir", "hadith_explanation", "hadith_lookup", "fiqh_ruling",
   "evidence_request", "story_or_theme", "comparison", "metadata_lookup", or "general".
3. "domain": one of ["quran", "hadith", "fiqh", "multi", "out_of_scope", "unknown"].
4. "specificity": "specific" only for a single direct reference, a narrow named item, or a constrained
   lookup. Use "general" for thematic, narrative, survey, ruling, or cross-document questions.
5. "query_scope": choose exactly one:
   - "single_reference": one ayah, one hadith, one known reference, or a very narrow phrase.
   - "range_reference": explicit ayah/page/range request.
   - "named_section": a named surah, book, chapter, narrator, madhhab, or source.
   - "narrative": a story or lifecycle spanning many passages, such as a prophet's story.
   - "broad_theme": thematic or survey query spanning many chunks.
   - "comparative": asks to compare views/sources.
   - "ruling": asks for a legal ruling or worship practice.
   - "metadata_lookup": asks about narrator, source, authenticity, page, book, or count.
   - "unknown": none of the above is clear.
6. "retrieval_depth":
   - "pinpoint": exact ayah/hadith/reference lookup, small top-k.
   - "focused": narrow concept or short answer, moderate top-k.
   - "expanded": named story/section/theme needing several chunks.
   - "survey": broad overview, comparison, or multi-source ruling needing high recall.
7. "tafsir_depth": for Quran questions:
   - "concise": short/simple tafsir is enough.
   - "detailed": needs detailed tafsir, reports, reasons, context, or story.
   - "both": explicitly asks for comparison between tafsirs.
   - "auto": not a Quran tafsir question or unclear.
8. "requested_references": object with any explicit references found. Examples:
   {"surah": 12}, {"surah": 2, "ayah_from": 255, "ayah_to": 255},
   {"book": "صحيح مسلم", "hadith_number": "16"}, {"madhhab": "حنفي"}.
9. "named_entities": object of lists. Use keys like:
   - "prophets", "surahs", "ayahs", "hadith_books", "narrators", "madhhabs", "fiqh_topics",
     "places", "people", "acts_of_worship", "legal_objects".
10. "key_concepts": 3-10 compact concepts that retrieval must cover.
11. "evidence_need": one of:
   - "direct_text": user needs exact Quran/hadith text.
   - "tafsir_context": user needs tafsir, theme, sabab al-nuzul, story, or lessons.
   - "hadith_grade": user asks authenticity/source/narrator/grade.
   - "fiqh_positions": user asks ruling, madhhab, conditions, exceptions.
   - "mixed": multiple evidence types are required.
   - "unknown": unclear.
12. "answer_style": one of "direct", "structured", "comparative", "step_by_step", "summary", "unknown".
13. "retrieval_notes": short operational notes for retrieval, such as:
   - "avoid single-ayah narrowing; topic spans many child chunks"
   - "use concise tafsir unless user asks for detailed reports"
   - "prefer hadiths with explanations"
   Keep this as short strings, not prose paragraphs.
14. "wants_explanation": true when the user asks for شرح, تفسير, meaning, lesson, wisdom, or explanation.
15. "wants_summary": true when the user asks for summary, overview, key points, or قصة.
16. "ambiguity_detected": true if important routing/reference details are missing or ambiguous.
17. "confidence": float from 0.0 to 1.0.

Important Quran chunking context:
- Quran retrieval has parent theme chunks and child tafsir chunks. Direct ayah questions need pinpoint/focused
  child chunks. Story or whole-surah questions need expanded/survey depth because relevant material is spread
  across many child chunks under multiple parent themes.
- Jalalayn is concise. Ibn Kathir is longer and often split across several children; choose detailed depth for
  narratives, sabab al-nuzul, reports, and broad tafsir requests.
- Do not classify a prophet story, whole-surah question, or broad moral theme as "specific" merely because it
  names one prophet/surah/person. Use "general" + "expanded" unless there is an explicit single ayah/hadith reference.

Source routing guidance:
- Quran-only: ayah text, tafsir, surah themes, Quranic stories, Quranic ethics, Quranic wording.
- Hadith-only: prophetic narrations, authenticity, narrator/source, hadith explanation.
- Fiqh-only: practical legal ruling, conditions, invalidators, madhhab positions.
- Multi: the user explicitly asks for evidence from multiple source types, asks for ruling with Quran/Hadith proof,
  or asks for a full Islamic answer requiring both textual evidence and legal explanation."""

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_PUNCT = re.compile(r"[^\w\s:/-]+|[؟،؛«»“”]")
_AYAH_REF_RE = re.compile(r"\b(?P<surah>\d{1,3})\s*[:/]\s*(?P<ayah>\d{1,3})(?:\s*[-–]\s*(?P<ayah_to>\d{1,3}))?\b")
_QURAN_WORDS = (
    "قرآن", "القرآن", "آية", "ايه", "آيات", "ايات", "سورة", "سوره", "تفسير", "تدبر",
    "سبب النزول", "أسباب النزول", "اسباب النزول", "مصحف",
)
_HADITH_WORDS = (
    "حديث", "أحاديث", "احاديث", "رواه", "رواية", "روايه", "صحيح", "ضعيف", "حسن",
    "البخاري", "مسلم", "الترمذي", "أبو داود", "ابو داود", "النسائي", "ابن ماجه", "الراوي",
)
_FIQH_WORDS = (
    "حكم", "يجوز", "لا يجوز", "حرام", "حلال", "واجب", "مستحب", "مكروه", "يبطل",
    "شروط", "أركان", "اركان", "زكاة", "صلاة", "صيام", "حج", "وضوء", "طلاق", "بيع",
)
_EXPLANATION_WORDS = ("اشرح", "شرح", "فسر", "تفسير", "معنى", "المعنى", "الدلالة", "الحكمة", "فوائد")
_SUMMARY_WORDS = ("لخص", "ملخص", "خلاصة", "باختصار", "نظرة عامة", "قصة", "قصه", "overview", "summary")
_STORY_WORDS = ("قصة", "قصه", "story", "حكاية", "رحلة", "سيرة")
_COMPARISON_WORDS = ("قارن", "مقارنة", "الفرق بين", "compare", "difference")


class QueryUnderstandingAgent(Agent):
    async def run(self, query: str) -> QueryUnderstanding:
        try:
            payload = await self.llm.generate_json(SYSTEM_PROMPT, json.dumps({"query": query}, ensure_ascii=False))
            understanding = QueryUnderstanding.model_validate(_coerce_understanding_payload(payload))
        except Exception:
            understanding = QueryUnderstanding(language="ar" if _looks_arabic(query) else "unknown", confidence=0.0)
        return _stabilize_understanding(query, understanding)


def _stabilize_understanding(query: str, understanding: QueryUnderstanding) -> QueryUnderstanding:
    normalized = query.translate(_ARABIC_DIGITS).casefold()
    refs = dict(understanding.requested_references or {})
    notes = list(understanding.retrieval_notes or [])
    entities = {key: list(value) for key, value in (understanding.named_entities or {}).items()}

    ayah_match = _AYAH_REF_RE.search(normalized)
    has_quran = _contains_any(normalized, _QURAN_WORDS) or bool(ayah_match)
    has_hadith = _contains_any(normalized, _HADITH_WORDS)
    has_fiqh = _contains_any(normalized, _FIQH_WORDS)
    wants_explanation = understanding.wants_explanation or _contains_any(normalized, _EXPLANATION_WORDS)
    wants_summary = understanding.wants_summary or _contains_any(normalized, _SUMMARY_WORDS)
    is_story = _contains_any(normalized, _STORY_WORDS)
    is_comparative = understanding.query_scope == "comparative" or _contains_any(normalized, _COMPARISON_WORDS)

    if ayah_match:
        refs.setdefault("surah", int(ayah_match.group("surah")))
        refs.setdefault("ayah_from", int(ayah_match.group("ayah")))
        refs.setdefault("ayah_to", int(ayah_match.group("ayah_to") or ayah_match.group("ayah")))
        entities.setdefault("ayahs", []).append(f"{refs['surah']}:{refs['ayah_from']}")

    domain = understanding.domain
    detected_sources = sum(bool(value) for value in (has_quran, has_hadith, has_fiqh))
    if detected_sources > 1:
        domain = "multi"
    elif has_quran:
        domain = "quran"
    elif has_hadith:
        domain = "hadith"
    elif has_fiqh:
        domain = "fiqh"

    query_scope = understanding.query_scope
    retrieval_depth = understanding.retrieval_depth
    specificity = understanding.specificity
    tafsir_depth = understanding.tafsir_depth
    evidence_need = understanding.evidence_need
    answer_style = understanding.answer_style

    if ayah_match:
        query_scope = "single_reference" if refs.get("ayah_from") == refs.get("ayah_to") else "range_reference"
        retrieval_depth = "pinpoint"
        specificity = "specific"
        evidence_need = "direct_text" if not wants_explanation else "tafsir_context"
        tafsir_depth = "concise" if tafsir_depth == "auto" else tafsir_depth
    elif has_quran and (is_story or query_scope in {"narrative", "broad_theme", "named_section"}):
        query_scope = "narrative" if is_story else query_scope
        retrieval_depth = "expanded" if retrieval_depth in {"pinpoint", "focused"} else retrieval_depth
        specificity = "general"
        evidence_need = "tafsir_context"
        tafsir_depth = "detailed" if tafsir_depth in {"auto", "concise"} else tafsir_depth
        notes.append("Quran topic may span multiple child tafsir chunks; avoid over-narrowing to one ayah.")

    if has_fiqh and understanding.query_scope == "unknown":
        query_scope = "ruling"
        evidence_need = "fiqh_positions"
    if has_hadith and evidence_need == "unknown":
        evidence_need = "hadith_grade" if any(word in normalized for word in ("صحة", "صحه", "درجة", "حكم الحديث")) else "direct_text"
    if domain == "multi":
        evidence_need = "mixed"

    if is_comparative:
        query_scope = "comparative"
        retrieval_depth = "survey" if retrieval_depth != "pinpoint" else retrieval_depth
        answer_style = "comparative"
    elif wants_summary:
        answer_style = "summary"
    elif wants_explanation or domain in {"multi", "fiqh"}:
        answer_style = "structured"
    elif answer_style == "unknown":
        answer_style = "direct"

    key_concepts = _dedupe([*understanding.key_concepts, *_important_terms(normalized)])
    confidence = max(understanding.confidence, 0.72 if domain != "unknown" else understanding.confidence)

    return understanding.model_copy(
        update={
            "domain": domain,
            "specificity": specificity,
            "query_scope": query_scope,
            "retrieval_depth": retrieval_depth,
            "tafsir_depth": tafsir_depth,
            "requested_references": refs,
            "named_entities": entities,
            "key_concepts": key_concepts[:12],
            "evidence_need": evidence_need,
            "answer_style": answer_style,
            "retrieval_notes": _dedupe(notes)[:6],
            "wants_explanation": wants_explanation,
            "wants_summary": wants_summary,
            "confidence": min(confidence, 1.0),
        }
    )


def _coerce_understanding_payload(payload: dict) -> dict:
    coerced = dict(payload or {})
    if isinstance(coerced.get("retrieval_notes"), str):
        coerced["retrieval_notes"] = [coerced["retrieval_notes"]]
    elif coerced.get("retrieval_notes") is None:
        coerced["retrieval_notes"] = []

    if not isinstance(coerced.get("key_concepts"), list):
        value = coerced.get("key_concepts")
        coerced["key_concepts"] = [str(value)] if value else []

    entities = coerced.get("named_entities")
    if not isinstance(entities, dict):
        coerced["named_entities"] = {}
    else:
        coerced["named_entities"] = {
            str(key): value if isinstance(value, list) else [str(value)]
            for key, value in entities.items()
            if value not in (None, "", [], {})
        }

    if not isinstance(coerced.get("requested_references"), dict):
        coerced["requested_references"] = {}
    return coerced


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle.casefold() in text for needle in needles)


def _important_terms(text: str) -> list[str]:
    text = _PUNCT.sub(" ", text)
    tokens = re.findall(r"[\w\u0600-\u06FF]{3,}", text)
    stop = {"ما", "من", "في", "عن", "على", "الى", "إلى", "هل", "كيف", "لماذا", "الذي", "التي", "هذا", "هذه"}
    return [token for token in tokens if token not in stop][:10]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean or clean in seen:
            continue
        result.append(clean)
        seen.add(clean)
    return result


def _looks_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06FF" for char in text)
