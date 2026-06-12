from __future__ import annotations

import json
import os
import re
import unicodedata

from agents.base import Agent
from orchestrator.models import QueryRewrite, QueryUnderstanding

SYSTEM_PROMPT = """You are a Query Rewriting Expert for an Islamic Knowledge RAG over Quran, Hadith, and Fiqh.
Optimize the query for dense retrieval, hybrid/BM25 search, reranking, and metadata filtering.
Return one valid JSON object matching the schema. Do NOT answer the question.

The rewrite must be stable and retrieval-focused. The same input should produce the same intent-preserving
search language.

Required fields:
1. "normalized_query": NFKC-normalized text with cleaned punctuation, normalized Arabic spacing, and stable
   spelling. Keep religious names and references intact.
2. "rewritten_query": a precise retrieval query that fixes spelling and uses canonical Islamic terminology.
3. "expanded_query": recall-oriented query expansion. Include useful synonyms, variant spellings, Arabic/English
   transliterations, source names, and thematic terms. Do not add unrelated concepts.
4. "source_queries": source-specific rewrites keyed by "quran", "hadith", and/or "fiqh" when relevant.
   - quran: include surah name/number, ayah reference, theme words, ayah text words if the user supplied or
     clearly referenced them, and tafsir terms (تفسير، معنى، سبب النزول، روايات) only when useful.
   - hadith: include matn keywords, narrator/source/grade words, "شرح الحديث" when the user wants explanation,
     and variants for common hadith terms.
   - fiqh: include ruling/action/object, madhhab terms if present, topic/category wording, and evidence terms.
5. "search_terms": compact list of the strongest lexical terms that should help BM25/hybrid search.
6. "must_match_terms": only exact terms/references that should remain mandatory if a filter can safely use them
   (e.g. surah number, hadith number, named book, explicit narrator). Leave empty when unsure.
7. "quran_reference_terms": Quran-specific references and lexical anchors: surah name/number, ayah number/range,
   distinctive ayah words supplied by the user, theme, prophet names, sabab al-nuzul words.
8. "hadith_reference_terms": Hadith-specific anchors: matn phrase, narrator, book, muhaddith, grade/authenticity,
   explanation intent.
9. "fiqh_reference_terms": Fiqh-specific anchors: action/object, madhhab, conditions, exceptions, topic labels.
10. "negative_terms": concepts explicitly excluded by the user. Leave empty unless explicit.

Expansion rules:
- For direct ayah queries, include the full Arabic ayah text only if you are certain. Otherwise include distinctive
  words from the user's query and the numeric reference.
- For whole-surah or story queries, expand with the named surah/story, main actors, events, morals, and tafsir
  context words; do not collapse the query to a single ayah.
- For Hadith, never fabricate a matn. If exact text is unknown, expand by concepts, narrator/source/grade, and
  explanation intent.
- For Fiqh, preserve madhhab names and jurisprudential terms exactly.
- For Quran parent-child chunks:
  - direct ayah/reference: keep the exact reference and distinctive ayah words near the front of quran source query.
  - story/whole-surah/theme: include enough theme and actor/event terms to retrieve multiple child chunks.
  - do not over-repeat generic words like "قرآن" and "تفسير" at the expense of specific terms.
- For source_queries, prefer compact but information-rich strings. They are sent directly to the retrievers.

Do not fabricate verses, hadiths, references, or rulings."""

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s:/-]+|[؟،؛«»“”]")
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_QURAN_EXPANSION_TERMS = ("تفسير", "معنى", "آيات", "سورة", "موضوع", "هداية", "دلالة", "سبب النزول")
_HADITH_EXPANSION_TERMS = ("حديث", "رواية", "شرح الحديث", "الراوي", "المصدر", "درجة الحديث", "صحيح")
_FIQH_EXPANSION_TERMS = ("حكم", "فقه", "شروط", "أقوال الفقهاء", "المذاهب", "دليل")
_TOPIC_EXPANSIONS: dict[str, dict[str, tuple[str, ...]]] = {}


class QueryRewriterAgent(Agent):
    async def run(self, query: str, understanding: QueryUnderstanding) -> QueryRewrite:
        if not _use_llm_query_rewrite():
            normalized = _normalize_query(query)
            return _stabilize_rewrite(
                query,
                understanding,
                QueryRewrite(
                    normalized_query=normalized,
                    rewritten_query=normalized,
                    expanded_query=normalized,
                ),
            )
        try:
            rewrite = await self._structured_json(
                SYSTEM_PROMPT,
                json.dumps(
                    {"query": query, "understanding": understanding.model_dump()},
                    ensure_ascii=False,
                ),
                QueryRewrite,
            )
            return _stabilize_rewrite(query, understanding, rewrite)
        except Exception:
            normalized = _normalize_query(query)
            rewrite = QueryRewrite(
                normalized_query=normalized,
                rewritten_query=normalized,
                expanded_query=normalized,
            )
            return _stabilize_rewrite(query, understanding, rewrite)


def _stabilize_rewrite(query: str, understanding: QueryUnderstanding, rewrite: QueryRewrite) -> QueryRewrite:
    normalized = _normalize_query(rewrite.normalized_query or query)
    rewritten = _normalize_query(rewrite.rewritten_query or normalized)
    expanded = _normalize_query(rewrite.expanded_query or rewritten)

    source_queries = dict(rewrite.source_queries or {})
    search_terms = _dedupe([*rewrite.search_terms, *understanding.key_concepts, *_important_terms(normalized)])
    must_match_terms = _dedupe([*rewrite.must_match_terms, *_reference_terms(understanding)])

    quran_terms = _dedupe([*rewrite.quran_reference_terms, *must_match_terms, *_quran_terms(understanding, search_terms)])
    hadith_terms = _dedupe([*rewrite.hadith_reference_terms, *_hadith_terms(understanding, search_terms)])
    fiqh_terms = _dedupe([*rewrite.fiqh_reference_terms, *_fiqh_terms(understanding, search_terms)])

    if understanding.domain in {"quran", "multi"}:
        quran_expansion_terms = (
            ("تفسير", "معنى", "الجلالين")
            if understanding.retrieval_depth == "pinpoint"
            else _QURAN_EXPANSION_TERMS if understanding.evidence_need in {"tafsir_context", "mixed"}
            else ()
        )
        source_queries["quran"] = _merge_query(
            source_queries.get("quran"),
            rewritten,
            expanded,
            quran_terms,
            _topic_terms(normalized, "quran"),
            quran_expansion_terms,
        )
    if understanding.domain in {"hadith", "multi"}:
        source_queries["hadith"] = _merge_query(
            source_queries.get("hadith"),
            rewritten,
            expanded,
            hadith_terms,
            _topic_terms(normalized, "hadith"),
            _HADITH_EXPANSION_TERMS if understanding.wants_explanation or understanding.evidence_need in {"hadith_grade", "mixed"} else (),
        )
    if understanding.domain in {"fiqh", "multi"}:
        source_queries["fiqh"] = _merge_query(
            source_queries.get("fiqh"),
            rewritten,
            expanded,
            fiqh_terms,
            _topic_terms(normalized, "fiqh"),
            _FIQH_EXPANSION_TERMS,
        )

    if not source_queries and understanding.domain in {"quran", "hadith", "fiqh"}:
        source_queries[understanding.domain] = expanded

    expanded = _merge_query(expanded, rewritten, normalized, search_terms, must_match_terms)

    return rewrite.model_copy(
        update={
            "normalized_query": normalized,
            "rewritten_query": rewritten,
            "expanded_query": expanded,
            "source_queries": source_queries,
            "search_terms": search_terms[:20],
            "must_match_terms": must_match_terms[:12],
            "quran_reference_terms": quran_terms[:20],
            "hadith_reference_terms": hadith_terms[:20],
            "fiqh_reference_terms": fiqh_terms[:20],
            "negative_terms": _dedupe(rewrite.negative_terms)[:10],
        }
    )


def _normalize_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = _ARABIC_DIACRITICS.sub("", text)
    text = _PUNCT.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _reference_terms(understanding: QueryUnderstanding) -> list[str]:
    refs = understanding.requested_references or {}
    terms: list[str] = []
    if "surah" in refs:
        terms.extend([f"سورة {refs['surah']}", f"surah {refs['surah']}"])
    if "ayah_from" in refs:
        ayah_to = refs.get("ayah_to", refs["ayah_from"])
        terms.append(f"آية {refs['ayah_from']}" if refs["ayah_from"] == ayah_to else f"آيات {refs['ayah_from']}-{ayah_to}")
    for key in ("book", "hadith_number", "rawi", "madhhab"):
        if refs.get(key):
            terms.append(str(refs[key]))
    return terms


def _quran_terms(understanding: QueryUnderstanding, search_terms: list[str]) -> list[str]:
    entities = understanding.named_entities or {}
    terms = [*search_terms]
    for key in ("prophets", "surahs", "ayahs", "places", "people"):
        terms.extend(entities.get(key, []))
    if understanding.query_scope in {"narrative", "broad_theme", "named_section"}:
        terms.extend(["قصة", "عبرة", "موضوع السورة", "تفسير الآيات"])
    if understanding.tafsir_depth == "detailed":
        terms.extend(["ابن كثير", "روايات التفسير", "السياق"])
    elif understanding.tafsir_depth == "concise":
        terms.extend(["الجلالين", "المعنى المختصر"])
    return terms


def _hadith_terms(understanding: QueryUnderstanding, search_terms: list[str]) -> list[str]:
    entities = understanding.named_entities or {}
    terms = [*search_terms]
    for key in ("hadith_books", "narrators", "people", "acts_of_worship"):
        terms.extend(entities.get(key, []))
    if understanding.wants_explanation:
        terms.extend(["شرح الحديث", "فوائد الحديث", "دلالة الحديث"])
    if understanding.evidence_need == "hadith_grade":
        terms.extend(["صحة الحديث", "درجة الحديث", "المحدث"])
    return terms


def _fiqh_terms(understanding: QueryUnderstanding, search_terms: list[str]) -> list[str]:
    entities = understanding.named_entities or {}
    terms = [*search_terms]
    for key in ("madhhabs", "fiqh_topics", "acts_of_worship", "legal_objects"):
        terms.extend(entities.get(key, []))
    if understanding.query_scope == "ruling":
        terms.extend(["حكم", "شروط", "استثناءات", "أقوال الفقهاء"])
    return terms


def _merge_query(*parts: object) -> str:
    values: list[str] = []
    for part in parts:
        if isinstance(part, str):
            values.append(part)
        elif isinstance(part, (list, tuple)):
            values.extend(str(item) for item in part)
    return " ".join(_dedupe([_normalize_query(value) for value in values])[:40])


def _important_terms(text: str) -> list[str]:
    tokens = re.findall(r"[\w\u0600-\u06FF]{3,}", text)
    stop = {"ما", "من", "في", "عن", "على", "الى", "إلى", "هل", "كيف", "لماذا", "هذا", "هذه", "ذلك", "تلك"}
    return [token for token in tokens if token not in stop][:12]


def _dedupe(values: list[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = _WHITESPACE.sub(" ", str(value or "")).strip()
        if not clean or clean in seen:
            continue
        result.append(clean)
        seen.add(clean)
    return result


def _topic_terms(normalized_query: str, source: str) -> list[str]:
    terms: list[str] = []
    for trigger, by_source in _TOPIC_EXPANSIONS.items():
        if trigger in normalized_query:
            terms.extend(by_source.get(source, []))
    return terms


def _use_llm_query_rewrite() -> bool:
    return os.getenv("YAQEEN_USE_LLM_QUERY_REWRITE", "").strip().lower() in {"1", "true", "yes"}
