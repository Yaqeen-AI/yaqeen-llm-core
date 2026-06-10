from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

from agents.base import Agent
from orchestrator.models import QueryRewrite, QueryUnderstanding, RagSelection, RagSource, RetrievalPlan, SourceRetrievalConfig

SYSTEM_PROMPT = """Create a precise retrieval configuration for each selected source.
Return one valid JSON object matching the requested schema. Place source-specific parameters inside `filters`.

The downstream code will enforce safety bounds, but you should still choose sensible values.

## Quran Retrieval Policy
The Quran index has parent theme chunks and child tafsir chunks:
- Parent chunks summarize a theme and ayah range.
- Child chunks contain actual ayah text and split tafsir text.
Prefer child chunks by retrieval behavior; do NOT set an `is_parent` filter.

Use retrieval depth from query understanding:
- pinpoint: exact ayah/reference/short phrase. Low top_k, concise tafsir usually enough.
- focused: narrow concept or short tafsir. Moderate top_k.
- expanded: named story, named surah, or theme spread over multiple chunks. Higher top_k.
- survey: broad overview, comparison, or cross-source request. Highest recall.

Tafsir edition:
- "ar.jalalayn": concise, good for direct meaning and short explanations.
- "ar.ibnkathir": detailed, good for stories, sabab al-nuzul, reports, context, and broad themes.
- If the user asks to compare tafsirs, use a list: ["ar.jalalayn", "ar.ibnkathir"].

Quran filters you may use only when explicit and safe:
- surah (int or list), juz (int/list), ruku (int/list), hizb_quarter (int/list)
- revelation_type ("Meccan", "Medinan")
- edition (string or list)
- ayah_text (full ayah text or a distinctive phrase from the ayah; do not use for broad thematic words)

## Hadith Retrieval Policy
Prefer authentic/reliable evidence. Use:
- grade, min_grade ("sahih", "hasan"), book, mohadeth, rawi
- category, subcategory_name
- has_explanation only when the user explicitly asks for شرح الحديث and enough recall remains
- prioritize_sahihayn (bool, default true)
- dedup_canonical (bool, default true)

## Fiqh Retrieval Policy
Use filters only when explicit:
- mazhab_filter: list of "جمهور", "حنبلي", "حنفي", "شافعي", "مالكي"
- topic_filter: list of strings, always a list even for one topic.

Do not answer the query. Do not fabricate source references."""

_AYAH_REF_RE = re.compile(r"(?:(?:آية|ايه|ayah|verse)\s*)?\b\d{1,3}\s*[:/]\s*\d{1,3}\b", re.IGNORECASE)
_DETAIL_WORDS = (
    "تفصيل",
    "مفصل",
    "ابن كثير",
    "ibn kathir",
    "روايات",
    "أسباب النزول",
    "اسباب النزول",
    "سبب النزول",
    "قصة",
    "قصه",
    "السياق",
)
_CONCISE_WORDS = ("مختصر", "ببساطة", "معنى", "المعنى", "شرح بسيط", "باختصار", "الجلالين", "jalalayn")
_HADITH_EXPLANATION_WORDS = ("شرح الحديث", "اشرح الحديث", "معنى الحديث", "فوائد الحديث", "شرحه")


class RetrievalConfigurationAgent(Agent):
    async def run(
        self,
        selection: RagSelection,
        rewrite: QueryRewrite,
        understanding: QueryUnderstanding | None = None,
    ) -> RetrievalPlan:
        if not selection.selected_sources or selection.route_type == "out_of_scope":
            return RetrievalPlan()
        if not _use_llm_retrieval_config():
            return _apply_retrieval_policy(RetrievalPlan(), selection, rewrite, understanding)
        try:
            payload_data: dict = {
                "selection": selection.model_dump(),
                "rewrite": rewrite.model_dump(),
            }
            if understanding is not None:
                payload_data["understanding"] = understanding.model_dump()

            payload = await self.llm.generate_json(
                SYSTEM_PROMPT,
                json.dumps(payload_data, ensure_ascii=False),
            )
            # The LLM may return a flat RetrievalPlan or wrap it inside a parent
            # object (e.g. {"retrieval_config": {...}, "selection": {...}}).
            # The key name varies: retrieval_config, retrieval_configs,
            # retrieval_configuration, configs, etc.  Match any key containing
            # "config" that maps to a dict of {source_name: config_dict}.
            _CONFIG_KEYS = ("retrieval_config", "retrieval_configs", "retrieval_configuration", "configs")
            for key in _CONFIG_KEYS:
                if key in payload and isinstance(payload[key], dict):
                    nested = payload[key]
                    if "configs" in nested and isinstance(nested["configs"], dict):
                        payload = nested
                        break
                    sample_key = next(iter(nested), None)
                    if sample_key and sample_key not in ("final_top_k",):
                        payload = {"configs": nested, **{k: v for k, v in payload.items() if k not in ("selection", "rewrite", "understanding", key)}}
                        break
            plan = RetrievalPlan.model_validate(payload)
            plan = _apply_retrieval_policy(plan, selection, rewrite, understanding)
            logger.debug("RetrievalPlan configs: %s", list(plan.configs.keys()))
            return plan
        except Exception:
            logger.exception("Failed to parse retrieval config from LLM; using defaults")
            return _apply_retrieval_policy(RetrievalPlan(), selection, rewrite, understanding)


def _apply_retrieval_policy(
    plan: RetrievalPlan,
    selection: RagSelection,
    rewrite: QueryRewrite,
    understanding: QueryUnderstanding | None,
) -> RetrievalPlan:
    configs: dict[RagSource, SourceRetrievalConfig] = {}
    for source in selection.selected_sources:
        existing = plan.configs.get(source) or SourceRetrievalConfig()
        if source == RagSource.QURAN:
            configs[source] = _quran_config(existing, rewrite, understanding)
        elif source == RagSource.HADITH:
            configs[source] = _hadith_config(existing, rewrite, understanding)
        elif source == RagSource.FIQH:
            configs[source] = _fiqh_config(existing, rewrite, understanding)

    final_top_k = _final_top_k(configs, selection)
    return RetrievalPlan(configs=configs, final_top_k=final_top_k)


def _quran_config(
    existing: SourceRetrievalConfig,
    rewrite: QueryRewrite,
    understanding: QueryUnderstanding | None,
) -> SourceRetrievalConfig:
    depth = _depth(rewrite, understanding)
    filters = _clean_filters(existing.filters, _quran_filter_keys())
    refs = understanding.requested_references if understanding else {}
    for key in ("surah", "juz", "ruku", "hizb_quarter", "revelation_type", "ayah_text"):
        if refs.get(key) is not None and filters.get(key) is None:
            filters[key] = refs[key]
    filters.pop("is_parent", None)
    filters.pop("prefer_child", None)
    filters["include_parent_context"] = depth in {"expanded", "survey"}
    filters["edition"] = _select_quran_edition(filters.get("edition"), rewrite, understanding, depth)

    if depth == "pinpoint":
        top_k, sim_k, rerank_n = 4, 16, 6
    elif depth == "focused":
        top_k, sim_k, rerank_n = 6, 30, 8
    elif depth == "expanded":
        top_k, sim_k, rerank_n = 10, 60, 12
    else:
        top_k, sim_k, rerank_n = 14, 80, 18

    if _has_exact_reference_filter(filters, depth):
        sim_k = max(12, min(sim_k, 24))
        top_k = min(top_k, 6)
        rerank_n = min(rerank_n, 8)

    return SourceRetrievalConfig(
        top_k=_prefer_policy(existing.top_k, top_k, lower_is_safer=depth == "pinpoint"),
        similarity_top_k=max(existing.similarity_top_k, sim_k) if depth in {"expanded", "survey"} else min(max(existing.similarity_top_k, sim_k), 40),
        rerank_top_n=_prefer_policy(existing.rerank_top_n, rerank_n, lower_is_safer=depth == "pinpoint"),
        mode=existing.mode or "hybrid",
        filters=filters,
        skip_rerank=existing.skip_rerank if depth == "pinpoint" else False,
    )


def _hadith_config(
    existing: SourceRetrievalConfig,
    rewrite: QueryRewrite,
    understanding: QueryUnderstanding | None,
) -> SourceRetrievalConfig:
    depth = _depth(rewrite, understanding)
    filters = _clean_filters(existing.filters, _hadith_filter_keys())
    filters.setdefault("prioritize_sahihayn", True)
    filters.setdefault("dedup_canonical", True)
    text = _all_query_text(rewrite)
    if not any(word in text for word in _HADITH_EXPLANATION_WORDS):
        filters.pop("has_explanation", None)

    if depth == "pinpoint":
        top_k, sim_k, rerank_n = 5, 20, 8
    elif depth == "focused":
        top_k, sim_k, rerank_n = 6, 28, 8
    elif depth == "expanded":
        top_k, sim_k, rerank_n = 8, 40, 12
    else:
        top_k, sim_k, rerank_n = 10, 50, 14

    return SourceRetrievalConfig(
        top_k=max(existing.top_k, top_k),
        similarity_top_k=max(existing.similarity_top_k, sim_k),
        rerank_top_n=max(existing.rerank_top_n, rerank_n),
        mode=existing.mode or "hybrid",
        filters=filters,
        skip_rerank=existing.skip_rerank,
    )


def _fiqh_config(
    existing: SourceRetrievalConfig,
    rewrite: QueryRewrite,
    understanding: QueryUnderstanding | None,
) -> SourceRetrievalConfig:
    depth = _depth(rewrite, understanding)
    filters = _clean_filters(existing.filters, _fiqh_filter_keys())
    for key in ("mazhab_filter", "topic_filter"):
        if key in filters and isinstance(filters[key], str):
            filters[key] = [filters[key]]

    if depth in {"pinpoint", "focused"}:
        top_k, sim_k, rerank_n = 6, 24, 8
    elif depth == "expanded":
        top_k, sim_k, rerank_n = 8, 36, 10
    else:
        top_k, sim_k, rerank_n = 10, 50, 14

    return SourceRetrievalConfig(
        top_k=max(existing.top_k, top_k),
        similarity_top_k=max(existing.similarity_top_k, sim_k),
        rerank_top_n=max(existing.rerank_top_n, rerank_n),
        mode=existing.mode or "hybrid",
        filters=filters,
        skip_rerank=existing.skip_rerank,
    )


def _depth(rewrite: QueryRewrite, understanding: QueryUnderstanding | None) -> str:
    text = _all_query_text(rewrite)
    if understanding:
        if understanding.retrieval_depth in {"pinpoint", "focused", "expanded", "survey"}:
            depth = understanding.retrieval_depth
        else:
            depth = "focused"
        if understanding.query_scope in {"narrative", "broad_theme"} and depth in {"pinpoint", "focused"}:
            return "expanded"
        if understanding.query_scope == "comparative" and depth != "survey":
            return "expanded"
        return depth
    if _AYAH_REF_RE.search(text):
        return "pinpoint"
    if any(word in text for word in ("قصة", "قصه", "سورة", "موضوع", "كل", "جميع")):
        return "expanded"
    return "focused"


def _select_quran_edition(
    requested: Any,
    rewrite: QueryRewrite,
    understanding: QueryUnderstanding | None,
    depth: str,
) -> str | list[str]:
    if understanding and understanding.tafsir_depth == "both":
        return ["ar.jalalayn", "ar.ibnkathir"]

    text = _all_query_text(rewrite)
    if isinstance(requested, list):
        valid = [item for item in requested if item in {"ar.jalalayn", "ar.ibnkathir"}]
        if valid:
            return valid
    if requested in {"ar.jalalayn", "ar.ibnkathir"}:
        if requested == "ar.jalalayn" and (depth in {"expanded", "survey"} or any(word in text for word in _DETAIL_WORDS)):
            return "ar.ibnkathir"
        return requested

    if understanding and understanding.tafsir_depth == "concise":
        return "ar.jalalayn"
    if understanding and understanding.tafsir_depth == "detailed":
        return "ar.ibnkathir"
    if any(word in text for word in _DETAIL_WORDS) or depth in {"expanded", "survey"}:
        return "ar.ibnkathir"
    if any(word in text for word in _CONCISE_WORDS) or depth == "pinpoint":
        return "ar.jalalayn"
    return "ar.ibnkathir"


def _final_top_k(configs: dict[RagSource, SourceRetrievalConfig], selection: RagSelection) -> int:
    if not configs:
        return 8
    if selection.route_type == "multi_source":
        return min(18, max(8, sum(min(config.top_k, 6) for config in configs.values())))
    return min(18, max(config.top_k for config in configs.values()))


def _has_exact_reference_filter(filters: dict[str, Any], depth: str) -> bool:
    if "ayah_text" in filters:
        return True
    return depth in {"pinpoint", "focused"} and any(key in filters for key in ("surah", "juz", "ruku", "hizb_quarter"))


def _prefer_policy(current: int, policy: int, *, lower_is_safer: bool = False) -> int:
    return min(current, policy) if lower_is_safer else max(current, policy)


def _all_query_text(rewrite: QueryRewrite) -> str:
    parts = [rewrite.normalized_query, rewrite.rewritten_query, rewrite.expanded_query]
    parts.extend(rewrite.source_queries.values())
    parts.extend(rewrite.search_terms)
    return " ".join(part for part in parts if part).casefold()


def _clean_filters(filters: dict[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in (filters or {}).items():
        if key not in allowed_keys or value in (None, "", [], {}):
            continue
        cleaned[key] = value
    return cleaned


def _quran_filter_keys() -> set[str]:
    return {
        "surah",
        "juz",
        "ruku",
        "hizb_quarter",
        "revelation_type",
        "edition",
        "ayah_text",
        "include_parent_context",
        "is_parent",
        "prefer_child",
    }


def _hadith_filter_keys() -> set[str]:
    return {
        "grade",
        "min_grade",
        "book",
        "mohadeth",
        "rawi",
        "category",
        "subcategory_name",
        "has_explanation",
        "canonical_group_id",
        "prioritize_sahihayn",
        "dedup_canonical",
    }


def _fiqh_filter_keys() -> set[str]:
    return {"mazhab_filter", "topic_filter", "category"}


def _use_llm_retrieval_config() -> bool:
    return os.getenv("YAQEEN_USE_LLM_RETRIEVAL_CONFIG", "").strip().lower() in {"1", "true", "yes"}
