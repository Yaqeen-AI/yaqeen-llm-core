# ============================================================
# YaqeenAI — LLM-Driven Query Expander for Arabic/Islamic Hadith Search
# ============================================================
#
# Architecture: Single LLM call → structured JSON expansion
#
#   The LLM (Gemini / gemma-3-27b-it) is prompted to act as an
#   expert in Arabic linguistics and Islamic hadith sciences. Given a
#   user query it returns a JSON object with:
#
#     • expanded_terms  — morphological variants, synonyms, narrower
#                         Islamic concepts (Arabic surface forms that
#                         appear in the hadith corpus)
#     • reformulations  — 1-3 full alternative query strings that
#                         match the hadith-corpus linguistic register
#     • dense_query     — a single rich query string for the dense
#                         (Jina) retriever; typically original + the
#                         most relevant hadith-corpus collocation
#     • sparse_query    — token-enriched query for TF-IDF char n-gram
#
#   Fallback:
#     If the LLM call fails (timeout, API error, malformed JSON) the
#     module falls back to a zero-latency "identity" expansion that
#     returns the original query unchanged so the pipeline never blocks.
#
#   Caching:
#     Results are cached in an in-process LRU dict (2 000 entries).
#     Identical queries skip the LLM call entirely.
#
#   Token budget:
#     The expansion prompt is intentionally tiny (<200 tokens in,
#     <150 tokens out) so it adds ≈ 200-400 ms at query time on Gemini
#     free tier, well within acceptable latency for a RAG pipeline.
#
# ============================================================

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

# from groq import Groq
from google import genai
from google.genai import types

from pipeline.config import settings

logger = logging.getLogger(__name__)

# ============================================================
# Output Dataclass  (same interface as the old hardcoded expander)
# ============================================================


@dataclass
class ExpandedQuery:
    """Result of query expansion."""

    original: str
    expanded_terms: list[str] = field(
        default_factory=list
    )  # Extra tokens for sparse index
    reformulations: list[str] = field(
        default_factory=list
    )  # Full alternative query strings
    dense_query: str = ""  # Best query for dense retrieval
    sparse_query: str = ""  # Enriched query for TF-IDF
    multi_queries: list[str] = field(
        default_factory=list
    )  # All variants for multi-query fusion


# ============================================================
# LLM Expansion Prompt
# ============================================================

_SYSTEM_PROMPT = """\
أنت خبير في علوم الحديث النبوي واللغة العربية.
مهمتك توسيع استعلامات البحث في قاعدة بيانات الأحاديث النبوية.

**أولاً — تحليل النية (داخلي، لا يُدرج في الإخراج):**
قبل توليد أي حقل، حلّل الاستعلام على ثلاثة مستويات:
1. **الموضوع العام**: ما الموضوع الكبير (صلاة، صدقة، صبر، جهاد...؟)
2. **القيد أو الشرط أو الحالة**: هل يوجد تخصيص مثل:
   - حالة العجز أو عدم الاستطاعة ("في حالة عدم المقدرة"، "من لا يستطيع"، "العاجز عن...")
   - ظرف زمني أو مكاني ("في السفر"، "في رمضان"، "في المسجد")
   - حالة جسدية ("للمريض"، "الجنب"، "الحائض")
   - حالة نفسية أو اجتماعية ("عند الغضب"، "بين الزوجين"، "مع الكفار")
   - فئة معينة ("للأطفال"، "للمرأة"، "للحاكم")
3. **النية الحقيقية**: ماذا يبحث المستخدم بالضبط؟

**قاعدة التركيز الحاسمة — اتبعها بدقة:**
- إذا وجد **قيد أو شرط أو حالة** في الاستعلام:
  * يجب أن تكون **غالبية** expanded_terms متعلقة بهذا القيد تحديداً كما تظهر في متون الحديث.
  * يجب أن يعكس dense_query النية الحقيقية المقيّدة، لا الموضوع العام.
  * يجب أن تعالج reformulations الحالة أو القيد المحدد.
  * **تجنّب** المصطلحات العامة التي تنطبق على آلاف الأحاديث في الموضوع دون صلة بالقيد.

أمثلة تطبيقية على قاعدة التركيز:
• "الصدقة في حالة عدم المقدرة" →
  الخطأ: ["فضل الصدقة"، "قبول الصدقة"، "الزكاة"]
  الصواب: ["من لم يجد ما يتصدق به"، "فليمسك عن الشر فإنها صدقة"، "كل معروف صدقة"، "الكلمة الطيبة صدقة"، "إماطة الأذى صدقة"، "التسبيح صدقة"، "صدقة البدن"]
• "الصلاة للمريض" →
  الخطأ: ["فضل الصلاة"، "أوقات الصلاة"]
  الصواب: ["يصلي قاعداً"، "يصلي مضطجعاً"، "صلاة المريض"، "إن لم يستطع فبجنبه"، "فإن لم يستطع فبإيماء"]
• "الصيام في السفر" →
  الخطأ: ["فضل الصيام"، "صيام رمضان"]
  الصواب: ["أفطر في السفر"، "المسافر يفطر"، "ليس من البر الصيام في السفر"، "رخصة المسافر"]

أعد **فقط** كائن JSON صحيح — لا نص قبله ولا بعده — بهذه الحقول الأربعة:

{
  "expanded_terms": [
    /* 5–15 مصطلحاً عربياً خالصاً يعكس النية الحقيقية للاستعلام:
       - إذا وجد قيد: الأولوية للمصطلحات المتعلقة بالقيد كما تظهر حرفياً في متون الحديث.
       - تصريفات صرفية، مرادفات، عبارات من البخاري ومسلم والترمذي وغيرها.
       - يُمنع منعاً باتاً إدراج أي كلمة أو حرف بالإنجليزية أو غير العربية. */
  ],
  "reformulations": [
    /* 1–3 صياغات بديلة للاستعلام كاملة بالعربية، بأسلوب متون الحديث،
       تعكس القيد أو الحالة في الاستعلام الأصلي إن وجد */
  ],
  "dense_query": "صياغة مركّزة على النية الحقيقية (مع القيد إن وجد) + أوثق عبارة حديثية مطابقة بالعربية فقط",
  "sparse_query": "الاستعلام الأصلي + أهم 10 مصطلحات من expanded_terms مفصولة بمسافة"
}

قواعد صارمة:
1. كل القيم يجب أن تكون **بالعربية حصراً** — لا لاتينية، لا أرقام إنجليزية، لا رموز.
2. expanded_terms يجب أن تظهر فعلاً في كتب الحديث النبوي.
3. لا تضف أي شرح أو نص خارج كائن JSON.
4. إذا كان الاستعلام تحيةً أو خارج نطاق الحديث أعد:
   {"expanded_terms":[],"reformulations":[],"dense_query":"","sparse_query":""}
5. **للاستعلامات المقيّدة**: expanded_terms وdense_query يجب أن يعكسا الجانب المميز
   للاستعلام (القيد/الحالة/الشرط) وليس الموضوع العام وحده.
"""

_USER_PROMPT_TEMPLATE = """\
وسّع الاستعلام التالي مع التركيز على النية الحقيقية وأي قيود أو حالات أو شروط فيه:

"{query}"

تذكّر: إذا تضمّن الاستعلام قيداً محدداً (مثل "في حالة عدم المقدرة"، "في السفر"، "للمريض"، "عند الغضب"...)،
فيجب أن تعكس expanded_terms ذلك القيد بعبارات حديثية حقيقية، لا الموضوع العام وحده.\
"""


# ============================================================
# LRU Cache (thread-safe)
# ============================================================

_cache_lock = threading.Lock()
_expansion_cache: dict[str, ExpandedQuery] = {}
_CACHE_MAX_SIZE = 2_000


def _cache_get(key: str) -> Optional[ExpandedQuery]:
    with _cache_lock:
        return _expansion_cache.get(key)


def _cache_set(key: str, value: ExpandedQuery) -> None:
    with _cache_lock:
        if len(_expansion_cache) >= _CACHE_MAX_SIZE:
            # Evict oldest 10 %
            drop = list(_expansion_cache.keys())[: _CACHE_MAX_SIZE // 10]
            for k in drop:
                del _expansion_cache[k]
        _expansion_cache[key] = value


# ============================================================
# Gemini client (lazy singleton)
# ============================================================

_gemini_client: Optional[genai.Client] = None
_gemini_lock = threading.Lock()
# _groq_client: Optional[Groq] = None
# _groq_lock = threading.Lock()


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        with _gemini_lock:
            if _gemini_client is None:
                _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


# def _get_groq() -> Groq:
#     global _groq_client
#     if _groq_client is None:
#         with _groq_lock:
#             if _groq_client is None:
#                 _groq_client = Groq(api_key=settings.GROQ_API_KEY)
#     return _groq_client


# ============================================================
# JSON extraction helper
# ============================================================

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
# Arabic Unicode blocks: basic Arabic, supplement, presentation forms A & B
_ARABIC_CHARS = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response string."""
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group())


def _is_arabic_token(token: str) -> bool:
    """Return True only if the token contains at least one Arabic character."""
    return bool(_ARABIC_CHARS.search(token))


# ============================================================
# Identity fallback
# ============================================================


def _identity_expansion(text: str) -> ExpandedQuery:
    """Return a no-op expansion used when the LLM call fails."""
    return ExpandedQuery(
        original=text,
        expanded_terms=[],
        reformulations=[],
        dense_query=text,
        sparse_query=text,
        multi_queries=[text],
    )


# ============================================================
# Core LLM call
# ============================================================

_SKIP_TYPES = frozenset({"metadata", "greeting", "out_of_scope", "dataset_stats"})

# ============================================================
# Intent-specific hard-coded corpus phrase clusters
# ============================================================
# Injected when a specific query_intent is detected, these are
# exact phrases that appear in hadith matn — highest-precision signal.
# The LLM expansion may miss them when it latches onto surface keywords
# (e.g. "الإسلام") rather than the assertive/exclusivity intent.

_INTENT_EXPANSION_TERMS: dict[str, list[str]] = {
    "truth_claim": [
        "يعلو ولا يعلى عليه",
        "بعثت إلى الناس كافة",
        "لا يسمع بي أحد من هذه الأمة يهودي ولا نصراني",
        "ثم يموت ولا يؤمن بالذي أرسلت به إلا كان من أصحاب النار",
        "الحنيفية السمحة",
        "فطرة الله التي فطر الناس عليها",
        "لا نبي بعدي",
        "خاتم النبيين",
        "الإسلام يعلو",
        "بعثت بالحق",
        "أرسلت إلى الأحمر والأسود",
        "جاء بالدين الحق",
    ],
}

# Intent-specific note appended to the LLM prompt to steer expansion
_INTENT_PROMPT_NOTES: dict[str, str] = {
    "truth_claim": (
        "\n\n⚠️ ملاحظة حاسمة للتوسيع: هذا الاستعلام يبحث عن **أدلة حديثية على حقانية وعالمية الإسلام**.\n"
        "يجب أن تركّز expanded_terms على عبارات الأحاديث التي تثبت ذلك صراحةً:\n"
        "• سمو الإسلام: «يعلو ولا يعلى عليه»\n"
        "• عالمية الرسالة: «بعثت إلى الناس كافة» / «أرسلت إلى الأحمر والأسود»\n"
        "• وجوب الإيمان: «لا يسمع بي أحد يهودي ولا نصراني ثم يموت ولا يؤمن إلا كان من أصحاب النار»\n"
        "• الفطرة: «فطرة الله التي فطر الناس عليها»\n"
        "• خاتم الأنبياء: «لا نبي بعدي» / «خاتم النبيين»\n"
        "• الحنيفية: «بعثت بالحنيفية السمحة»\n"
        "تجنّب تماماً: أركان الإسلام الخمسة، وصف الصلاة والزكاة، والنصيحة لأئمة المسلمين — "
        "فهذه تصف بنية الإسلام ولا تثبت حقانيته.\n"
    ),
}


def _llm_expand(query: str, extra_note: str = "") -> dict:
    """
    Ask the LLM to expand the query.
    Returns the parsed JSON dict.
    Raises on any failure so the caller can fall back.
    """
    client = _get_gemini()
    # gemma-3-* does not support system_instruction, so merge it into contents.
    merged_prompt = (
        f"تعليمات النظام:\n{_SYSTEM_PROMPT}\n\n"
        f"طلب المستخدم:\n{_USER_PROMPT_TEMPLATE.format(query=query)}"
        + (f"\n{extra_note}" if extra_note else "")
    )

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=merged_prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,  # low temperature -> consistent, focused output
            max_output_tokens=512,  # richer conditional expansions need more tokens
        ),
    )
    # client = _get_groq()
    # response = client.chat.completions.create(
    #     model=settings.GROQ_MODEL,
    #     messages=[
    #         {"role": "system", "content": _SYSTEM_PROMPT},
    #         {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(query=query)},
    #     ],
    #     temperature=0.2,       # low temperature → consistent, focused output
    #     max_tokens=256,        # expansion JSON is small
    #     timeout=8,             # fail fast — don't block retrieval
    # )
    raw = response.text or ""
    # raw = response.choices[0].message.content or ""
    logger.debug("LLM expansion raw response: %s", raw[:400])
    return _extract_json(raw)


# ============================================================
# Validation & assembly
# ============================================================


def _safe_str_list(val, max_items: int = 20, arabic_only: bool = False) -> list[str]:
    """
    Coerce a value to a list of non-empty strings, capped at max_items.
    If arabic_only=True, silently drops any item that contains no Arabic character.
    """
    if not isinstance(val, list):
        return []
    result = []
    for item in val:
        if not isinstance(item, str):
            continue
        token = item.strip()
        if not token:
            continue
        if arabic_only and not _is_arabic_token(token):
            logger.debug("Dropping non-Arabic expansion token: %r", token)
            continue
        result.append(token)
        if len(result) >= max_items:
            break
    return result


def _assemble(
    original: str, data: dict, max_expansion_tokens: int = 40
) -> ExpandedQuery:
    """
    Turn the raw LLM JSON into a validated ExpandedQuery.
    All string fields are sanitised; missing fields use safe defaults.
    """
    expanded_terms = _safe_str_list(
        data.get("expanded_terms", []), max_expansion_tokens, arabic_only=True
    )
    reformulations = _safe_str_list(data.get("reformulations", []), 4, arabic_only=True)

    raw_dense = data.get("dense_query", "")
    dense_query = (
        raw_dense.strip()
        if isinstance(raw_dense, str) and raw_dense.strip()
        else original
    )

    raw_sparse = data.get("sparse_query", "")
    if isinstance(raw_sparse, str) and raw_sparse.strip():
        sparse_query = raw_sparse.strip()
    else:
        # Build from original + top expansion tokens
        extras = " ".join(expanded_terms[:max_expansion_tokens])
        sparse_query = (original + " " + extras).strip() if extras else original

    # multi_queries: original first, then reformulations
    multi_queries = [original]
    for ref in reformulations:
        if ref not in multi_queries and len(ref) > 3:
            multi_queries.append(ref)
        if len(multi_queries) >= 4:
            break

    return ExpandedQuery(
        original=original,
        expanded_terms=expanded_terms,
        reformulations=reformulations,
        dense_query=dense_query,
        sparse_query=sparse_query,
        multi_queries=multi_queries,
    )


# ============================================================
# Public API  (drop-in replacement for old expand_query)
# ============================================================


def expand_query(
    normalized_text: str,
    is_arabic: bool = True,
    query_type: str = "general",
    max_expansion_tokens: int = 40,
    query_intent: str = "",
) -> ExpandedQuery:
    """
    Main query expansion entry point.

    Calls the Gemini LLM to intelligently expand the query with:
      - Morphological surface forms
      - Islamic synonym / ontology terms
      - Hadith-corpus collocations
      - Alternative query framings

    When query_intent is set (e.g. 'truth_claim'), two extra mechanisms fire:
      1. An intent-specific note is appended to the LLM prompt so the model
         focuses on the right semantic cluster.
      2. Hard-coded high-precision corpus phrases are merged into the result
         *after* the LLM call, guaranteeing they are always present even if
         the LLM mis-focuses on surface keywords.

    Falls back silently to identity expansion on any error.

    Args:
        normalized_text: Pre-normalized query (tashkeel/tatweel stripped)
        is_arabic: Whether the query is Arabic (unused — kept for API compat)
        query_type: QueryType value string; skipped types return identity
        max_expansion_tokens: Cap on terms added to sparse query
        query_intent: Fine-grained intent tag (e.g. 'truth_claim'); '' = none

    Returns:
        ExpandedQuery with all expansion results
    """
    if not normalized_text:
        return _identity_expansion(normalized_text)

    # Skip expansion for non-retrieval query types
    if query_type in _SKIP_TYPES:
        result = ExpandedQuery(
            original=normalized_text,
            dense_query=normalized_text,
            sparse_query=normalized_text,
            multi_queries=[normalized_text],
        )
        return result

    # Cache hit — key includes intent so different intents on the same query
    # text are not served the same (possibly un-enriched) cached result.
    cache_key = (
        f"{normalized_text}||{query_intent}" if query_intent else normalized_text
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        extra_note = _INTENT_PROMPT_NOTES.get(query_intent, "")
        data = _llm_expand(normalized_text, extra_note=extra_note)
        result = _assemble(normalized_text, data, max_expansion_tokens)
    except Exception as exc:
        logger.warning(
            "LLM query expansion failed for %r — using identity fallback. Error: %s",
            normalized_text[:80],
            exc,
        )
        result = _identity_expansion(normalized_text)

    # ── Intent-specific term injection ──────────────────────────────────
    # Hard-coded corpus phrases are merged in AFTER the LLM result.
    # They are prepended so they rank highest in the sparse index, ensuring
    # the retriever sees exact-match phrases even when the LLM under-expands.
    if query_intent and query_intent in _INTENT_EXPANSION_TERMS:
        intent_terms = _INTENT_EXPANSION_TERMS[query_intent]
        # Prepend intent terms; drop duplicates already returned by LLM
        merged_terms = intent_terms + [
            t for t in result.expanded_terms if t not in intent_terms
        ]
        # Sparse query: original text + top intent phrases (character n-gram gold)
        top_intent_str = " ".join(intent_terms[:6])
        intent_sparse = f"{normalized_text} {top_intent_str}".strip()
        # Add one intent-anchored multi-query variant for RRF diversity
        intent_variant = f"{normalized_text} {' '.join(intent_terms[:3])}".strip()
        extra_multi = (
            [intent_variant] if intent_variant not in result.multi_queries else []
        )
        result = ExpandedQuery(
            original=result.original,
            expanded_terms=merged_terms[:max_expansion_tokens],
            reformulations=result.reformulations,
            dense_query=result.dense_query,
            sparse_query=intent_sparse,
            multi_queries=result.multi_queries + extra_multi,
        )
        logger.info(
            "QueryExpander[intent=%s]: injected %d hard-coded terms into sparse query",
            query_intent,
            len(intent_terms),
        )

    _cache_set(cache_key, result)

    logger.info(
        "QueryExpander: +%d terms, %d reformulations, sparse_len=%d tokens",
        len(result.expanded_terms),
        len(result.reformulations),
        len(result.sparse_query.split()),
    )

    return result


# ============================================================
# Convenience: Multi-Query Retrieval Helper
# (kept for backward compat with hybrid_retriever.py)
# ============================================================


def get_all_query_variants(
    expanded: ExpandedQuery,
    include_original: bool = True,
) -> list[str]:
    """Returns all query variants for multi-query retrieval."""
    variants: list[str] = []
    if include_original:
        variants.append(expanded.original)
    for q in expanded.multi_queries:
        if q not in variants:
            variants.append(q)
    return variants


# ============================================================
# CLI Test Harness
# ============================================================

if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    logging.basicConfig(level=logging.DEBUG)

    test_queries = [
        # ── General topic queries (no qualifier) ──────────────────────────
        "كيفية التيمم",
        "أحاديث عن الصبر",
        "ما حكم ترك الصلاة",
        "النية في العبادة",
        "من غشنا فليس منا",
        "فضل الصدقة",
        "يوم القيامة والحساب",
        "حديث عن بر الوالدين",
        "ما فضل قيام الليل",
        "الأخلاق في الإسلام",
        # ── Conditional / situational queries — KEY regression tests ──────
        # These MUST produce expanded_terms focused on the qualifier,
        # NOT on the broad topic.
        "اعطني حديث عن الصدقات في حالة عدم المقدرة",  # inability to give charity
        "الصلاة للمريض الذي لا يستطيع القيام",  # prayer for the sick
        "الصيام في السفر هل يجوز الإفطار",  # fasting while travelling
        "حكم الوضوء عند عدم وجود الماء",  # tayammum when no water
        "الذكر والدعاء عند الغضب الشديد",  # dhikr when angry
        "كيف يؤدي المسلم العبادة في المرض",  # worship during illness
        "أحاديث عن الصبر عند فقدان الولد",  # patience on losing a child
        "حديث عن التوبة بعد ارتكاب الكبائر",  # repentance after major sins
    ]

    print(f"{'=' * 70}")
    print("YaqeenAI — LLM Query Expansion Test")
    print(f"{'=' * 70}\n")

    for q in test_queries:
        exp = expand_query(q)
        print(f"Original:       {q}")
        print(f"Dense query:    {exp.dense_query}")
        print(f"Sparse query:   {exp.sparse_query[:120]}")
        print(f"Reformulations: {exp.reformulations}")
        print(f"Extra tokens:   {exp.expanded_terms[:10]}")
        print(f"Multi-queries:  {exp.multi_queries}")
        print(f"{'-' * 60}")
