# ============================================================
# YaqeenAI — Generation Module (Production Enhanced)
# ============================================================
# Generates answers using Gemini API with Arabic-first system prompt.
# Assembles context from reranked hadiths and enforces citation rules.
#
# KEY FIX (v2): Decision-tree gate is now enforced in Python BEFORE
# the LLM is called. The system prompt gate alone was insufficient —
# the LLM was ignoring BRANCH 3 and picking arbitrary hadiths.
#
# KEY FIX (v3):
#   • Import check_context from answer_policy for explicit empty-chunk guard.
#   • HadithGenerator.generate() is the SINGLE source of truth for gate logic.
#   • Module-level generate() is now a pure thin wrapper — it no longer
#     duplicates _apply_decision_tree_gate or classify_answer_intent, which
#     previously caused (a) double gate execution and (b) the risk that the
#     gate fired on a different intent value than the one used for generation.
#
# KEY FIX (v4):
#   • REMOVED _generate_general_explanatory_fallback() and
#     SYSTEM_PROMPT_GENERAL_EXPLANATION_FALLBACK entirely.
#   • When evidence is insufficient/partial for EXPLANATORY intent, the system
#     now returns _REFUSAL_NO_CONTEXT directly — NO LLM call is made.
#   • This closes the self-knowledge escape hatch: the LLM can no longer answer
#     from training data when retrieval fails.
#   • Contract: no hadith in context → no answer. Period.
#
# KEY FIX (v5):
#   • ROOT CAUSE: The LLM was re-running the decision tree internally after the
#     Python gate had already cleared the query. Sensitive fiqh topics (زنا,
#     حيض, طلاق) were being mis-fired as BRANCH 2 (anachronistic) by the LLM
#     even though the Python gate correctly passed them as BRANCH 5.
#
#   • FIX A (architectural): Created BRANCH5-only system prompt variants that
#     omit _DECISION_TREE_GATE entirely. The gate is Python-enforced; repeating
#     it in the LLM prompt creates a second, unreliable gate.
#
#   • FIX B (belt-and-suspenders): When evidence is sufficient and the Python
#     gate has already cleared the query, an explicit instruction block is
#     appended to user_message telling the LLM it is in BRANCH 5 and must not
#     re-evaluate the decision tree.
#
#   • _select_system_prompt() now always returns a BRANCH5 prompt variant,
#     since generate() only reaches that function after the Python gate passes.
#
# KEY FIX (v6):
#   • ROOT CAUSE: LLM was saying "لا يوجد في السياق ما يدل" even when there
#     were sahih/hasan hadiths on a *related* topic (e.g. kissing/wudu when
#     asked about zina/wudu). It was refusing to synthesise an answer from
#     nearby evidence.
#
#   • FIX A: _ANSWER_RULES_GENERAL now explicitly instructs the LLM to use
#     related hadiths to construct a partial/contextual answer rather than
#     returning a bare "nothing found" response.
#
#   • FIX B: EXPLANATORY intent policy rule #3 rewritten.
#
#   • FIX C: _format_hadith_context for EXPLANATORY no longer emits the
#     "لا يوجد حديث صحيح أو حسن" section header when authentic hadiths do
#     exist.
#
#   • FIX D: _wrap_audited_answer now returns (clean_answer, debug_block) as
#     a tuple — warnings are separated from user-facing content.
#
#   • FIX E: _filter_offtopic_hadiths() pre-filters completely off-topic
#     hadiths (topic-overlap < threshold) before they are sent to the LLM.
#
# KEY FIX (v7):  ← THIS VERSION
#   • FIX A (verdict-first): Added _VERDICT_FIRST_RULE to all BRANCH5 system
#     prompts. The LLM must open with a single direct-verdict sentence before
#     any elaboration or hadith listing.
#
#   • FIX B (answer-relevance filter): Added _extract_answer_target_tokens()
#     and _filter_answer_irrelevant_hadiths(). Hadiths that share a surface
#     token with the query but address a completely different legal topic are
#     removed before generation. The filter is fully algorithmic — no
#     hardcoded domain terms.
#
#   • FIX C (output separation): GeneratedResponse gains an `answer_debug`
#     field. _wrap_audited_answer() returns (clean_answer, debug_block).
#     The user-facing `answer` field is now free of excluded-narration noise;
#     the UI layer decides whether to render `answer_debug`.

import logging
import re
from difflib import SequenceMatcher
from typing import Optional
from dataclasses import dataclass, field, asdict

from google import genai
from google.genai import types

from pipeline.answer_policy import (
    AnswerIntent,
    classify_answer_intent,
    check_context,
    grade_priority,
)
from pipeline.config import (
    audit_grade,
    resolve_grade_bucket,
    resolve_grade_label,
    settings,
)
from pipeline.retrieve import RetrievedHadith

logger = logging.getLogger(__name__)


# ============================================================
# HARDCODING PROHIBITION (shared across all system prompts)
# ============================================================

_HARDCODING_PROHIBITION = """
════════════════════════════════════════
منع الترميز الثابت (HARDCODING PROHIBITION)
════════════════════════════════════════
أنت آلة استخراج، لا قاعدة معرفة.
كل حقيقة تكتبها يجب أن تأتي من السياق المقدم فقط — لا من تدريبك.

لا تكتب أبداً:
  ✗ اسم كتاب لم يرد في السياق
  ✗ رقم حديث أو صفحة لم يرد في السياق
  ✗ اسم راوٍ لم يرد في السياق
  ✗ درجة (صحيح/حسن/ضعيف) لم ترد في السياق
  ✗ نص متن لم يرد في السياق
  ✗ اسم محدِّث أو عالم لم يرد في السياق
  ✗ أي رقم أو تاريخ أو إحصاء لم يرد في السياق

الاختبار: إذا كان بإمكانك كتابة هذه المعلومة دون النظر إلى السياق
          فهي معلومة مُرمَّزة ثابتة — لا تكتبها.

إذا كان الحقل غائباً من السياق → احذف السطر كاملاً، لا تكتب "غير محدد".
"""

# ============================================================
# VERDICT FIRST RULE (v7 — appended to all BRANCH5 prompts)
# ============================================================

_VERDICT_FIRST_RULE = """
════════════════════════════════════════
قاعدة الحكم المباشر — يجب تطبيقها أولاً قبل أي شيء
════════════════════════════════════════
السطر الأول من إجابتك يجب أن يكون جملة واحدة مكتملة تُجيب على السؤال
بشكل مباشر وصريح — قبل أي شرح أو عرض أحاديث.

أمثلة إلزامية:
  سؤال: «هل الزنا يبطل الوضوء؟»
  ✓ الصواب (السطر الأول): «لا، الزنا لا يبطل الوضوء، لكنه يوجب الغسل.»
  ✗ الخطأ: «لم يرد في الأحاديث الصحيحة نص صريح يجعل الزنا ناقضاً للوضوء...»

  سؤال: «هل الحسد حرام؟»
  ✓ الصواب (السطر الأول): «نعم، الحسد منهي عنه وقد حذّر منه النبي ﷺ.»
  ✗ الخطأ: البدء بسرد الأحاديث قبل الحكم.

  سؤال: «ما حكم الكذب على الزوجة؟»
  ✓ الصواب (السطر الأول): «الكذب مذموم في الأصل، وقد أجاز الشرع الكذب في حالات
    محددة منها الإصلاح بين الناس.»
  ✗ الخطأ: البدء بـ «جاء في السنة النبوية...»

الترتيب الإلزامي الثابت:
  1. جملة الحكم (سطر واحد في أول الإجابة)
  2. الشرح والتفصيل المستمد من السياق
  3. الأحاديث الداعمة مع درجاتها
"""

# ============================================================
# DECISION TREE GATE — Python-enforced only, NOT sent to LLM.
# ============================================================

_DECISION_TREE_GATE = """
════════════════════════════════════════
بوابة القرار — اتبع هذا الترتيب أولاً قبل أي إجابة
════════════════════════════════════════
قبل كتابة أي شيء، صنّف السؤال في رأسك ثم توقف عند أول فرع ينطبق:

BRANCH 1: هل السؤال رموز أو أرقام عشوائية أو نص غير مفهوم؟
  نعم → اكتب بالضبط: «السؤال غير صالح أو غير مفهوم.»
        ثم توقف. لا تضف شيئاً.

BRANCH 2: هل يسأل عن شيء لا يمكن أن يوجد في الحديث النبوي؟
  (أمثلة: ذكاء اصطناعي، إنترنت، أحداث 2025، كتب خيالية، رواة مخترعون)
  نعم → اكتب بالضبط: «لا يوجد في المصادر الحديثية ما يدعم هذا السؤال.»
        ثم توقف. لا تعرض أحاديث غير ذات صلة. لا تشرح.

BRANCH 3: هل يقول "هذا الحديث" أو "الحديث" دون تحديد نص الحديث؟
  أو هل ينقصه معلومة أساسية لتحديد المطلوب؟
  نعم → اكتب بالضبط: «السؤال غير واضح. يرجى تحديد نص الحديث أو توضيح المطلوب بدقة.»
        ثم توقف. لا تختر حديثاً عشوائياً من السياق وتجيب عنه.

BRANCH 4: هل السياق فارغ [] أو كل الأجزاء لا علاقة لها بالسؤال؟
  نعم → اكتب بالضبط: «لا يوجد معلومات كافية للإجابة على هذا السؤال من المصادر المتاحة.»
        ثم توقف.

BRANCH 5: يوجد سياق وثيق الصلة → انتقل إلى قواعد الإجابة أدناه.
"""

_FEW_SHOT_EXAMPLES = """
════════════════════════════════════════
أمثلة — الصواب والخطأ
════════════════════════════════════════

مثال 1 — BRANCH 3 (سؤال غامض)
  السؤال: «ما صحة هذا الحديث؟»
  السياق: [5 أحاديث مختلفة]
  ✓ الصواب: «السؤال غير واضح. يرجى تحديد نص الحديث أو توضيح المطلوب بدقة.»
  ✗ الخطأ: اختيار الحديث الأول من السياق والإجابة عنه
  السبب: السؤال لا يحدد حديثاً، فاختيار أي حديث هو اختلاق للنية.

مثال 2 — منع الترميز الثابت (أخطر نوع من الهلوسة)
  السياق: «الراوي: أبو هريرة | الدرجة: صحيح | المتن: من غشنا فليس منا»
          (بدون رقم، بدون اسم كتاب)
  ✓ الصواب:
      الراوي: أبو هريرة
      الدرجة: صحيح
      المتن: من غشنا فليس منا
  ✗ الخطأ: إضافة «المصدر: صحيح مسلم، الرقم: 101»
  السبب: هذا الرقم لم يكن في السياق — هو هلوسة من ذاكرة التدريب.

مثال 3 — الاكتمال (سؤال بصيغة الجمع)
  السؤال: «أعطني أحاديث صحيحة عن الصيام»
  السياق: [3 أحاديث صحيح، 1 ضعيف]
  ✓ الصواب: عرض الثلاثة الصحيحة كاملة + الضعيف في قسم المستبعدات
  ✗ الخطأ: عرض الحديث الأول فقط
  السبب: السؤال بصيغة الجمع يستلزم جميع النتائج الصالحة.

مثال 4 — القضايا الفقهية الحساسة (مثال بالغ الأهمية)
  السؤال: «هل ينقض الزنا الوضوء؟» أو «ما حكم الحيض في الصلاة؟»
  ✓ الصواب: الإجابة مباشرة من الأحاديث الواردة في السياق.
  ✗ الخطأ: رفض السؤال أو تصنيفه على أنه "لا يمكن أن يوجد في الحديث"
  السبب: هذه مسائل فقهية كلاسيكية وردت في كتب الحديث منذ القرن الأول.
          الوصول إلى هنا يعني أن البوابة البرمجية قد تحققت من صحة السؤال.

مثال 5 — الإجابة بالأدلة ذات الصلة (مثال بالغ الأهمية — FIX v6)
  السؤال: «هل ينقض الزنا الوضوء؟»
  السياق: يحتوي أحاديث عن نواقض الوضوء (لمس المرأة، الضحك في الصلاة...)
          لكن لا يوجد حديث يذكر الزنا صراحةً بوصفه ناقضاً للوضوء.
  ✓ الصواب:
      «لا، الزنا لا يبطل الوضوء — لكنه يوجب الغسل. لم يرد في الأحاديث الصحيحة
       نص صريح يجعل الزنا ناقضاً للوضوء من حيث الوضوء نفسه. والسياق يتضمن
       أحاديث في نواقض الوضوء: [اذكر الأحاديث المتاحة مع درجاتها].»
  ✗ الخطأ: «لا يوجد في السياق ما يدل على أن الزنا يبطل الوضوء.» (سطر واحد)
  السبب: الإجابة يجب أن تستثمر الأحاديث المتاحة وتُقدّم سياقاً فقهياً مفيداً،
          لا أن تُعيد صياغة غياب الدليل المباشر كجواب نهائي.
"""

# ============================================================
# BRANCH5 GATE SUPPRESSION NOTE
# ============================================================

_BRANCH5_GATE_SUPPRESSION = (
    "\n\n## ⚠️ تعليمات البوابة البرمجية:\n"
    "هذا السؤال اجتاز جميع فروع شجرة القرار البرمجية (BRANCH 1-4) وتم التحقق من "
    "كفاية الأدلة المسترجعة. أنت الآن في BRANCH 5 حصراً. "
    "لا تعد تطبيق شجرة القرار ولا تُصدر أي رفض من تلقاء نفسك. "
    "انتقل مباشرة إلى قواعد الإجابة واستخرج الإجابة من السياق المقدم."
)

# ============================================================
# Python-level decision tree gate
# ============================================================

_VAGUE_HADITH_REFS = re.compile(
    r"""
    (?:
        هذا\s+الحديث           |
        هذه\s+الرواية          |
        هذا\s+الكلام           |
        (?<!\S)الحديث(?!\s+(?:النبوي|الشريف|المذكور|التالي|الآتي|عن|في|من))
    )
    """,
    re.VERBOSE,
)

_ANACHRONISTIC_TOPICS = re.compile(
    r"""
    # ── A: modern technology ─────────────────────────────────────────────────
    (?:ال)?ذكاء\s+الاصطناعي      |
    ذكاء\s+اصطناعي\s+توليدي     |
    إنترنت | الإنترنت            |
    الفضاء\s+الإلكتروني          |
    كورونا | كوفيد               |
    لقاح\s+(?:كورونا|فيروس)      |
    برنامج\s+(?:حاسوب|كمبيوتر)  |
    (?:روبوت|أندرويد|سيبراني)    |
    تويتر | فيسبوك | انستغرام | يوتيوب |
    واتساب | تيك\s*توك           |
    # ── B: anachronistic years ───────────────────────────────────────────────
    (?:سنة|عام)\s+(?:19[5-9]\d|20\d{2})  |
    # ── C: fictional / impossible entities ───────────────────────────────────
    المريخ                       |
    (?:الطيران|السفر)\s+إلى\s+(?:الفضاء|المريخ|القمر) |
    كتاب\s+الزمرد               |
    الزمرد\s+الخفي              |
    باب\s+الطيران               |
    راوٍ?\s+(?:خيالي|مخترع|وهمي) |
    (?:كتاب|باب|حديث)\s+\w+\s+الخيالي
    """,
    re.VERBOSE,
)

_UNSPECIFIED_REQUEST = re.compile(
    r"""
    موضوع\s+غير\s+محدد                      |
    (?:اي|أي)\s+حديث                         |
    حديث(?:اً)?\s+عشوائي(?:اً)?              |
    (?:اكتب|اذكر|اعطني)\s+(?:نص\s+)?حديث(?:اً)?\s+(?:طويل[^،.؟\n]{0,20})?(?:عن\s+موضوع\s+)?غير\s+محدد
    """,
    re.VERBOSE,
)

_GIBBERISH = re.compile(
    r"""
    ^[\s\d\!\@\#\$\%\^\&\*\(\)\[\]\{\}\|\\\/\?\،\؟\.\,\-\_\+\=]*$
    """,
    re.VERBOSE,
)

_REFUSAL_INVALID      = "السؤال غير صالح أو غير مفهوم."
_REFUSAL_ANACHRONISM  = "لا يوجد في المصادر الحديثية ما يدعم هذا السؤال."
_REFUSAL_VAGUE        = "السؤال غير واضح. يرجى تحديد نص الحديث أو توضيح المطلوب بدقة."
_REFUSAL_NO_CONTEXT   = "لا يوجد معلومات كافية للإجابة على هذا السؤال من المصادر المتاحة."

# Minimum token overlap ratio to consider a hadith "not completely off-topic"
_OFFTOPIC_FILTER_MIN_OVERLAP = 0.10

# Minimum fraction of answer-target tokens a hadith must contain
# to be considered answer-relevant (used in second-pass filter)
_ANSWER_RELEVANCE_MIN_TARGET_COVERAGE = 0.25


def _has_specific_hadith_text(query: str) -> bool:
    clean = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670]", "", query)
    clean = re.sub(r"[أإآٱ]", "ا", clean)
    stopwords = {
        "ما", "هل", "من", "في", "عن", "على", "إلى", "الى", "هذا", "هذه",
        "صحة", "صحيح", "حكم", "درجة", "اسناد", "رواه", "أخرجه", "خرجه",
        "الحديث", "الرواية", "هل", "سند", "متن",
    }
    words = [w for w in clean.split() if len(w) >= 3 and w not in stopwords]
    return len(words) >= 4


def _apply_decision_tree_gate(
    query: str,
    hadiths: list,
) -> Optional[str]:
    """
    Enforces branches 1-4 of the decision tree in Python.

    Returns:
        A refusal string  → caller should short-circuit and return it as the answer.
        None              → normal generation should proceed (BRANCH 5).
    """
    stripped = query.strip()

    # ── BRANCH 1: gibberish / empty ──────────────────────────────────────────
    if not stripped or _GIBBERISH.match(stripped):
        logger.info("Gate: BRANCH 1 (gibberish/empty query)")
        return _REFUSAL_INVALID

    # ── BRANCH 2: anachronistic / impossible topic ───────────────────────────
    if _ANACHRONISTIC_TOPICS.search(stripped):
        logger.info("Gate: BRANCH 2 (anachronistic/impossible topic)")
        return _REFUSAL_ANACHRONISM

    # ── BRANCH 3a: explicit unspecified-topic request ────────────────────────
    if _UNSPECIFIED_REQUEST.search(stripped):
        logger.info("Gate: BRANCH 3a (explicitly unspecified hadith request)")
        return _REFUSAL_VAGUE

    # ── BRANCH 3b: query references an unspecified hadith ────────────────────
    if _VAGUE_HADITH_REFS.search(stripped) and not _has_specific_hadith_text(stripped):
        logger.info("Gate: BRANCH 3b (vague hadith reference without text)")
        return _REFUSAL_VAGUE

    # ── BRANCH 4: no context at all ──────────────────────────────────────────
    empty_refusal = check_context(hadiths)
    if empty_refusal:
        logger.info("Gate: BRANCH 4 (empty hadith context)")
        return empty_refusal

    # ── BRANCH 5: proceed ────────────────────────────────────────────────────
    return None


# ============================================================
# Off-topic hadith pre-filter  (FIX v6 — FIX E)
# ============================================================

def _normalize_for_overlap(text: str) -> set[str]:
    """Return a bag of significant Arabic tokens (≥3 chars, no diacritics)."""
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]+", "", str(text or ""))
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    tokens = re.findall(r"[\u0600-\u06FF]{3,}", text)
    stopwords = {
        "قال", "عن", "ان", "على", "في", "من", "الى", "إلى", "ما", "لا",
        "كان", "له", "هو", "هي", "ذلك", "هذا", "هذه", "يوم", "عند",
        "رسول", "الله", "صلى", "عليه", "وسلم", "النبي", "حديث", "روي",
    }
    return {t for t in tokens if t not in stopwords}


def _hadith_query_overlap(query_tokens: set[str], hadith: "RetrievedHadith") -> float:
    """Jaccard-style overlap between query tokens and all hadith text fields."""
    haystack_text = " ".join(filter(None, [
        hadith.text_ar,
        hadith.category,
        hadith.subcategory_name,
        hadith.rawi,
    ]))
    hadith_tokens = _normalize_for_overlap(haystack_text)
    if not query_tokens or not hadith_tokens:
        return 0.0
    intersection = query_tokens & hadith_tokens
    return len(intersection) / len(query_tokens)


def _filter_offtopic_hadiths(
    query: str,
    hadiths: list["RetrievedHadith"],
    min_overlap: float = _OFFTOPIC_FILTER_MIN_OVERLAP,
) -> list["RetrievedHadith"]:
    """
    Remove hadiths whose content shares virtually no tokens with the query.
    This prevents completely unrelated hadiths from polluting the LLM context.

    A hadith passes if its overlap ratio >= min_overlap OR if it is the only
    hadith in the list (we never return an empty list from this function when
    the input is non-empty — the gate handles the truly empty case).
    """
    if not hadiths:
        return hadiths

    query_tokens = _normalize_for_overlap(query)
    if not query_tokens:
        return hadiths

    scored = [
        (h, _hadith_query_overlap(query_tokens, h))
        for h in hadiths
    ]
    filtered = [h for h, score in scored if score >= min_overlap]

    if not filtered:
        filtered = [max(scored, key=lambda x: x[1])[0]]
        logger.info(
            "Off-topic filter: all hadiths below threshold — keeping best match "
            f"(overlap={max(s for _, s in scored):.3f})"
        )
    else:
        removed = len(hadiths) - len(filtered)
        if removed > 0:
            logger.info(f"Off-topic filter: removed {removed} completely unrelated hadith(s)")

    return filtered


# ============================================================
# Answer-relevance filter  (FIX v7 — FIX B)
# ============================================================
#
# _filter_offtopic_hadiths catches hadiths with no surface token overlap.
# This second-pass filter catches hadiths that share a surface token with the
# query (e.g. "زنا") but are fundamentally answering a different legal question
# (e.g. the moral gravity of zina vs. its effect on ritual purity).
#
# The filter is fully algorithmic:
#   1. Extract "answer-target tokens" from the query using structural patterns
#      (the part of the question asking ABOUT something, not the subject).
#   2. For each hadith, compute what fraction of the answer-target tokens it
#      covers.
#   3. Hadiths below the coverage threshold are downgraded; if enough hadiths
#      pass, the low-coverage ones are dropped.
#
# No domain-specific blocklists or hardcoded Arabic phrases are used.
# ============================================================

# Structural patterns to extract the "answer target" of the question.
# Group 1 of each pattern captures the answer-target phrase.
_ANSWER_TARGET_PATTERNS: list[tuple[re.Pattern, int]] = [
    # هل X يبطل/ينقض/يوجب/يحرم Y → Y is what we're asking about
    (re.compile(
        r"(?:هل|يجوز|هل\s+يجوز)\s+[\u0600-\u06FF\s]{1,20}"
        r"(?:يبطل|ينقض|يحرم|يجيز|يوجب|يسقط|يفسد|يكفر|يحل|يُبطل|يُنقض)\s+"
        r"([\u0600-\u06FF]{3,}(?:\s+[\u0600-\u06FF]{3,}){0,3})"
    ), 1),
    # ما أثر X على Y → Y is the answer target
    (re.compile(
        r"(?:ما|ما\s+هو)\s+أثر\s+[\u0600-\u06FF\s]{1,20}\s+على\s+"
        r"([\u0600-\u06FF]{3,}(?:\s+[\u0600-\u06FF]{3,}){0,3})"
    ), 1),
    # هل X يؤثر على Y → Y is the answer target
    (re.compile(
        r"هل\s+[\u0600-\u06FF\s]{1,20}يؤثر\s+على\s+"
        r"([\u0600-\u06FF]{3,}(?:\s+[\u0600-\u06FF]{3,}){0,3})"
    ), 1),
    # ما حكم X في/على Y → Y is the legal domain being asked about
    (re.compile(
        r"(?:ما|هل)\s+حكم\s+[\u0600-\u06FF\s]{1,25}(?:في|على)\s+"
        r"([\u0600-\u06FF]{3,}(?:\s+[\u0600-\u06FF]{3,}){0,3})"
    ), 1),
]


def _extract_answer_target_tokens(query: str) -> set[str]:
    """
    Extract tokens representing the "answer target" of the question —
    the concept the user wants information ABOUT, as opposed to the
    grammatical subject.

    Example:
        "هل الزنا يبطل الوضوء؟"
        → answer target: {وضوء}   (what we're asking about)
        → NOT {زنا}               (the subject / context)

    Returns an empty set when no structural pattern matches; in that case
    the caller should skip answer-relevance filtering.
    """
    # Strip diacritics and normalise alefs
    q = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u0640]+", "", query)
    q = re.sub(r"[أإآٱ]", "ا", q)
    q = re.sub(r"ة", "ه", q)

    for pattern, group_index in _ANSWER_TARGET_PATTERNS:
        m = pattern.search(q)
        if m:
            target_phrase = m.group(group_index)
            tokens = _normalize_for_overlap(target_phrase)
            if tokens:
                logger.debug(f"Answer-target extracted: {tokens!r} from query: {query!r:.60}")
                return tokens

    return set()


def _hadith_answer_target_coverage(
    target_tokens: set[str],
    hadith: "RetrievedHadith",
) -> float:
    """
    Fraction of answer-target tokens covered by the hadith's content.
    A score of 0 means the hadith does not address the answer target at all.
    """
    if not target_tokens:
        return 1.0  # can't evaluate → assume relevant

    haystack_text = " ".join(filter(None, [
        hadith.text_ar,
        hadith.category,
        hadith.subcategory_name,
    ]))
    hadith_tokens = _normalize_for_overlap(haystack_text)
    if not hadith_tokens:
        return 0.0

    covered = target_tokens & hadith_tokens
    return len(covered) / len(target_tokens)


def _filter_answer_irrelevant_hadiths(
    query: str,
    hadiths: list["RetrievedHadith"],
    min_target_coverage: float = _ANSWER_RELEVANCE_MIN_TARGET_COVERAGE,
) -> list["RetrievedHadith"]:
    """
    Second-pass filter (v7): removes hadiths that passed _filter_offtopic_hadiths
    (i.e. share some surface tokens with the query) but do not actually address
    the answer target of the question.

    Algorithm:
      1. Extract answer-target tokens from the query using structural patterns.
      2. If no target tokens are found, skip filtering (cannot evaluate).
      3. Score each hadith by how many target tokens it covers.
      4. Drop hadiths below min_target_coverage, but only when enough hadiths
         remain above the threshold (never return an empty list).

    No hardcoded Arabic domain terms are used anywhere in this function.
    """
    if not hadiths:
        return hadiths

    target_tokens = _extract_answer_target_tokens(query)
    if not target_tokens:
        # Structural pattern did not match → cannot determine target → skip filter
        return hadiths

    scored = [
        (h, _hadith_answer_target_coverage(target_tokens, h))
        for h in hadiths
    ]

    passing = [h for h, score in scored if score >= min_target_coverage]

    if not passing:
        # All hadiths failed → keep the single best rather than returning empty
        best = max(scored, key=lambda x: x[1])
        logger.info(
            f"Answer-relevance filter: no hadith above coverage threshold "
            f"({min_target_coverage:.2f}) — keeping best "
            f"(coverage={best[1]:.3f}): {best[0].text_ar[:60]!r}"
        )
        return [best[0]]

    removed = len(hadiths) - len(passing)
    if removed > 0:
        logger.info(
            f"Answer-relevance filter: removed {removed} answer-irrelevant hadith(s) "
            f"(target_tokens={target_tokens!r})"
        )

    return passing


# ============================================================
# BRANCH5-only system prompts
# ============================================================

_ANSWER_RULES_GENERAL = """
════════════════════════════════════════
قواعد الإجابة
════════════════════════════════════════

طريقة الإجابة:
  • قدم إجابة مباشرة ووافية للسؤال أولاً، موضحاً الحكم أو الفائدة بأسلوب واضح يربط
    بين نصوص الأحاديث وسؤال المستخدم. لا تكتفِ بسرد نص الحديث فقط.

  • استخدام الأدلة ذات الصلة (مهم جداً — تجنب الإجابة الفارغة):
    إذا لم يوجد حديث يذكر موضوع السؤال صراحةً، لكن يوجد في السياق أحاديث ذات صلة
    وثيقة بالموضوع (مثل: سُئل عن حكم الزنا على الوضوء والسياق يحتوي أحاديث نواقض الوضوء)،
    فيجب عليك:
    أ) استخدام تلك الأحاديث المتاحة لتقديم إجابة جزئية أو سياقية مفيدة.
    ب) التصريح بأن السياق لا يتضمن نصاً صريحاً في ذلك تحديداً.
    ج) الاستنتاج الفقهي من الأحاديث المتاحة إن كان ممكناً.
    ✗ لا تكتفِ بسطر واحد «لا يوجد في السياق ما يدل على...» — هذا تقصير، ليس إجابة.

  • التعاطف والحكمة في القضايا الحساسة: عند الإجابة على أسئلة تتعلق بقضايا حساسة
    أو اجتماعية (مثل قضايا المرأة، أو الأحكام التي قد تُفهم خطأً)، قدم إجابة متزنة،
    حكيمة، ومراعية لشعور السائل. اشرح السياق والمقصد الشرعي بلطف ولا تكتفِ بسرد
    ظواهر النصوص بطريقة قد تبدو قاسية، مع إبراز جوانب الرحمة والعدل في الشريعة.
  • بعد الإجابة المباشرة، اذكر الأحاديث التي استندت إليها.

الترتيب: رتّب الأحاديث حسب قوة الإسناد: صحيح ← حسن ← ضعيف.

التحذير من الضعيف والموضوع:
  • اكتب قبله: «⚠️ [الدرجة]: لا يُحتج به في إثبات الأحكام»
  • لا تبنِ عليه حكماً شرعياً.

منع التكرار:
  • إذا تكررت روايات متقاربة جداً، اذكر الأقوى سنداً فقط.

منع الإضافات:
  • لا تقل «والله أعلم» أو أي عبارة ختامية شخصية.
  • لا تشرح معنى الحديث بزيادات لم ترد في السياق، لكن صغ الإجابة المباشرة المستمدة
    من ظاهر الأحاديث.
  • لا تضف مقدمات من نوع «جاء في السنة النبوية...».

مسائل الخلاف:
  • إذا وُجد خلاف واضح في السياق، اعرض الأقوال المختلفة ولا تنحز.

التنسيق:
  • استخدم بنية واضحة مع إبراز الإجابة المباشرة، متبوعة بالأحاديث مع فواصل بينها.
  • اللغة: العربية الفصحى فقط.
"""

SYSTEM_PROMPT_GENERAL_BRANCH5 = (
    "أنت عالم متخصص في الحديث النبوي الشريف. "
    "مهمتك استخراج المعلومات من السياق المقدم فقط والإجابة عن أسئلة المستخدم. "
    "أنت لا تضيف شيئاً من معرفتك الخاصة. "
    "لقد تحققت البوابة البرمجية من أن السؤال صالح وأن الأدلة كافية — "
    "انتقل مباشرة إلى الإجابة دون إعادة تقييم السؤال."
    + _VERDICT_FIRST_RULE
    + _HARDCODING_PROHIBITION
    + _FEW_SHOT_EXAMPLES
    + _ANSWER_RULES_GENERAL
)

SYSTEM_PROMPT_METADATA_BRANCH5 = (
    "أنت متخصص في الحديث النبوي. المستخدم يسأل عن بيانات وصفية محددة "
    "(الراوي، الدرجة، المصدر، الرقم، المحدث، التصنيف). "
    "أنت آلة استخراج بيانات فقط. "
    "لقد تحققت البوابة البرمجية من أن السؤال صالح — انتقل مباشرة إلى الإجابة."
    + _VERDICT_FIRST_RULE
    + _HARDCODING_PROHIBITION
    + _FEW_SHOT_EXAMPLES
    + """
════════════════════════════════════════
قواعد إجابات البيانات الوصفية
════════════════════════════════════════

ابدأ بالإجابة المباشرة أولاً:
  • «من رواه؟»         → ابدأ باسم الراوي مباشرة
  • «ما درجته؟»        → ابدأ بالدرجة مباشرة
  • «في أي كتاب؟»      → ابدأ باسم الكتاب مباشرة
  • «ما رقمه؟»         → ابدأ بالرقم مباشرة
  • «من حكم عليه؟»     → ابدأ باسم المحدث مباشرة

إذا لم تجد المعلومة في السياق:
  → اكتب: «لم أجد هذه المعلومة في البيانات المتاحة»
  → لا تخمن أو تستنتج.

بعد الإجابة المباشرة، أضف السياق الكامل المتاح في السياق:
  نص المتن + المصدر + الراوي + الدرجة (فقط ما هو موجود في السياق).

إذا وُجد الحديث في أكثر من مصدر في السياق: اذكر جميعها.
"""
)

SYSTEM_PROMPT_NARRATOR_BRANCH5 = (
    "أنت متخصص في الحديث النبوي. المستخدم يبحث عن أحاديث مرتبطة براوٍ محدد. "
    "أنت تعرض فقط ما هو في السياق. "
    "لقد تحققت البوابة البرمجية من أن السؤال صالح — انتقل مباشرة إلى الإجابة."
    + _VERDICT_FIRST_RULE
    + _HARDCODING_PROHIBITION
    + _FEW_SHOT_EXAMPLES
    + """
════════════════════════════════════════
قواعد عرض أحاديث الراوي
════════════════════════════════════════

  • اعرض فقط الأحاديث الموجودة في السياق لهذا الراوي.
  • لكل حديث: المتن + المصدر + الرقم + المحدث + الدرجة
    (احذف أي حقل غائب من السياق، لا تكتب «غير محدد»).
  • الترتيب: صحيح ← حسن ← ضعيف.
  • للضعيف والموضوع: نبّه بـ ⚠️ قبله.
  • لا تذكر أحاديث خارج السياق.
"""
)

SYSTEM_PROMPT_EXPLAIN_BRANCH5 = (
    "أنت متخصص في شرح الأحاديث النبوية. "
    "أنت لا تشرح إلا ما هو في السياق المقدم. "
    "لقد تحققت البوابة البرمجية من أن السؤال صالح — انتقل مباشرة إلى الإجابة."
    + _VERDICT_FIRST_RULE
    + _HARDCODING_PROHIBITION
    + _FEW_SHOT_EXAMPLES
    + """
════════════════════════════════════════
قواعد شرح الحديث
════════════════════════════════════════

الخطوة أ — هل الحديث موجود في السياق؟
  • إذا كانت الأحاديث في السياق لا تطابق الحديث المطلوب:
    → اكتب: «لم يُعثر على هذا الحديث في قاعدة بيانات الأحاديث المتاحة.»
    → اذكر الأحاديث المتاحة في السياق إن كانت وثيقة الصلة بالموضوع.
    → لا تشرح حديثاً لم يكن في السياق.

الخطوة ب — إذا كان موجوداً:
  1. قدّم إجابة وافية تربط بين سؤال المستخدم والأحاديث المسترجعة. بين الحكم الشرعي
     أو الفائدة المستفادة بشكل واضح ومباشر، ولا تكتفِ بسرد نص الحديث فقط.
     في القضايا الحساسة (مثل قضايا المرأة)، كن متعاطفاً وحكيماً، واشرح المقصد
     الشرعي بلطف وبصياغة متزنة.
  2. اذكر المتن الكامل كما ورد في السياق.
  3. اذكر الدرجة والمحدث (من السياق فقط).
  4. اشرح المعنى — فقط إذا وُجد شرح أو سياق يدعمه.
  5. اذكر المصدر والراوي (من السياق فقط).

الخطوة ج — للضعيف والموضوع:
  → اكتب: «⚠️ [الدرجة]: [الحكم من السياق]»
  → لا تبنِ عليه حكماً شرعياً.
"""
)

# ============================================================
# Legacy aliases — kept for backwards compat
# ============================================================

SYSTEM_PROMPT_GENERAL = SYSTEM_PROMPT_GENERAL_BRANCH5
SYSTEM_PROMPT_METADATA = SYSTEM_PROMPT_METADATA_BRANCH5
SYSTEM_PROMPT_NARRATOR = SYSTEM_PROMPT_NARRATOR_BRANCH5
SYSTEM_PROMPT_EXPLAIN  = SYSTEM_PROMPT_EXPLAIN_BRANCH5

INTENT_AR_LABELS = {
    AnswerIntent.EXPLANATORY: "إجابة تفسيرية أو تعليمية",
    AnswerIntent.VERIFICATION: "تحقق من صحة الحديث أو درجته",
    AnswerIntent.COLLECTION: "جمع شامل للروايات",
    AnswerIntent.LOOKUP: "بحث عن حديث بعينه",
}


def _build_intent_policy_prompt(answer_intent: AnswerIntent) -> str:
    common_rules = """
## قواعد الدرجات الملزمة:
1. اذكر درجة كل رواية تذكرها صراحة.
2. قدّم الصحيح ثم الحسن قبل غيرهما كلما أمكن.
3. الرواية الضعيفة أو الموضوعة أو غير المتحققة لا تُبنى عليها أحكام ولا فضائل ولا توجيه ديني.
4. إذا كانت درجة الرواية غير واضحة فعدّها «غير متحققة» ولا تستخدمها دليلاً.
5. إذا تعارضت الدرجة المختصرة مع الحكم التفصيلي فاعتبر الرواية غير صالحة للاحتجاج.
6. لا تستعمل روايات أحكام الزكاة وتوزيعها لإثبات فضائل الصدقة إلا إذا كان وجه الاستدلال صريحاً في السياق.
7. اكتب متن الجواب فقط — لا تضف أي سطور تقييم أو إحصاء خارجي.
8. إن كانت الأدلة المتاحة محدودة، استخدم صياغة حذرة وتجنب الجزم الزائد.
9. في الأسئلة المتعلقة بقضايا حساسة تخص فئات المجتمع (كالنساء) أو الأمور الشائكة،
   قدم الشرح بأدب وموضوعية وتعاطف، مبيناً المقاصد الشرعية بلطف واجتنب الأجوبة
   القصيرة الجافة التي قد توحي بالسلبية.
"""
    if answer_intent == AnswerIntent.EXPLANATORY:
        return common_rules + """
## سياسة الإجابة التفسيرية:
1. استعمل في الاستدلال الأحاديث الصحيحة والحسنة في المقام الأول.
2. إذا وُجدت روايات ضعيفة أو موضوعة أو غير متحققة ذات صلة، فاذكرها فقط بوصفها
   غير صالحة للاحتجاج مع بيان درجتها.
3. إذا لم يوجد في السياق حديث صريح في موضوع السؤال تحديداً، لكن توجد أحاديث ذات
   صلة وثيقة، فاستخدمها لبناء إجابة سياقية مفيدة مع التصريح بأنها ليست نصاً صريحاً
   في المسألة. لا تُخرج جواباً بسطر واحد «لا يوجد»  بينما السياق يحتوي أدلة مرتبطة.
4. إذا كان السياق خالياً تماماً من أي أحاديث مفيدة، فصرّح بذلك بوضوح واعذر السائل.
"""
    if answer_intent == AnswerIntent.VERIFICATION:
        return common_rules + """
## سياسة التحقق من الحديث:
1. اذكر درجة الحديث أولاً قبل أي شرح لمعناه.
2. يجوز ذكر الضعيف أو الموضوع أو غير المتحقق مع بيان درجته بوضوح.
3. إذا وُجد أكثر من حكم في السياق، فاذكره كما هو من غير خلط.
"""
    if answer_intent == AnswerIntent.COLLECTION:
        return common_rules + """
## سياسة الجمع الشامل للروايات:
1. يجوز عرض جميع الروايات المسترجعة.
2. افصل الروايات في الإجابة بحسب الدرجة: صحيح، حسن، ضعيف، موضوع، غير متحقق.
3. لا تخلط الروايات الضعيفة مع الأحاديث الصحيحة والحسنة في فقرة استدلال واحدة.
"""
    return common_rules + """
## سياسة البحث عن حديث بعينه:
1. ابدأ بأقرب الروايات مطابقة، مع ذكر درجة كل رواية.
2. إذا كانت الرواية ضعيفة أو موضوعة أو غير متحققة فصرّح بذلك قبل أي شرح.
3. عند تعدد النتائج، قدّم الصحيح ثم الحسن، ثم بيّن ما دونهما مع التحذير.
"""


# ============================================================
# Structured Response dataclasses
# ============================================================

@dataclass
class Citation:
    hadith_index: int
    hadith_id: str
    matn_snippet: str
    grade: str
    grade_ar: str
    masdar: str
    rawi: str
    muhaddith: str
    is_weak: bool = False


@dataclass
class IgnoredNarration:
    hadith_index: int
    hadith_id: str
    grade: str
    grade_ar: str
    reason: str
    matn_snippet: str


@dataclass
class EvidenceEvaluation:
    authenticity_of_evidence: str
    relevance_to_question: str
    final_sufficiency: str


@dataclass
class GeneratedResponse:
    answer: str                                              # user-facing, clean
    answer_debug: str = ""                                   # excluded-narration block (UI layer)
    citations: list[Citation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    grounding_verified: bool = False
    grounding_issues: list[str] = field(default_factory=list)
    raw_text: str = ""
    query_type: str = ""
    answer_intent: str = ""
    evidence_sufficient: bool = False
    authenticity_of_evidence: str = "insufficient"
    relevance_to_question: str = "weak"
    final_sufficiency: str = "insufficient"
    ignored_narrations: list[IgnoredNarration] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "answer_debug": self.answer_debug,
            "citations": [asdict(c) for c in self.citations],
            "warnings": self.warnings,
            "grounding_verified": self.grounding_verified,
            "grounding_issues": self.grounding_issues,
            "answer_intent": self.answer_intent,
            "evidence_sufficient": self.evidence_sufficient,
            "authenticity_of_evidence": self.authenticity_of_evidence,
            "relevance_to_question": self.relevance_to_question,
            "final_sufficiency": self.final_sufficiency,
            "ignored_narrations": [asdict(item) for item in self.ignored_narrations],
        }


# ============================================================
# Citation Grounding Verification
# ============================================================

def _verify_citation_grounding(
    answer_text: str,
    provided_hadiths: list[RetrievedHadith],
) -> tuple[bool, list[str]]:
    issues = []
    known_masdar = {h.masdar.strip() for h in provided_hadiths if h.masdar.strip()}

    source_pattern = re.compile(r"(?:المصدر[:\s]+|كتاب\s+)([\u0600-\u06FF\s]+?)(?:\s*[،,\.\-\n])")
    mentioned_sources = source_pattern.findall(answer_text)
    for source in mentioned_sources:
        source = source.strip()
        if source and not any(source in m for m in known_masdar):
            issues.append(f"مصدر غير موجود في السياق: {source}")

    prophet_said_pattern = re.compile(
        r"قال\s+(?:رسول\s+الله|النبي).*?[:\s]+[«\"](.*?)[»\"]", re.DOTALL
    )
    quoted_texts = prophet_said_pattern.findall(answer_text)
    for quoted in quoted_texts:
        quoted_clean = quoted.strip()[:50]
        if quoted_clean and not any(quoted_clean in h.text_ar for h in provided_hadiths):
            issues.append(f"نص مقتبس قد لا يطابق السياق: {quoted_clean}...")

    number_pattern = re.compile(r"(?:رقم|حديث\s+رقم|الصفحة)\s*[:\s]*(\d+)")
    mentioned_numbers = number_pattern.findall(answer_text)
    known_numbers = {h.safha_raqam.strip() for h in provided_hadiths if h.safha_raqam.strip()}
    for num in mentioned_numbers:
        if num and not any(num in n for n in known_numbers):
            issues.append(f"رقم حديث/صفحة غير موجود في السياق: {num}")

    return len(issues) == 0, issues


# ============================================================
# Hadith auditing helpers
# ============================================================

@dataclass
class _AuditedHadith:
    source_index: int
    hadith: RetrievedHadith
    canonical_grade: str
    grade_label: str
    is_authentic: bool
    is_directly_relevant: bool
    exclusion_reason: str = ""


_TASHKEEL    = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]+")
_TATWEEL     = re.compile(r"\u0640+")
_WHITESPACE  = re.compile(r"\s+")
_ALEF_VARIANTS = re.compile(r"[أإآٱ]")

_CHARITY_TERMS = ("صدقه", "الصدقه", "صدقة", "الصدقة", "انفاق", "الانفاق", "إنفاق", "تبرع")
_VIRTUE_TERMS  = ("فضل", "فضائل", "فوائد", "اجر", "أجر", "ثواب", "ترغيب", "منفعه", "منفعة")
_ZAKAT_LEGAL_TERMS = (
    "زكاه", "الزكاه", "زكاة", "الزكاة", "نصاب", "مصارف", "مصرف",
    "العاملين عليها", "ابن السبيل", "الفقراء", "المساكين",
    "صدقه الفطر", "صدقة الفطر",
)
_CHARITY_VIRTUE_TERMS = (
    "الصدقه برهان", "الصدقة برهان", "تطفئ الخطيئه", "تطفئ الخطيئة",
    "ظل", "اجر الصدقه", "أجر الصدقة", "فضل الصدقه", "فضل الصدقة",
)
_SOURCE_PRIORITY_RULES = (
    ("صحيح البخاري", 0), ("البخاري", 0), ("bukhari", 0),
    ("صحيح مسلم",   1), ("مسلم",    1), ("muslim",  1),
)


def _normalize_audit_text(text: str) -> str:
    text = _TASHKEEL.sub("", str(text or "").strip().lower())
    text = _TATWEEL.sub("", text)
    text = _ALEF_VARIANTS.sub("ا", text)
    return _WHITESPACE.sub(" ", text).strip()


def _source_priority(masdar: str) -> int:
    normalized = _normalize_audit_text(masdar)
    for needle, priority in _SOURCE_PRIORITY_RULES:
        if needle in normalized:
            return priority
    return 2


def _is_charity_virtue_query(query: str) -> bool:
    normalized = _normalize_audit_text(query)
    return any(t in normalized for t in _CHARITY_TERMS) and any(t in normalized for t in _VIRTUE_TERMS)


def _is_legal_zakat_narration(hadith: RetrievedHadith) -> bool:
    haystack = _normalize_audit_text(" ".join(
        p for p in (hadith.text_ar, hadith.category, hadith.subcategory_name, hadith.masdar) if p
    ))
    return any(t in haystack for t in _ZAKAT_LEGAL_TERMS) and not any(t in haystack for t in _CHARITY_VIRTUE_TERMS)


def _detect_topic_exclusion_reason(query: str, hadith: RetrievedHadith) -> str:
    if _is_charity_virtue_query(query) and _is_legal_zakat_narration(hadith):
        return "يتعلق بأحكام الزكاة أو مصارفها، لا بفضائل الصدقة وثوابها"
    return ""


def _audit_hadiths_for_answer(
    query: str,
    hadiths: list[RetrievedHadith],
) -> tuple[list[_AuditedHadith], list[IgnoredNarration]]:
    audited: list[_AuditedHadith] = []
    ignored: list[IgnoredNarration] = []

    for index, hadith in enumerate(hadiths, 1):
        grade_audit  = audit_grade(hadith.grade, hadith.grade_ar, hadith.ruling)
        canonical_grade = grade_audit.effective_bucket
        grade_label  = resolve_grade_label(hadith.grade, hadith.grade_ar, hadith.ruling)
        is_authentic = grade_audit.is_usable_for_evidence

        topic_reason       = _detect_topic_exclusion_reason(query, hadith)
        is_directly_relevant = is_authentic and not topic_reason
        exclusion_reason   = ""

        if not is_authentic:
            exclusion_reason = grade_audit.exclusion_reason
        elif topic_reason:
            exclusion_reason = topic_reason

        audited.append(_AuditedHadith(
            source_index=index, hadith=hadith,
            canonical_grade=canonical_grade, grade_label=grade_label,
            is_authentic=is_authentic, is_directly_relevant=is_directly_relevant,
            exclusion_reason=exclusion_reason,
        ))

        if exclusion_reason:
            ignored.append(IgnoredNarration(
                hadith_index=index, hadith_id=hadith.id,
                grade=canonical_grade, grade_ar=grade_label,
                reason=exclusion_reason, matn_snippet=(hadith.text_ar or "")[:120],
            ))

    return audited, ignored


def _evaluate_retrieved_evidence(
    audited_hadiths: list[_AuditedHadith],
    answer_intent: AnswerIntent,
) -> EvidenceEvaluation:
    authentic_hadiths = [i for i in audited_hadiths if i.is_authentic]
    direct_hadiths    = [i for i in audited_hadiths if i.is_directly_relevant]

    authenticity = "sufficient" if authentic_hadiths else "insufficient"
    relevance    = "direct" if direct_hadiths else ("partial" if authentic_hadiths else "weak")

    min_direct = 2 if answer_intent in {AnswerIntent.EXPLANATORY, AnswerIntent.COLLECTION} else 1
    if len(direct_hadiths) >= min_direct:
        final_sufficiency = "sufficient"
    elif authentic_hadiths:
        if answer_intent in {AnswerIntent.EXPLANATORY, AnswerIntent.COLLECTION}:
            final_sufficiency = "sufficient"
        else:
            final_sufficiency = "partial"
    else:
        final_sufficiency = "insufficient"

    return EvidenceEvaluation(
        authenticity_of_evidence=authenticity,
        relevance_to_question=relevance,
        final_sufficiency=final_sufficiency,
    )


def _build_citations(hadiths: list[RetrievedHadith]) -> list[Citation]:
    citations = []
    seen_ids: set[str] = set()
    for i, h in enumerate(hadiths, 1):
        if h.id in seen_ids:
            continue
        seen_ids.add(h.id)
        canonical_grade = resolve_grade_bucket(h.grade, h.grade_ar, h.ruling)
        grade_ar        = resolve_grade_label(h.grade, h.grade_ar, h.ruling)
        citations.append(Citation(
            hadith_index=i, hadith_id=h.id,
            matn_snippet=h.text_ar[:100] if h.text_ar else "",
            grade=canonical_grade, grade_ar=grade_ar,
            masdar=h.masdar, rawi=h.rawi, muhaddith=h.muhaddith,
            is_weak=canonical_grade in ("daif", "mawdu"),
        ))
    return citations


def _build_warning_text(grade: str, grade_ar: str) -> str:
    if grade in ("daif", "mawdu"):
        return f"⚠️ {grade_ar}: لا يُحتج به في إثبات الأحكام والفضائل"
    if grade == "unknown":
        return "⚠️ غير متحقق: لم تثبت درجته فلا يُستخدم دليلاً"
    return ""


def _order_hadiths_for_generation(
    hadiths: list[RetrievedHadith],
    answer_intent: AnswerIntent,
) -> list[RetrievedHadith]:
    indexed = list(enumerate(hadiths))
    indexed.sort(key=lambda item: (
        _source_priority(item[1].masdar),
        grade_priority(resolve_grade_bucket(item[1].grade, item[1].grade_ar, item[1].ruling)),
        item[0],
    ))
    return [hadith for _, hadith in indexed]


def _normalize_hadith_text_for_dedup(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", normalized)
    normalized = re.sub(r"[أإآٱ]", "ا", normalized)
    return re.sub(r"\s+", " ", normalized)


def _tokenize_for_similarity(text: str) -> set[str]:
    tokens = set()
    for token in _normalize_hadith_text_for_dedup(text).split():
        cleaned = re.sub(r"[^\u0600-\u06FFa-z0-9]", "", token)
        if len(cleaned) >= 3:
            tokens.add(cleaned)
    return tokens


def _token_jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _token_overlap_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _are_near_duplicate_narrations(text_a: str, text_b: str) -> bool:
    norm_a = _normalize_hadith_text_for_dedup(text_a)
    norm_b = _normalize_hadith_text_for_dedup(text_b)
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    if SequenceMatcher(None, norm_a, norm_b).ratio() >= 0.72:
        return True
    anchor_phrases = (
        "ناقصات عقل ودين", "شهادة امراتين", "نقصان عقل",
        "نقصان دين", "تكثرن اللعن", "تكفرن العشير",
    )
    if sum(1 for p in anchor_phrases if p in norm_a and p in norm_b) >= 2:
        return True
    ta, tb = _tokenize_for_similarity(norm_a), _tokenize_for_similarity(norm_b)
    overlap = _token_overlap_similarity(ta, tb)
    if overlap >= 0.72:
        return True
    return overlap >= 0.62 and _token_jaccard_similarity(ta, tb) >= 0.45


def _hadith_representative_rank(hadith: RetrievedHadith) -> tuple:
    return (
        _source_priority(hadith.masdar),
        grade_priority(resolve_grade_bucket(hadith.grade, hadith.grade_ar, hadith.ruling)),
        float(hadith.distance or 1.0),
        -len(str(hadith.text_ar or "")),
    )


def _deduplicate_hadiths_for_answer(hadiths: list[RetrievedHadith]) -> list[RetrievedHadith]:
    if not hadiths:
        return []
    clusters: list[list[RetrievedHadith]] = []
    for hadith in hadiths:
        assigned = False
        for cluster in clusters:
            if _are_near_duplicate_narrations(hadith.text_ar, cluster[0].text_ar):
                cluster.append(hadith)
                assigned = True
                break
        if not assigned:
            clusters.append([hadith])
    return [min(c, key=_hadith_representative_rank) for c in clusters]


def _format_hadith_block(index: int, hadith: RetrievedHadith, metadata_first: bool = False) -> str:
    canonical_grade = resolve_grade_bucket(hadith.grade, hadith.grade_ar, hadith.ruling)
    grade_label     = resolve_grade_label(hadith.grade, hadith.grade_ar, hadith.ruling)
    warning         = _build_warning_text(canonical_grade, grade_label)
    warning_line    = f"\n   {warning}" if warning else ""

    if metadata_first:
        return (
            f"=== الحديث [{index}] ==={warning_line}\n"
            f"   📋 الراوي: {hadith.rawi}\n"
            f"   📋 المحدِّث: {hadith.muhaddith}\n"
            f"   📋 الدرجة: {grade_label}\n"
            f"   📋 الحكم التفصيلي: {hadith.ruling}\n"
            f"   📋 المصدر: {hadith.masdar}\n"
            f"   📋 الرقم/الصفحة: {hadith.safha_raqam}\n"
            f"   📋 التصنيف: {hadith.category} — {hadith.subcategory_name}\n"
            f"   📋 المتن: {hadith.text_ar}"
        )
    return (
        f"--- الحديث [{index}] ---{warning_line}\n"
        f"   المتن: {hadith.text_ar}\n"
        f"   الدرجة: {grade_label}\n"
        f"   الحكم التفصيلي: {hadith.ruling}\n"
        f"   الراوي: {hadith.rawi}\n"
        f"   المحدِّث: {hadith.muhaddith}\n"
        f"   المصدر: {hadith.masdar}\n"
        f"   الرقم/الصفحة: {hadith.safha_raqam}\n"
        f"   التصنيف: {hadith.category} — {hadith.subcategory_name}"
    )


def _group_hadiths_by_grade(hadiths: list[RetrievedHadith]) -> dict[str, list[RetrievedHadith]]:
    groups = {g: [] for g in ("sahih", "hasan", "daif", "mawdu", "unknown")}
    for hadith in hadiths:
        groups[resolve_grade_bucket(hadith.grade, hadith.grade_ar, hadith.ruling)].append(hadith)
    return groups


def _format_grouped_sections(
    sections: list[tuple[str, list[RetrievedHadith]]],
    metadata_first: bool = False,
) -> str:
    rendered_sections = []
    current_index = 1
    for title, section_hadiths in sections:
        if not section_hadiths:
            continue
        rendered_sections.append(title)
        blocks = []
        for hadith in section_hadiths:
            blocks.append(_format_hadith_block(current_index, hadith, metadata_first=metadata_first))
            current_index += 1
        rendered_sections.append("\n\n".join(blocks))
    return "\n\n".join(rendered_sections)


def _format_hadith_context(hadiths: list[RetrievedHadith], answer_intent: AnswerIntent) -> str:
    if answer_intent == AnswerIntent.EXPLANATORY:
        groups       = _group_hadiths_by_grade(hadiths)
        authentic    = groups["sahih"] + groups["hasan"]
        non_evidence = groups["daif"] + groups["mawdu"] + groups["unknown"]

        sections = []
        if authentic:
            sections.append(("### الأحاديث الصحيحة والحسنة (يُستدل بها)", authentic))
        else:
            sections.append(("### تنبيه: لا يوجد في النتائج حديث صحيح أو حسن يمكن الاستدلال به مباشرة", []))
        if non_evidence:
            sections.append(("### روايات غير صالحة للاحتجاج (ضعيف أو موضوع أو غير متحقق)", non_evidence))
        return _format_grouped_sections(sections)

    if answer_intent == AnswerIntent.COLLECTION:
        groups   = _group_hadiths_by_grade(hadiths)
        sections = [
            ("### الأحاديث الصحيحة",      groups["sahih"]),
            ("### الأحاديث الحسنة",       groups["hasan"]),
            ("### الأحاديث الضعيفة",      groups["daif"]),
            ("### الأحاديث الموضوعة",     groups["mawdu"]),
            ("### الروايات غير المتحققة", groups["unknown"]),
        ]
        return _format_grouped_sections(sections)

    return "\n\n".join(_format_hadith_block(i, h) for i, h in enumerate(hadiths, 1))


def _format_metadata_context(hadiths: list[RetrievedHadith], answer_intent: AnswerIntent) -> str:
    if answer_intent in {AnswerIntent.EXPLANATORY, AnswerIntent.COLLECTION}:
        groups   = _group_hadiths_by_grade(hadiths)
        sections = [
            ("### الأحاديث الصحيحة",      groups["sahih"]),
            ("### الأحاديث الحسنة",       groups["hasan"]),
            ("### الأحاديث الضعيفة",      groups["daif"]),
            ("### الأحاديث الموضوعة",     groups["mawdu"]),
            ("### الروايات غير المتحققة", groups["unknown"]),
        ]
        return _format_grouped_sections(sections, metadata_first=True)
    return "\n\n".join(_format_hadith_block(i, h, metadata_first=True) for i, h in enumerate(hadiths, 1))


def _format_ignored_narrations(ignored_narrations: list[IgnoredNarration]) -> str:
    if not ignored_narrations:
        return "لا توجد روايات مستبعدة."
    lines = []
    for item in ignored_narrations:
        snippet = item.matn_snippet.strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        lines.append(f"{item.hadith_index}. الدرجة: {item.grade_ar} | السبب: {item.reason} | النص: {snippet}")
    return "\n".join(lines)


def _wrap_audited_answer(
    evaluation: EvidenceEvaluation,
    core_answer: str,
    ignored_narrations: list[IgnoredNarration],
) -> tuple[str, str]:
    """
    v7 FIX C: Returns (clean_answer, debug_block) as separate strings.

    clean_answer  → user-facing content, free of excluded-narration noise.
                    Stored in GeneratedResponse.answer.
    debug_block   → formatted excluded-narration warnings for the UI layer.
                    Stored in GeneratedResponse.answer_debug.
                    The UI decides whether/where to render this.
    """
    clean_answer = core_answer.strip()
    debug_block  = ""

    if ignored_narrations:
        lines = []
        for item in ignored_narrations:
            snippet = (item.matn_snippet or "").strip()
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            lines.append(
                f"  • ⚠️ رواية مستبعدة [{item.hadith_index}] — "
                f"{item.grade_ar}: {item.reason}"
                + (f"\n    النص: {snippet}" if snippet else "")
            )
        debug_block = "**الروايات المستبعدة من الاستدلال:**\n" + "\n".join(lines)

    return clean_answer, debug_block


def _select_system_prompt(query_type: str, answer_intent: AnswerIntent) -> str:
    """
    Always returns a BRANCH5-only system prompt — _DECISION_TREE_GATE is
    intentionally omitted from all variants here. The gate is enforced in
    Python by _apply_decision_tree_gate() before this function is ever reached.
    Including the gate in the LLM prompt creates a second unreliable gate that
    mis-fires on valid fiqh topics (زنا, حيض, طلاق, ردة, …).

    _VERDICT_FIRST_RULE is included in all variants (v7).
    """
    if answer_intent == AnswerIntent.EXPLANATORY:
        return SYSTEM_PROMPT_GENERAL_BRANCH5
    if query_type == "metadata":
        return SYSTEM_PROMPT_METADATA_BRANCH5
    elif query_type == "narrator":
        return SYSTEM_PROMPT_NARRATOR_BRANCH5
    elif query_type == "explain_hadith":
        return SYSTEM_PROMPT_EXPLAIN_BRANCH5
    else:
        return SYSTEM_PROMPT_GENERAL_BRANCH5


def _check_hadith_relevance(
    requested_text: str,
    hadiths: list[RetrievedHadith],
    min_overlap_chars: int = 4,
) -> bool:
    if not requested_text or not hadiths:
        return False

    def _norm(s: str) -> str:
        s = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]+", "", s)
        s = re.sub(r"\u0640+", "", s)
        s = re.sub(r"[أإآٱ]", "ا", s)
        s = s.replace("ة", "ه")
        return re.sub(r"\s+", " ", s).strip()

    norm_req  = _norm(requested_text)
    key_words = [w for w in norm_req.split() if len(w) >= min_overlap_chars]
    if not key_words:
        return True

    for hadith in hadiths:
        norm_matn = _norm(hadith.text_ar or "")
        matches   = sum(1 for w in key_words if w in norm_matn)
        if matches >= max(1, len(key_words) // 2):
            return True
    return False


# ============================================================
# Generator
# ============================================================

class HadithGenerator:
    """
    Generates answers using Gemini API with Arabic system prompt.

    Gate contract (v7):
      • Python gate (_apply_decision_tree_gate) enforces BRANCHES 1-4.
      • _filter_offtopic_hadiths() removes hadiths with no surface token overlap.
      • _filter_answer_irrelevant_hadiths() removes hadiths that share a surface
        token with the query but address a different legal topic (v7 NEW).
      • LLM is called ONLY for BRANCH 5 (sufficient evidence, valid query).
      • System prompts are BRANCH5-only variants — no _DECISION_TREE_GATE.
      • _VERDICT_FIRST_RULE is included in all system prompts (v7 NEW).
      • _BRANCH5_GATE_SUPPRESSION is appended to user_message.
      • insufficient evidence → _REFUSAL_NO_CONTEXT, no LLM call.
      • partial evidence → LLM called with contextual-answer instruction.
      • _wrap_audited_answer() returns (clean_answer, debug_block) (v7 NEW).
        GeneratedResponse.answer is now always clean/user-facing.
        GeneratedResponse.answer_debug carries the excluded-narration block.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key    = api_key or settings.GEMINI_API_KEY
        self.model_name = model or settings.GEMINI_MODEL

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Set it in .env or pass it directly. "
                "Get your FREE key at https://aistudio.google.com/apikey"
            )

        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"Gemini generator initialized: model={self.model_name}")

    def _make_gate_response(
        self,
        refusal: str,
        query_type: str,
        answer_intent: AnswerIntent,
    ) -> GeneratedResponse:
        """Wrap a gate refusal in a GeneratedResponse."""
        return GeneratedResponse(
            answer=refusal,
            answer_debug="",
            grounding_verified=True,
            raw_text=refusal,
            query_type=query_type,
            answer_intent=answer_intent.value,
            evidence_sufficient=False,
            authenticity_of_evidence="insufficient",
            relevance_to_question="weak",
            final_sufficiency="insufficient",
        )

    def generate(
        self,
        query: str,
        hadiths: list[RetrievedHadith],
        temperature: float = 0.3,
        max_output_tokens: int = 4096,
        verify_grounding: bool = True,
        query_type: str = "general",
        metadata_fields: list[str] | None = None,
        excluded_masdar: list[str] | None = None,
    ) -> GeneratedResponse:

        # ── Classify intent once — used everywhere below ──────────────────────
        answer_intent = classify_answer_intent(
            query=query,
            query_type=query_type,
            metadata_fields=metadata_fields,
        )

        # ── Enforce decision tree in Python before touching the LLM ──────────
        gate_refusal = _apply_decision_tree_gate(query, hadiths)
        if gate_refusal is not None:
            logger.info(f"Decision gate fired for query: {query!r:.80} → {gate_refusal!r:.60}")
            return self._make_gate_response(gate_refusal, query_type, answer_intent)

        # ── Guard: non-empty list guaranteed by gate, but handle edge case ────
        if not hadiths:
            logger.info("No hadiths after gate — returning hard refusal.")
            return self._make_gate_response(_REFUSAL_NO_CONTEXT, query_type, answer_intent)

        # ── Pass 1: remove hadiths with no surface token overlap ──────────────
        hadiths = _filter_offtopic_hadiths(query, hadiths)

        # ── Pass 2 (v7): remove hadiths that share a surface token but address
        #    a different legal topic from the answer target ────────────────────
        hadiths = _filter_answer_irrelevant_hadiths(query, hadiths)

        audited_hadiths, ignored_narrations = _audit_hadiths_for_answer(query, hadiths)
        evaluation      = _evaluate_retrieved_evidence(audited_hadiths, answer_intent)
        direct_hadiths  = [item.hadith for item in audited_hadiths if item.is_directly_relevant]

        authentic_hadiths = [item.hadith for item in audited_hadiths if item.is_authentic]
        hadiths_for_generation = direct_hadiths if direct_hadiths else authentic_hadiths

        deduplicated_hadiths = _deduplicate_hadiths_for_answer(hadiths_for_generation)
        ordered_hadiths      = _order_hadiths_for_generation(deduplicated_hadiths, answer_intent)

        citations = _build_citations(ordered_hadiths)
        warnings  = [
            f"⚠️ استُبعد الحديث [{item.hadith_index}] — {item.reason}"
            for item in ignored_narrations
        ]

        # ── Insufficient evidence → hard refusal, no LLM call ─────────────────
        if evaluation.final_sufficiency == "insufficient":
            logger.info("Evidence insufficient — returning hard refusal without LLM call.")
            return GeneratedResponse(
                answer=_REFUSAL_NO_CONTEXT,
                answer_debug="",
                citations=citations,
                warnings=warnings,
                grounding_verified=True,
                raw_text=_REFUSAL_NO_CONTEXT,
                query_type=query_type,
                answer_intent=answer_intent.value,
                evidence_sufficient=False,
                authenticity_of_evidence=evaluation.authenticity_of_evidence,
                relevance_to_question=evaluation.relevance_to_question,
                final_sufficiency=evaluation.final_sufficiency,
                ignored_narrations=ignored_narrations,
            )

        # ── Select BRANCH5-only system prompt (includes _VERDICT_FIRST_RULE) ──
        system_prompt        = _select_system_prompt(query_type, answer_intent)
        intent_policy_prompt = _build_intent_policy_prompt(answer_intent)

        if query_type == "metadata":
            context     = _format_metadata_context(ordered_hadiths, answer_intent)
            temperature = min(temperature, 0.1)
        else:
            context = _format_hadith_context(ordered_hadiths, answer_intent)

        # ── Build user message ────────────────────────────────────────────────
        user_message = (
            f"## السياق (الأحاديث المسترجعة):\n{context}\n\n"
            f"## سؤال المستخدم:\n{query}"
        )

        # ── Suppress LLM re-evaluation of the decision tree ──────────────────
        user_message += _BRANCH5_GATE_SUPPRESSION

        # ── Hint when evidence is partial (related but not exact topic) ───────
        if evaluation.relevance_to_question == "partial":
            user_message += (
                "\n\n## ملاحظة حول السياق:\n"
                "الأحاديث المسترجعة لا تتناول موضوع السؤال بصورة صريحة ومباشرة، "
                "غير أنها وثيقة الصلة بالموضوع. "
                "استخدمها لبناء إجابة سياقية مفيدة، وصرّح بأن النص الصريح غير متوفر "
                "في هذه القاعدة. لا تكتفِ بسطر رفض واحد."
            )

        duplicate_collapsed_count = max(0, len(hadiths_for_generation) - len(ordered_hadiths))
        if duplicate_collapsed_count > 0:
            user_message += (
                "\n\n## تنبيه مهم لأسلوب العرض:\n"
                "بعض النتائج كانت روايات متقاربة جداً في المعنى والنص لنفس الخبر. "
                "لا تعرضها كأحاديث مستقلة متعددة أمام المستخدم. "
                "اكتف بتمثيل موجز غير مُرقم على أنها روايات متعددة لخبر واحد عند اللزوم، "
                "واذكر فقط الروايات الأوضح والأقوى دون تعدادٍ طويل."
            )

        if query_type in {"explain_hadith", "metadata", "hadith_lookup", "ruling"} and answer_intent == AnswerIntent.LOOKUP:
            from retrieval.query_preprocessor import _extract_hadith_text_from_explain_query, _extract_hadith_text_from_metadata_query
            norm_q = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u0640]", "", query)
            norm_q = re.sub(r"[أإآٱ]", "ا", norm_q)
            norm_q = re.sub(r"\s+", " ", norm_q).strip()

            if query_type == "explain_hadith":
                requested_text = _extract_hadith_text_from_explain_query(norm_q)
            else:
                requested_text = _extract_hadith_text_from_metadata_query(norm_q)

            corpus_has_it = _check_hadith_relevance(requested_text, ordered_hadiths)
            if not corpus_has_it:
                logger.info(f"Gate: BRANCH 5 (requested hadith '{requested_text}' not found in context)")
                refusal = f"لم يتم العثور على الحديث «{requested_text}» في قاعدة البيانات المتاحة حالياً."
                return self._make_gate_response(refusal, query_type, answer_intent)

        if metadata_fields:
            fields_ar = {
                "rawi": "الراوي", "grade": "الدرجة/الصحة",
                "masdar": "المصدر/الكتاب", "safha_raqam": "الرقم/الصفحة",
                "muhaddith": "المحدث", "category": "التصنيف/الباب",
            }
            requested    = [fields_ar.get(f, f) for f in metadata_fields]
            user_message += f"\n\n## تنبيه: المستخدم يسأل تحديداً عن: {', '.join(requested)}"

        if excluded_masdar:
            excluded_str  = ", ".join(f"«{b}»" for b in excluded_masdar)
            user_message += (
                f"\n\n## ⚠️ تعليمات الإقصاء:\n"
                f"المستخدم يريد فقط الأحاديث التي لم تُذكر في: {excluded_str}.\n"
                f"افحص كل حديث في السياق: إذا كانت قاعدة البيانات لا تتضمن معلومات مقارنة بين الكتب، "
                f"فأخبر المستخدم بذلك وعرض الأحاديث المسترجعة مع هذا التوضيح."
            )

        logger.info(
            f"Generating answer: model={self.model_name}, "
            f"hadiths={len(ordered_hadiths)}, query_type={query_type}, "
            f"answer_intent={answer_intent.value}, "
            f"evaluation={evaluation.final_sufficiency}, "
            f"relevance={evaluation.relevance_to_question}, temp={temperature}"
        )

        merged_prompt = (
            f"## تعليمات النظام:\n{system_prompt}\n\n"
            f"## سياسة الإجابة بحسب نية السؤال:\n"
            f"التصنيف الداخلي: {INTENT_AR_LABELS[answer_intent]}\n"
            f"{intent_policy_prompt}\n\n"
            f"## رسالة المستخدم:\n{user_message}"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=merged_prompt,
            config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens),
        )

        core_answer = response.text or ""
        logger.info(f"Generation complete: {len(core_answer)} chars")

        grounding_verified, grounding_issues = True, []
        if verify_grounding:
            grounding_verified, grounding_issues = _verify_citation_grounding(
                core_answer, ordered_hadiths
            )
            if not grounding_verified:
                logger.warning(f"Grounding issues detected: {grounding_issues}")

        # v7 FIX C: clean_answer is user-facing; debug_block goes to answer_debug
        clean_answer, debug_block = _wrap_audited_answer(
            evaluation=evaluation,
            core_answer=core_answer,
            ignored_narrations=ignored_narrations,
        )

        return GeneratedResponse(
            answer=clean_answer,
            answer_debug=debug_block,
            citations=citations,
            warnings=warnings,
            grounding_verified=grounding_verified,
            grounding_issues=grounding_issues,
            raw_text=core_answer,
            query_type=query_type,
            answer_intent=answer_intent.value,
            evidence_sufficient=True,
            authenticity_of_evidence=evaluation.authenticity_of_evidence,
            relevance_to_question=evaluation.relevance_to_question,
            final_sufficiency=evaluation.final_sufficiency,
            ignored_narrations=ignored_narrations,
        )


# ============================================================
# Module-level convenience
# ============================================================

_generator: Optional[HadithGenerator] = None


def get_generator() -> HadithGenerator:
    global _generator
    if _generator is None:
        _generator = HadithGenerator()
    return _generator


def generate(
    query: str,
    hadiths: list[RetrievedHadith],
    temperature: float = 0.3,
    query_type: str = "general",
    metadata_fields: list[str] | None = None,
    excluded_masdar: list[str] | None = None,
) -> GeneratedResponse:
    return get_generator().generate(
        query=query,
        hadiths=hadiths,
        temperature=temperature,
        query_type=query_type,
        metadata_fields=metadata_fields,
        excluded_masdar=excluded_masdar,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Generator ready. Model: {settings.GEMINI_MODEL}")
    print("System prompt (general/branch5) length:", len(SYSTEM_PROMPT_GENERAL_BRANCH5), "chars")
    print("System prompt (metadata/branch5) length:", len(SYSTEM_PROMPT_METADATA_BRANCH5), "chars")
    print("System prompt (narrator/branch5) length:", len(SYSTEM_PROMPT_NARRATOR_BRANCH5), "chars")
    print("System prompt (explain/branch5) length:", len(SYSTEM_PROMPT_EXPLAIN_BRANCH5), "chars")
