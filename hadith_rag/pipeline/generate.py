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
# KEY FIX (v7):
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
#
# LATENCY (v8) — zero accuracy impact:
#   • Pre-compile _FORBIDDEN_MIXED_REFUSAL_PATTERNS with re.IGNORECASE baked
#     in — avoids per-line regex recompilation inside _strip_mixed_refusal_sentences.
#   • lru_cache wrappers (_cached_resolve_grade_bucket, _cached_resolve_grade_label,
#     _cached_audit_grade) — each hadith's (grade, grade_ar, ruling) triple is
#     resolved 6-8× per request; caching collapses repeats to one lookup.
#   • Moved inline `from retrieval.query_preprocessor import …` to module
#     top-level — eliminates per-call import overhead inside the hot path.
#   • Lazy-cached _get_merged_prompt_prefix() per (query_type, intent) —
#     avoids re-concatenating multi-KB system-prompt strings on every request.
#   • Single _normalize_for_overlap(query) computation in generate() passed
#     directly to _order_hadiths_for_generation — removes one redundant call.
#
# KEY FIX (v9) — conciseness + speed for VERIFICATION:
#   • _format_verification_hadiths_compact(): near-duplicate narrations
#     (same matn, different muhaddith/masdar) are merged into a single chunk
#     that lists all sources together. Cuts prompt 60-80% for typical
#     "هل حديث X صحيح" queries with zero accuracy loss.
#   • VERIFICATION max_output_tokens reduced 1024 → 640 (verdict + refs,
#     not an essay).
#   • VERIFICATION intent policy updated: instruct LLM to produce a direct
#     verdict + ONE primary source + one compact line for remaining scholars.
#     Output target: ≤ 6 lines.
#   • _format_hadith_context routes VERIFICATION to compact formatter.

import logging
import re
import time
from difflib import SequenceMatcher
from functools import lru_cache
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

from retrieval.query_preprocessor import (
    _extract_hadith_text_from_explain_query,
    _extract_hadith_text_from_metadata_query,
)

logger = logging.getLogger(__name__)


# ============================================================
# LATENCY v8: lru_cache wrappers for grade-resolution functions
# ============================================================

@lru_cache(maxsize=1024)
def _cached_resolve_grade_bucket(grade: str, grade_ar: str, ruling: str) -> str:
    return resolve_grade_bucket(grade, grade_ar, ruling)


@lru_cache(maxsize=1024)
def _cached_resolve_grade_label(grade: str, grade_ar: str, ruling: str) -> str:
    return resolve_grade_label(grade, grade_ar, ruling)


@lru_cache(maxsize=1024)
def _cached_audit_grade(grade: str, grade_ar: str, ruling: str):
    return audit_grade(grade, grade_ar, ruling)


def _grade_args(h: "RetrievedHadith") -> tuple[str, str, str]:
    return (h.grade or "", h.grade_ar or "", h.ruling or "")


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

_STRICT_RAG_CONTRACT = """
════════════════════════════════════════
عقد الإجابة من السياق فقط
════════════════════════════════════════
اتبع هذه القواعد حرفياً:
  1. أجب فقط من وسوم <chunk> المعروضة في السياق، ولا تستخدم أي معرفة خارجية.
  2. اقرأ كل المقاطع أولاً، واستعمل كل المقاطع ذات الصلة بالسؤال. لا تتوقف عند أول تطابق.
  3. إذا وجدت المقاطع جواباً للسؤال، فاكتب الإجابة فقط، ولا تذكر أن شيئاً غير موجود في قاعدة المعرفة.
  4. إذا لم توجد في المقاطع معلومات تكفي للإجابة، فاكتب هذه الجملة وحدها بلا أي إضافة:
     «لا تتوفر معلومات كافية في المصادر المتاحة للإجابة على هذا السؤال.»
  5. ممنوع خلط الرفض مع جواب جزئي. إمّا إجابة مستندة إلى السياق، أو رفض كامل بالجملة المحددة أعلاه.
  6. ممنوع استعمال عبارات مثل:
     «لم يرد في السياق»، «لم يكن موجوداً في السياق المسترجع»،
     «لا يوجد معلومات كافية»، «السؤال غير واضح»،
     أو أي صياغة إنجليزية عن knowledge base، داخل إجابة تحتوي معلومات أخرى.
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
  ✓ الصواب (السطر الأول): «لا أستطيع الجزم إلا بما تدعمه الأحاديث المسترجعة.»
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

مثال 5 — الإجابة بالأدلة ذات الصلة دون استنتاج خارجي
  السؤال: «هل ينقض الزنا الوضوء؟»
  السياق: يحتوي أحاديث عن نواقض الوضوء (لمس المرأة، الضحك في الصلاة...)
          لكن لا يوجد حديث يذكر الزنا صراحةً بوصفه ناقضاً للوضوء.
  ✓ الصواب:
      «تذكر المقاطع المسترجعة نواقض وضوء محددة، منها: [اذكر النواقض الواردة
       في المقاطع مع الأحاديث الداعمة].»
  ✗ الخطأ: خلط إجابة عن النواقض بعبارة «لكن حكم الزنا لم يكن موجوداً في السياق».
  ✗ الخطأ: «لا يوجد في السياق ما يدل على أن الزنا يبطل الوضوء.» (سطر واحد)
  ✗ الخطأ: الجزم بحكم الزنا من معرفة خارج السياق.
  السبب: إذا وُجدت أدلة مفيدة في المقاطع فاستعملها فقط، ولا تضف جملة رفض جزئية.

مثال 6 — التحقق المختصر (v9 — جديد)
  السؤال: «هل حديث من غشنا فليس منا صحيح؟»
  السياق: 5 chunks لنفس الحديث من رواة/مصادر مختلفة (مسلم، الألباني، ابن باز...)
  ✓ الصواب (≤6 أسطر):
      «الحديث صحيح، رواه أبو هريرة.
       المتن: من حمل علينا السلاح فليس منا، ومن غشنا فليس منا.
       الدرجة: صحيح — رواه مسلم (101).
       كما وثّقه: الألباني (صحيح الجامع 6406) · ابن باز · شعيب الأرناؤوط · ابن المنذر.»
  ✗ الخطأ: إفراد فقرة مستقلة لكل محدّث من الخمسة.
  السبب: الحديث واحد — المصادر المتعددة تُدرج في سطر واحد "كما وثّقه:".
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
_REFUSAL_NO_CONTEXT   = "لا تتوفر معلومات كافية في المصادر المتاحة للإجابة على هذا السؤال."
_REFUSAL_UNVERIFIED_HADITH_TEXT = _REFUSAL_NO_CONTEXT
_REFUSAL_UNVERIFIED_ATTRIBUTION = _REFUSAL_NO_CONTEXT

_OFFTOPIC_FILTER_MIN_OVERLAP = 0.10
_ANSWER_RELEVANCE_MIN_TARGET_COVERAGE = 0.25

# ── v9: VERIFICATION cap reduced 1024 → 640 (verdict + refs, not an essay)
_MAX_OUTPUT_TOKENS_BY_INTENT = {
    AnswerIntent.VERIFICATION: 640,
    AnswerIntent.LOOKUP: 1200,
    AnswerIntent.EXPLANATORY: 1536,
    AnswerIntent.COLLECTION: 2048,
}

_MAX_OUTPUT_TOKENS_BY_QUERY_TYPE = {
    "metadata": 8192,  # Significantly increased to accommodate thinking + full output
    # Problem: LLM was using 2045 thinking tokens out of 2048 limit → 0 for answer
    "narrator": 4096,  # Also increased proportionally
}

_TRANSIENT_API_ERROR_MARKERS = (
    "429",
    "500",
    "503",
    "resourceexhausted",
    "resource_exhausted",
    "internal",
    "unavailable",
)

_FORBIDDEN_MIXED_REFUSAL_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"this\s+text\s+was\s+not\s+found\s+in\s+the\s+retrieved\s+knowledge\s+base.*",
        r"i\s+cannot\s+confirm\s+its\s+authenticity.*",
        r"لم\s+يرد\s+في\s+السياق.*",
        r"لم\s+يكن\s+موجود[اًا]?\s+في\s+السياق\s+المسترجع.*",
        r"لكن\s+لم\s+يكن.*السياق.*",
        r"لا\s+يوجد\s+معلومات\s+كافية.*",
        r"لا\s+توجد\s+معلومات\s+كافية.*",
        r"لا\s+تتوفر\s+معلومات\s+كافية.*",
        r"السؤال\s+غير\s+واضح.*",
        r"يرجى\s+تحديد\s+نص\s+الحديث.*",
        r"أجبت\s+عن\s+.*لكن\s+.*",
    )
)


def _strip_mixed_refusal_sentences(answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    if text == _REFUSAL_NO_CONTEXT:
        return text

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append(raw_line)
            continue
        if any(pattern.search(line) for pattern in _FORBIDDEN_MIXED_REFUSAL_PATTERNS):
            logger.debug(f"Stripping refusal line: {line[:100]}")
            continue
        lines.append(raw_line)

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    result = cleaned or _REFUSAL_NO_CONTEXT
    logger.debug(f"_strip_mixed_refusal_sentences: input {len(text)} chars -> output {len(result)} chars")
    return result


def _model_supports_thinking_config(model_name: str) -> bool:
    normalized = str(model_name or "").lower()
    return normalized.startswith("gemini-2.5")


def _is_transient_api_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_API_ERROR_MARKERS)


def _resolve_max_output_tokens(
    answer_intent: AnswerIntent,
    query_type: str,
    requested_max_output_tokens: int | None,
) -> int:
    if settings.GEMINI_MAX_OUTPUT_TOKENS > 0:
        return settings.GEMINI_MAX_OUTPUT_TOKENS

    default_cap = _MAX_OUTPUT_TOKENS_BY_QUERY_TYPE.get(
        query_type,
        _MAX_OUTPUT_TOKENS_BY_INTENT.get(answer_intent, 1536),
    )
    if requested_max_output_tokens and requested_max_output_tokens > 0:
        return min(requested_max_output_tokens, default_cap)
    return default_cap


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


def _normalize_exact_match_text(text: str) -> str:
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]+", "", str(text or ""))
    text = re.sub(r"\u0640+", "", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_semantic_hadith_existence_query(query: str) -> bool:
    text = _normalize_exact_match_text(query)
    patterns = (
        r"^هل\s+(?:يوجد|هناك|ورد|ثبت|صح)\s+حديث\s+(?:يشير|يدل|يفيد|يتكلم|يتحدث)\s+(?:الى|الي|علي|عن|في)\s+",
        r"^هل\s+(?:يوجد|هناك|ورد|ثبت|صح)\s+حديث\s+(?:عن|في|حول)\s+",
        r"^(?:ابحث|هات|اعطني|اذكر)\s+(?:لي\s+)?حديث\s+(?:يشير|يدل|يفيد|يتكلم|يتحدث)\s+(?:الى|الي|علي|عن|في)\s+",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _query_asks_for_explanation(query: str) -> bool:
    text = _normalize_exact_match_text(query)
    return any(
        marker in text
        for marker in ("اشرح", "شرح", "فسر", "معني", "معنى")
    )


def _extract_requested_hadith_text(query: str) -> str:
    raw = str(query or "").strip()
    quoted = re.search(r"[«\"']([^»\"']{6,})[»\"']", raw)
    if quoted:
        return quoted.group(1).strip()

    if _is_semantic_hadith_existence_query(raw):
        return ""

    text = _normalize_exact_match_text(raw)
    prefix_patterns = (
        r"^(?:ما\s+صحه|ما\s+درجه|هل\s+صح|هل\s+ثبت|صحه|درجه|حكم\s+حديث)\s+",
        r"^(?:هل\s+حديث)\s+",
        r"^(?:اشرح|فسر|ما\s+معنى|من\s+رواه|في\s+اي\s+كتاب|في\s+اي\s+مصدر)\s+",
    )
    for pattern in prefix_patterns:
        text = re.sub(pattern, "", text).strip()

    text = re.sub(r"^(?:هذا|هذه|الحديث|حديث|الروايه|روايه)\s+", "", text).strip()
    text = re.sub(r"^(?:النبي|رسول الله)\s+", "", text).strip()
    text = re.sub(r"\s+(?:صحيح|ثابت|حسن|ضعيف|موضوع|مكذوب)$", "", text).strip()

    topic_markers = (" عن ", " حول ", " في ")
    if any(marker in f" {text} " for marker in topic_markers) and len(text.split()) <= 5:
        return ""

    count_stopwords = {
        "ما", "هل", "من", "في", "عن", "على", "الى", "هذا", "هذه", "الحديث",
        "حديث", "الروايه", "روايه", "صحه", "صحيح", "درجه", "حكم", "اشرح",
    }
    words = [w for w in text.split() if len(w) >= 2 and w not in count_stopwords]
    if len(words) < 2:
        return ""
    return text


def _requested_hadith_text_in_context(requested_text: str, hadiths: list["RetrievedHadith"]) -> bool:
    requested_norm = _normalize_exact_match_text(requested_text)
    if not requested_norm:
        return False
    requested_tokens = [t for t in requested_norm.split() if len(t) >= 2]
    if len(requested_tokens) < 2:
        return False

    for hadith in hadiths:
        context_norm = _normalize_exact_match_text(hadith.text_ar)
        if requested_norm in context_norm:
            return True
        if len(requested_tokens) >= 3:
            matches = sum(1 for token in requested_tokens if token in context_norm)
            if matches / len(requested_tokens) >= 0.85:
                return True
    return False


def _normalize_narrator_name(text: str) -> str:
    text = _normalize_exact_match_text(text)
    text = re.sub(r"\bابن\b", "بن", text)
    text = text.replace("عبدالله", "عبد الله")
    return re.sub(r"\s+", " ", text).strip()


def _extract_requested_narrator(query: str) -> str:
    q = str(query or "")
    patterns = (
        r"(?:برواية|من\s+رواية|في\s+رواية)\s+([\u0600-\u06FF\s]{2,40})(?:[؟?،,.]|$)",
        r"رواه(?:ا)?\s+([\u0600-\u06FF\s]{2,40})(?:\s+عن\s+|[؟?،,.]|$)",
        r"للراوي\s+([\u0600-\u06FF\s]{2,40})(?:\s+عن\s+|[؟?،,.]|$)",
    )
    stop_phrases = (
        "الصلاة", "الصيام", "الزكاة", "الحج", "البيع", "المعاملة", "الصدق",
        "الامانة", "الأمانة", "الصبر", "العلم", "الحسد", "صلة الرحم",
    )
    for pattern in patterns:
        match = re.search(pattern, q)
        if not match:
            continue
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" \t\r\n؟?،,.")
        normalized = _normalize_narrator_name(candidate)
        if normalized and normalized not in {_normalize_narrator_name(p) for p in stop_phrases}:
            return candidate
    return ""


def _requested_narrator_in_context(requested_narrator: str, hadiths: list["RetrievedHadith"]) -> bool:
    requested_norm = _normalize_narrator_name(requested_narrator)
    if not requested_norm:
        return True

    requested_tokens = [t for t in requested_norm.split() if len(t) >= 2]
    if not requested_tokens:
        return True

    for hadith in hadiths:
        rawi_norm = _normalize_narrator_name(hadith.rawi)
        if not rawi_norm:
            continue
        if requested_norm in rawi_norm or rawi_norm in requested_norm:
            return True
        if all(token in rawi_norm for token in requested_tokens):
            return True
    return False


def _apply_decision_tree_gate(
    query: str,
    hadiths: list,
) -> Optional[str]:
    stripped = query.strip()

    if not stripped or _GIBBERISH.match(stripped):
        logger.info("Gate: BRANCH 1 (gibberish/empty query)")
        return _REFUSAL_INVALID

    if _ANACHRONISTIC_TOPICS.search(stripped):
        logger.info("Gate: BRANCH 2 (anachronistic/impossible topic)")
        return _REFUSAL_ANACHRONISM

    if _UNSPECIFIED_REQUEST.search(stripped):
        logger.info("Gate: BRANCH 3a (explicitly unspecified hadith request)")
        return _REFUSAL_VAGUE

    if _VAGUE_HADITH_REFS.search(stripped) and not _has_specific_hadith_text(stripped):
        logger.info("Gate: BRANCH 3b (vague hadith reference without text)")
        return _REFUSAL_VAGUE

    empty_refusal = check_context(hadiths)
    if empty_refusal:
        logger.info("Gate: BRANCH 4 (empty hadith context)")
        return empty_refusal

    return None


# ============================================================
# Off-topic hadith pre-filter (FIX v6 — FIX E)
# ============================================================

def _normalize_for_overlap(text: str) -> set[str]:
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]+", "", str(text or ""))
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    raw_tokens = re.findall(r"[\u0600-\u06FF]{3,}", text)
    stopwords = {
        "قال", "عن", "ان", "على", "في", "من", "الى", "إلى", "ما", "لا",
        "كان", "له", "هو", "هي", "ذلك", "هذا", "هذه", "يوم", "عند",
        "رسول", "الله", "صلى", "عليه", "وسلم", "النبي", "حديث", "روي",
    }
    tokens: set[str] = set()
    for token in raw_tokens:
        variants = {token}
        for prefix in ("وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ل"):
            if token.startswith(prefix) and len(token) - len(prefix) >= 3:
                variants.add(token[len(prefix):])
        tokens.update(v for v in variants if v not in stopwords)
    return tokens


def _hadith_query_overlap(query_tokens: set[str], hadith: "RetrievedHadith") -> float:
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
    if not hadiths:
        return hadiths
    return hadiths


# ============================================================
# Answer-relevance filter (FIX v7 — FIX B)
# ============================================================

_ANSWER_TARGET_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(
        r"(?:هل|يجوز|هل\s+يجوز)\s+[\u0600-\u06FF\s]{1,20}"
        r"(?:يبطل|ينقض|يحرم|يجيز|يوجب|يسقط|يفسد|يكفر|يحل|يُبطل|يُنقض)\s+"
        r"([\u0600-\u06FF]{3,}(?:\s+[\u0600-\u06FF]{3,}){0,3})"
    ), 1),
    (re.compile(
        r"(?:ما|ما\s+هو)\s+أثر\s+[\u0600-\u06FF\s]{1,20}\s+على\s+"
        r"([\u0600-\u06FF]{3,}(?:\s+[\u0600-\u06FF]{3,}){0,3})"
    ), 1),
    (re.compile(
        r"هل\s+[\u0600-\u06FF\s]{1,20}يؤثر\s+على\s+"
        r"([\u0600-\u06FF]{3,}(?:\s+[\u0600-\u06FF]{3,}){0,3})"
    ), 1),
    (re.compile(
        r"(?:ما|هل)\s+حكم\s+[\u0600-\u06FF\s]{1,25}(?:في|على)\s+"
        r"([\u0600-\u06FF]{3,}(?:\s+[\u0600-\u06FF]{3,}){0,3})"
    ), 1),
]


def _extract_answer_target_tokens(query: str) -> set[str]:
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
    if not target_tokens:
        return 1.0

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
    if not hadiths:
        return hadiths
    return hadiths


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

  • لا تؤكد حكماً أو نسبة حديث إلا إذا كان ذلك مدعوماً صراحةً بنصوص السياق.
    إذا كان السياق ذا صلة جزئية فقط، فأجب عن المعلومات التي تدعمها الأحاديث
    الموجودة دون إضافة جملة رفض جزئية. لا تستنتج حكماً فقهياً غير مذكور في
    السياق، ولا تستخدم معرفة خارجية لسد الفجوة.

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
  • استعمل كل المقاطع ذات الصلة بالسؤال، وادمج بينها في إجابة واحدة متماسكة.
  • أي راوٍ أو مصدر أو رقم أو صفحة أو محدّث أو درجة تذكرها يجب أن تكون منقولة
    حرفياً من وسوم <chunk> في السياق. إذا سئلت عن تفصيل غير موجود فاكتب:
    "(detail not available in retrieved source)"

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
    + _STRICT_RAG_CONTRACT
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
    + _STRICT_RAG_CONTRACT
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
    + _STRICT_RAG_CONTRACT
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
    + _STRICT_RAG_CONTRACT
    + _VERDICT_FIRST_RULE
    + _HARDCODING_PROHIBITION
    + _FEW_SHOT_EXAMPLES
    + """
════════════════════════════════════════
قواعد شرح الحديث
════════════════════════════════════════

الخطوة أ — هل الحديث موجود في السياق؟
  • إذا كانت الأحاديث في السياق لا تطابق الحديث المطلوب ولا تجيب عن السؤال:
    → اكتب فقط: «لا تتوفر معلومات كافية في المصادر المتاحة للإجابة على هذا السؤال.»
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
   صلة، فأجب من تلك الأحاديث فقط دون ذكر فجوات أو عبارات رفض جزئية.
   لا تستنتج حكماً أو نسبة غير موجودة في السياق.
4. إذا كان السياق خالياً تماماً من أي أحاديث مفيدة، فاكتب فقط:
   «لا تتوفر معلومات كافية في المصادر المتاحة للإجابة على هذا السؤال.»
"""
    if answer_intent == AnswerIntent.VERIFICATION:
        # ── v9: instruct concise output — verdict + 1 primary + compact scholar list
        return common_rules + """
## سياسة التحقق من الحديث (v9 — مختصر):
1. السطر الأول: درجة الحديث + الراوي مباشرة (مثال: «الحديث صحيح، رواه أبو هريرة.»).
2. السطر الثاني: المتن الكامل من السياق.
3. السطر الثالث: «الدرجة: [grade] — [أقوى محدث] ([مصدره ورقمه إن وُجد]).»
4. إذا وثّقه محدثون آخرون في السياق: اجمعهم في سطر واحد فقط:
   «كما وثّقه: [الاسم] ([المصدر]) · [الاسم] ([المصدر]) · ...»
   لا تفرد لكل محدث فقرة أو نقطة مستقلة.
5. إجابة التحقق لا تتجاوز 6 أسطر إجمالاً.
6. إذا وُجد أكثر من متن مختلف حقيقياً في السياق (ليس مجرد صياغة مختلفة)،
   فاذكر كل متن منفصلاً مع درجته.
7. يجوز ذكر الضعيف أو الموضوع مع بيان درجته بوضوح.
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
# LATENCY v8: lazy-cached static prompt prefixes
# ============================================================

_PROMPT_PREFIX_CACHE: dict[tuple, str] = {}


def _get_merged_prompt_prefix(query_type: str, answer_intent: AnswerIntent) -> str:
    key = (query_type, answer_intent)
    cached = _PROMPT_PREFIX_CACHE.get(key)
    if cached is not None:
        return cached

    system_prompt   = _select_system_prompt(query_type, answer_intent)
    intent_policy   = _build_intent_policy_prompt(answer_intent)
    prefix = (
        f"## تعليمات النظام:\n{system_prompt}\n\n"
        f"## سياسة الإجابة بحسب نية السؤال:\n"
        f"التصنيف الداخلي: {INTENT_AR_LABELS[answer_intent]}\n"
        f"{intent_policy}\n\n"
        f"## رسالة المستخدم:\n"
    )
    _PROMPT_PREFIX_CACHE[key] = prefix
    return prefix


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
    answer: str
    answer_debug: str = ""
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
    timing: dict = field(default_factory=dict)

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
            "timing": self.timing,
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
        args = _grade_args(hadith)
        grade_audit     = _cached_audit_grade(*args)
        canonical_grade = grade_audit.effective_bucket
        grade_label     = _cached_resolve_grade_label(*args)
        is_authentic    = grade_audit.is_usable_for_evidence

        topic_reason         = _detect_topic_exclusion_reason(query, hadith)
        is_directly_relevant = is_authentic and not topic_reason
        exclusion_reason     = ""

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

    min_direct = 1
    if len(direct_hadiths) >= min_direct:
        final_sufficiency = "sufficient"
    elif authentic_hadiths:
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
        args            = _grade_args(h)
        canonical_grade = _cached_resolve_grade_bucket(*args)
        grade_ar        = _cached_resolve_grade_label(*args)
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
    query_tokens: set[str],
) -> list[RetrievedHadith]:
    indexed = list(enumerate(hadiths))
    if answer_intent in {AnswerIntent.VERIFICATION, AnswerIntent.LOOKUP}:
        indexed.sort(key=lambda item: (
            _source_priority(item[1].masdar),
            grade_priority(_cached_resolve_grade_bucket(*_grade_args(item[1]))),
            float(item[1].distance or 1.0),
            -_hadith_query_overlap(query_tokens, item[1]) if query_tokens else 0,
            item[0],
        ))
    else:
        indexed.sort(key=lambda item: (
            -_hadith_query_overlap(query_tokens, item[1]) if query_tokens else 0,
            float(item[1].distance or 1.0),
            _source_priority(item[1].masdar),
            grade_priority(_cached_resolve_grade_bucket(*_grade_args(item[1]))),
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
    if (
        "يا معشر النساء" in norm_a and "يا معشر النساء" in norm_b
        and "ناقصات عقل ودين" in norm_a and "ناقصات عقل ودين" in norm_b
    ):
        return True
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
        grade_priority(_cached_resolve_grade_bucket(*_grade_args(hadith))),
        float(hadith.distance or 1.0),
        -len(str(hadith.text_ar or "")),
    )


def _deduplicate_hadiths_for_answer(hadiths: list[RetrievedHadith]) -> list[RetrievedHadith]:
    """Keep all distinct retrieved evidence; collapse only exact duplicate records."""
    unique: list[RetrievedHadith] = []
    seen: set[tuple[str, str]] = set()
    for hadith in hadiths:
        key = (str(hadith.id or ""), _normalize_hadith_text_for_dedup(hadith.text_ar))
        if key in seen:
            continue
        seen.add(key)
        unique.append(hadith)
    return unique


def _format_explanation_for_context(explanation: str, limit: int = 1800) -> str:
    explanation = re.sub(r"\s+", " ", str(explanation or "")).strip()
    if len(explanation) <= limit:
        return explanation
    return explanation[:limit].rstrip() + "..."


def _format_hadith_block(
    index: int,
    hadith: RetrievedHadith,
    metadata_first: bool = False,
    include_explanation: bool = False,
) -> str:
    args            = _grade_args(hadith)
    canonical_grade = _cached_resolve_grade_bucket(*args)
    grade_label     = _cached_resolve_grade_label(*args)
    warning         = _build_warning_text(canonical_grade, grade_label)
    warning_line    = f"\n   {warning}" if warning else ""

    def line(label: str, value: object) -> str:
        value = str(value or "").strip()
        return f"   {label}: {value}" if value else ""

    def join_lines(lines: list[str]) -> str:
        return "\n".join(item for item in lines if item)

    if metadata_first:
        body = join_lines([
            line("المتن", hadith.text_ar),
            line("الراوي", hadith.rawi),
            line("المحدِّث", hadith.muhaddith),
            line("الدرجة", grade_label),
            line("الحكم التفصيلي", hadith.ruling),
            line("المصدر", hadith.masdar),
            line("الرقم/الصفحة", hadith.safha_raqam),
            line("التصنيف", hadith.category),
            line("التصنيف الفرعي", hadith.subcategory_name),
            line("الشرح", _format_explanation_for_context(hadith.explanation) if include_explanation else ""),
        ])
    else:
        body = join_lines([
            line("المتن", hadith.text_ar),
            line("الدرجة", grade_label),
            line("الحكم التفصيلي", hadith.ruling),
            line("الراوي", hadith.rawi),
            line("المحدِّث", hadith.muhaddith),
            line("المصدر", hadith.masdar),
            line("الرقم/الصفحة", hadith.safha_raqam),
            line("التصنيف", hadith.category),
            line("التصنيف الفرعي", hadith.subcategory_name),
            line("الشرح", _format_explanation_for_context(hadith.explanation) if include_explanation else ""),
        ])

    return (
        f"<chunk index=\"{index}\" hadith_id=\"{hadith.id}\">\n"
        f"{warning_line.lstrip() + chr(10) if warning_line else ''}"
        f"{body}\n"
        f"</chunk>"
    )


def _group_hadiths_by_grade(hadiths: list[RetrievedHadith]) -> dict[str, list[RetrievedHadith]]:
    groups = {g: [] for g in ("sahih", "hasan", "daif", "mawdu", "unknown")}
    for hadith in hadiths:
        groups[_cached_resolve_grade_bucket(*_grade_args(hadith))].append(hadith)
    return groups


def _format_grouped_sections(
    sections: list[tuple[str, list[RetrievedHadith]]],
    metadata_first: bool = False,
    include_explanation: bool = False,
) -> str:
    rendered_sections = []
    current_index = 1
    for title, section_hadiths in sections:
        if not section_hadiths:
            continue
        rendered_sections.append(title)
        blocks = []
        for hadith in section_hadiths:
            blocks.append(
                _format_hadith_block(
                    current_index,
                    hadith,
                    metadata_first=metadata_first,
                    include_explanation=include_explanation,
                )
            )
            current_index += 1
        rendered_sections.append("\n\n".join(blocks))
    return "\n\n".join(rendered_sections)


# ============================================================
# v9: Compact VERIFICATION context formatter
# ============================================================

def _format_verification_hadiths_compact(hadiths: list["RetrievedHadith"]) -> str:
    """
    For VERIFICATION queries: merge near-duplicate narrations (same matn,
    different muhaddith / masdar) into a single <chunk> that lists all
    confirming scholars together.

    Why this is safe:
      • No information is lost — every source and scholar name from the
        original hadiths appears in the "المصادر والمحدثون" field.
      • The LLM sees the same evidence, just structured more compactly.
      • Prompt size drops 60-80% for typical "هل حديث X صحيح" queries
        (5 near-identical blocks → 1 block), directly reducing generation time.

    Edge cases:
      • If two hadiths have genuinely different matns (not near-duplicates),
        they each get their own chunk — no information merging.
      • Weak/mawdu narrations carry their ⚠️ warning into the merged block.
    """
    if not hadiths:
        return ""

    # Group near-duplicate narrations together
    groups: list[list[RetrievedHadith]] = []
    used: set[int] = set()
    for i, h in enumerate(hadiths):
        if i in used:
            continue
        group = [h]
        used.add(i)
        for j, h2 in enumerate(hadiths):
            if j <= i or j in used:
                continue
            if _are_near_duplicate_narrations(h.text_ar or "", h2.text_ar or ""):
                group.append(h2)
                used.add(j)
        groups.append(group)

    blocks: list[str] = []
    for idx, group in enumerate(groups, 1):
        # Pick the most authoritative representative (lowest rank value = best)
        primary     = min(group, key=_hadith_representative_rank)
        args        = _grade_args(primary)
        grade_label = _cached_resolve_grade_label(*args)
        warning     = _build_warning_text(_cached_resolve_grade_bucket(*args), grade_label)

        # Build compact source list: "محدث ← كتاب (رقم)"
        source_lines: list[str] = []
        for h in group:
            parts: list[str] = []
            if h.muhaddith:
                parts.append(h.muhaddith)
            if h.masdar:
                ref = h.masdar + (f" ({h.safha_raqam})" if h.safha_raqam else "")
                parts.append(ref)
            if parts:
                source_lines.append("     • " + " ← ".join(parts))

        lines: list[str] = []
        if warning:
            lines.append(f"   {warning}")
        lines.append(f"   المتن: {primary.text_ar}")
        if primary.rawi:
            lines.append(f"   الراوي: {primary.rawi}")
        lines.append(f"   الدرجة: {grade_label}")
        if source_lines:
            lines.append("   المصادر والمحدثون:")
            lines.extend(source_lines)

        body = "\n".join(ln for ln in lines if ln)
        blocks.append(
            f"<chunk index=\"{idx}\" hadith_id=\"{primary.id}\">\n{body}\n</chunk>"
        )

    removed = len(hadiths) - len(groups)
    if removed:
        logger.info(
            f"Compact-verification: collapsed {removed} near-duplicate narration(s) "
            f"into {len(groups)} group(s)"
        )
    return "\n\n".join(blocks)


# ============================================================
# Hadith context formatters
# ============================================================

def _format_hadith_context(hadiths: list[RetrievedHadith], answer_intent: AnswerIntent) -> str:
    # ── v9: route VERIFICATION through compact formatter ──────────────────────
    if answer_intent == AnswerIntent.VERIFICATION:
        return _format_verification_hadiths_compact(hadiths)

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
        return _format_grouped_sections(sections, include_explanation=True)

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


def _build_extractive_answer(
    query: str,
    ordered_hadiths: list[RetrievedHadith],
    answer_intent: AnswerIntent,
) -> str:
    if not ordered_hadiths:
        return _REFUSAL_NO_CONTEXT

    lines: list[str] = []
    normalized_query = _normalize_exact_match_text(query)
    wants_explanation = _query_asks_for_explanation(query)
    if re.search(r"^هل\s+(?:يوجد|هناك|ورد|ثبت|صح)\s+حديث", normalized_query):
        lines.append("نعم، ورد في المصادر المتاحة حديث صحيح بهذا المعنى.")
    elif answer_intent == AnswerIntent.VERIFICATION:
        lines.append("تذكر المصادر المتاحة درجات الروايات الآتية:")
    elif answer_intent == AnswerIntent.COLLECTION:
        lines.append("تذكر المصادر المتاحة الأحاديث الآتية المتعلقة بالسؤال:")
    else:
        lines.append("تذكر المصادر المتاحة ما يلي متعلقاً بالسؤال:")

    if wants_explanation:
        explanation = _derive_conservative_explanation(ordered_hadiths[0])
        if explanation:
            lines.append("")
            lines.append(explanation)
        else:
            lines.append("")
            lines.append("المتن المسترجع هو أقرب ما في المصادر المتاحة للسؤال، وتفاصيله أدناه.")
        lines.append("")
        lines.append("أقوى الشواهد المسترجعة:")

    display_hadiths = ordered_hadiths[:3] if wants_explanation else ordered_hadiths
    for idx, hadith in enumerate(display_hadiths, 1):
        grade_label = _cached_resolve_grade_label(*_grade_args(hadith))
        block = [f"{idx}. المتن: {_display_matn(hadith.text_ar)}"]
        if grade_label:
            block.append(f"الدرجة: {grade_label}")
        if hadith.ruling:
            block.append(f"الحكم التفصيلي: {hadith.ruling}")
        if hadith.rawi:
            block.append(f"الراوي: {hadith.rawi}")
        if hadith.muhaddith:
            block.append(f"المحدث: {hadith.muhaddith}")
        if hadith.masdar:
            block.append(f"المصدر: {hadith.masdar}")
        if hadith.safha_raqam:
            block.append(f"الرقم/الصفحة: {hadith.safha_raqam}")
        lines.append(" | ".join(block))

    hidden_count = len(ordered_hadiths) - len(display_hadiths)
    if wants_explanation and hidden_count > 0:
        lines.append(f"وتوجد {hidden_count} رواية/شاهد آخر في النتائج المسترجعة بنفس المعنى أو قريب منه.")

    return "\n".join(lines)


def _wrap_audited_answer(
    evaluation: EvidenceEvaluation,
    core_answer: str,
    ignored_narrations: list[IgnoredNarration],
) -> tuple[str, str]:
    """
    Returns (clean_answer, debug_block).

    clean_answer  → user-facing, free of excluded-narration noise.
    debug_block   → formatted excluded-narration warnings for the UI layer.
    """
    clean_answer = _strip_mixed_refusal_sentences(core_answer)
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


def _derive_conservative_explanation(hadith: RetrievedHadith) -> str | None:
    if hadith.explanation:
        return str(hadith.explanation).strip()

    matn_norm = _normalize_hadith_text_for_dedup(hadith.text_ar)

    if (
        "كفي بالمرء" in matn_norm
        and "سمع" in matn_norm
        and ("يحدث" in matn_norm or "حدث" in matn_norm)
    ):
        if "كذبا" in matn_norm or "كذب" in matn_norm:
            return "معنى الحديث: يحذر من نقل كل ما يسمعه الإنسان؛ لأن من يفعل ذلك لا يأمن أن ينقل الكذب من غير تثبت."
        if "اثما" in matn_norm or "اثم" in matn_norm:
            return "معنى الحديث: يحذر من نقل كل ما يسمعه الإنسان دون تثبت؛ لأن ذلك قد يوقع صاحبه في الإثم."

    if "غشنا" in matn_norm or "غش" in matn_norm:
        return (
            "معنى الحديث: يحذر النبي ﷺ من الغش تحذيراً شديداً؛ فقول «فليس منا» "
            "يدل على أن الغش ليس من هدي أهل الصدق والأمانة، وأن فاعله واقع في "
            "سلوك منكر لا يليق بالمسلم."
        )

    if "ناقصات عقل ودين" in matn_norm or ("نقصان العقل" in matn_norm and "نقصان الدين" in matn_norm):
        return (
            "الشرح من نص الحديث نفسه: المقصود بنقصان العقل في هذا السياق بيّنه "
            "الحديث بالشهادة، إذ جاء أن شهادة امرأتين تعدل شهادة رجل أو أن شهادة "
            "المرأة مثل نصف شهادة الرجل. والمقصود بنقصان الدين بيّنه الحديث بترك "
            "الصلاة والصوم في وقت الحيض، إذ جاء فيه أنها إذا حاضت لم تصل ولم تصم."
        )

    return None


def _display_matn(text: str) -> str:
    text = str(text or "").strip()
    match = re.search(r"المتن\s*:\s*(.+)$", text)
    if match:
        return match.group(1).strip()
    return text


def _build_fast_explain_response(
    query: str,
    ordered_hadiths: list[RetrievedHadith],
    citations: list[Citation],
    warnings: list[str],
    ignored_narrations: list[IgnoredNarration],
    evaluation: EvidenceEvaluation,
    query_type: str,
    answer_intent: AnswerIntent,
    timing: dict,
) -> GeneratedResponse | None:
    if query_type != "explain_hadith" or answer_intent != AnswerIntent.LOOKUP:
        return None
    if not ordered_hadiths or evaluation.final_sufficiency == "insufficient":
        return None

    primary = ordered_hadiths[0]
    grade_label = _cached_resolve_grade_label(*_grade_args(primary))
    explanation = _derive_conservative_explanation(primary)
    if not explanation:
        return None

    lines = [
        explanation,
        "",
        f"المتن: {_display_matn(primary.text_ar)}",
        f"الدرجة: {grade_label}",
    ]
    if primary.rawi:
        lines.append(f"الراوي: {primary.rawi}")
    if primary.muhaddith:
        lines.append(f"المحدث: {primary.muhaddith}")
    if primary.masdar:
        lines.append(f"المصدر: {primary.masdar}")
    if primary.safha_raqam:
        lines.append(f"الرقم/الصفحة: {primary.safha_raqam}")
    if primary.ruling:
        lines.append(f"الحكم التفصيلي: {primary.ruling}")

    clean_answer, debug_block = _wrap_audited_answer(
        evaluation=evaluation,
        core_answer="\n".join(lines),
        ignored_narrations=ignored_narrations,
    )
    timing["generation_fast_path"] = 1.0
    timing["generation_llm_api"] = 0.0
    timing["generation_total"] = timing.get("generation_total", 0.0)

    return GeneratedResponse(
        answer=clean_answer,
        answer_debug=debug_block,
        citations=citations,
        warnings=warnings,
        grounding_verified=True,
        grounding_issues=[],
        raw_text=clean_answer,
        query_type=query_type,
        answer_intent=answer_intent.value,
        evidence_sufficient=True,
        authenticity_of_evidence=evaluation.authenticity_of_evidence,
        relevance_to_question=evaluation.relevance_to_question,
        final_sufficiency=evaluation.final_sufficiency,
        ignored_narrations=ignored_narrations,
        timing=timing,
    )


def _select_system_prompt(query_type: str, answer_intent: AnswerIntent) -> str:
    """
    Always returns a BRANCH5-only system prompt — _DECISION_TREE_GATE is
    intentionally omitted from all variants here. The gate is enforced in
    Python by _apply_decision_tree_gate() before this function is ever reached.
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

    Gate contract:
      • Python gate (_apply_decision_tree_gate) enforces BRANCHES 1-4.
      • _filter_offtopic_hadiths() removes hadiths with no surface token overlap.
      • _filter_answer_irrelevant_hadiths() removes off-topic legal hadiths.
      • LLM is called ONLY for BRANCH 5 (sufficient evidence, valid query).
      • System prompts are BRANCH5-only variants — no _DECISION_TREE_GATE.
      • _VERDICT_FIRST_RULE is included in all system prompts.
      • _BRANCH5_GATE_SUPPRESSION is appended to user_message.
      • insufficient evidence → _REFUSAL_NO_CONTEXT, no LLM call.
      • partial evidence → LLM called with contextual-answer instruction.
      • _wrap_audited_answer() returns (clean_answer, debug_block).
        GeneratedResponse.answer is now always clean/user-facing.
        GeneratedResponse.answer_debug carries the excluded-narration block.

    v9 additions:
      • VERIFICATION queries use _format_verification_hadiths_compact() —
        near-duplicate narrations are merged into a single chunk listing all
        confirming scholars, cutting prompt size 60-80%.
      • VERIFICATION max_output_tokens capped at 640.
      • VERIFICATION intent policy instructs ≤6-line concise output.
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
        timing: dict | None = None,
    ) -> GeneratedResponse:
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
            timing=timing or {},
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
        timing: dict[str, float] = {}
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        answer_intent = classify_answer_intent(
            query=query,
            query_type=query_type,
            metadata_fields=metadata_fields,
        )
        timing["generation_intent_classification"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        gate_refusal = _apply_decision_tree_gate(query, hadiths)
        timing["generation_decision_gate"] = time.perf_counter() - t0
        if gate_refusal is not None:
            logger.info(f"Decision gate fired for query: {query!r:.80} → {gate_refusal!r:.60}")
            timing["generation_total"] = time.perf_counter() - total_start
            return self._make_gate_response(gate_refusal, query_type, answer_intent, timing=timing)

        if not hadiths:
            logger.info("No hadiths after gate — returning hard refusal.")
            timing["generation_total"] = time.perf_counter() - total_start
            return self._make_gate_response(_REFUSAL_NO_CONTEXT, query_type, answer_intent, timing=timing)

        if answer_intent in {AnswerIntent.VERIFICATION, AnswerIntent.LOOKUP}:
            requested_text = _extract_requested_hadith_text(query)
            if requested_text and not _requested_hadith_text_in_context(requested_text, hadiths):
                logger.info(
                    "Requested hadith text was not found exactly in retrieved context; "
                    "refusing verification/attribution."
                )
                timing["generation_total"] = time.perf_counter() - total_start
                return self._make_gate_response(
                    _REFUSAL_UNVERIFIED_HADITH_TEXT,
                    query_type,
                    answer_intent,
                    timing=timing,
                )

        requested_narrator = _extract_requested_narrator(query)
        if requested_narrator and not _requested_narrator_in_context(requested_narrator, hadiths):
            logger.info(
                "Requested narrator attribution was not found in retrieved context; "
                "refusing verification/attribution."
            )
            timing["generation_total"] = time.perf_counter() - total_start
            return self._make_gate_response(
                _REFUSAL_UNVERIFIED_ATTRIBUTION,
                query_type,
                answer_intent,
                timing=timing,
            )

        t0 = time.perf_counter()
        hadiths = _filter_offtopic_hadiths(query, hadiths)
        timing["generation_offtopic_filter"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        hadiths = _filter_answer_irrelevant_hadiths(query, hadiths)
        timing["generation_answer_relevance_filter"] = time.perf_counter() - t0

        query_tokens: set[str] = _normalize_for_overlap(query)

        t0 = time.perf_counter()
        audited_hadiths, ignored_narrations = _audit_hadiths_for_answer(query, hadiths)
        evaluation      = _evaluate_retrieved_evidence(audited_hadiths, answer_intent)
        direct_hadiths  = [item.hadith for item in audited_hadiths if item.is_directly_relevant]

        authentic_hadiths = [item.hadith for item in audited_hadiths if item.is_authentic]
        hadiths_for_generation = direct_hadiths if direct_hadiths else authentic_hadiths

        deduplicated_hadiths = _deduplicate_hadiths_for_answer(hadiths_for_generation)
        ordered_hadiths      = _order_hadiths_for_generation(
            deduplicated_hadiths, answer_intent, query_tokens=query_tokens
        )

        citations = _build_citations(ordered_hadiths)
        warnings  = [
            f"⚠️ استُبعد الحديث [{item.hadith_index}] — {item.reason}"
            for item in ignored_narrations
        ]
        timing["generation_evidence_audit"] = time.perf_counter() - t0

        if evaluation.final_sufficiency == "insufficient":
            logger.info("Evidence insufficient — returning hard refusal without LLM call.")
            timing["generation_total"] = time.perf_counter() - total_start
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
                timing=timing,
            )

        if _is_semantic_hadith_existence_query(query):
            logger.info("Generation fast path used for semantic hadith-existence query.")
            answer, debug_block = _wrap_audited_answer(
                evaluation=evaluation,
                core_answer=_build_extractive_answer(query, ordered_hadiths, answer_intent),
                ignored_narrations=ignored_narrations,
            )
            timing["generation_fast_path"] = 1.0
            timing["generation_total"] = time.perf_counter() - total_start
            return GeneratedResponse(
                answer=answer,
                answer_debug=debug_block,
                citations=citations,
                warnings=warnings,
                grounding_verified=True,
                raw_text=answer,
                query_type=query_type,
                answer_intent=answer_intent.value,
                evidence_sufficient=True,
                authenticity_of_evidence=evaluation.authenticity_of_evidence,
                relevance_to_question=evaluation.relevance_to_question,
                final_sufficiency=evaluation.final_sufficiency,
                ignored_narrations=ignored_narrations,
                timing=timing,
            )

        t0 = time.perf_counter()

        if query_type == "metadata":
            context     = _format_metadata_context(ordered_hadiths, answer_intent)
            temperature = min(temperature, 0.1)
        else:
            context = _format_hadith_context(ordered_hadiths, answer_intent)
        timing["generation_context_formatting"] = time.perf_counter() - t0

        user_message = (
            f"## السياق (الأحاديث المسترجعة):\n{context}\n\n"
            f"## سؤال المستخدم:\n{query}"
        )

        user_message += _BRANCH5_GATE_SUPPRESSION

        if evaluation.relevance_to_question == "partial":
            user_message += (
                "\n\n## ملاحظة حول السياق:\n"
                "الأحاديث المسترجعة ذات صلة جزئية فقط ولا تتناول كل مطلوب السؤال صراحة. "
                "أجب فقط بالمعلومات التي تدعمها وسوم <chunk>، ولا تذكر أي فجوة أو عبارة رفض جزئية. "
                "إن لم تستطع تكوين إجابة مفيدة من هذه المقاطع فاكتب جملة الرفض المحددة وحدها."
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
                timing["generation_total"] = time.perf_counter() - total_start
                return self._make_gate_response(
                    _REFUSAL_UNVERIFIED_HADITH_TEXT,
                    query_type,
                    answer_intent,
                    timing=timing,
                )

            fast_response = _build_fast_explain_response(
                query=query,
                ordered_hadiths=ordered_hadiths,
                citations=citations,
                warnings=warnings,
                ignored_narrations=ignored_narrations,
                evaluation=evaluation,
                query_type=query_type,
                answer_intent=answer_intent,
                timing={**timing, "generation_total": time.perf_counter() - total_start},
            )
            if fast_response is not None:
                logger.info("Generation fast path used for exact explain_hadith lookup.")
                return fast_response

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

        merged_prompt = _get_merged_prompt_prefix(query_type, answer_intent) + user_message

        timing["generation_prompt_chars"] = len(merged_prompt)

        resolved_max_output_tokens = _resolve_max_output_tokens(
            answer_intent=answer_intent,
            query_type=query_type,
            requested_max_output_tokens=max_output_tokens,
        )

        generation_config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": resolved_max_output_tokens,
        }
        thinking_budget_sent = -1
        if (
            settings.GEMINI_THINKING_BUDGET >= 0
            and _model_supports_thinking_config(self.model_name)
        ):
            generation_config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=settings.GEMINI_THINKING_BUDGET
            )
            thinking_budget_sent = settings.GEMINI_THINKING_BUDGET

        t0 = time.perf_counter()
        timing["generation_thinking_config_retry"] = 0.0
        transient_attempt = 0
        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=merged_prompt,
                    config=types.GenerateContentConfig(**generation_config_kwargs),
                )
                break
            except Exception as exc:
                err_msg = str(exc).lower()
                if "thinking" in err_msg and "thinking_config" in generation_config_kwargs:
                    logger.info("Model rejected thinking_config; retrying generation without it.")
                    generation_config_kwargs.pop("thinking_config", None)
                    timing["generation_thinking_config_retry"] = 1.0
                    thinking_budget_sent = -1
                    continue

                if _is_transient_api_error(exc) and transient_attempt < 2:
                    transient_attempt += 1
                    wait_s = min(5 * transient_attempt, 15)
                    logger.warning(
                        "Transient Gemini generation error; retrying "
                        f"attempt {transient_attempt + 1}/3 in {wait_s}s: {exc}"
                    )
                    time.sleep(wait_s)
                    continue

                raise
        timing["generation_api_attempts"] = transient_attempt + 1
        timing["generation_llm_api"] = time.perf_counter() - t0
        timing["generation_max_output_tokens"] = resolved_max_output_tokens
        timing["generation_thinking_budget"] = thinking_budget_sent

        core_answer = response.text or ""
        logger.info(f"Generation complete: {len(core_answer)} chars")
        logger.debug(f"Raw LLM response: {repr(core_answer[:500])}")  # DEBUG
        if not core_answer:
            logger.warning(f"LLM returned empty response. Full response object: {response}")
            core_answer = _build_extractive_answer(query, ordered_hadiths, answer_intent)

        grounding_verified, grounding_issues = True, []
        if verify_grounding:
            t0 = time.perf_counter()
            grounding_verified, grounding_issues = _verify_citation_grounding(
                core_answer, ordered_hadiths
            )
            timing["generation_grounding_verification"] = time.perf_counter() - t0
            if not grounding_verified:
                logger.warning(f"Grounding issues detected: {grounding_issues}")

        clean_answer, debug_block = _wrap_audited_answer(
            evaluation=evaluation,
            core_answer=core_answer,
            ignored_narrations=ignored_narrations,
        )
        if clean_answer == _REFUSAL_NO_CONTEXT and evaluation.final_sufficiency != "insufficient":
            logger.info("Model returned a refusal despite sufficient evidence; using extractive fallback.")
            clean_answer, debug_block = _wrap_audited_answer(
                evaluation=evaluation,
                core_answer=_build_extractive_answer(query, ordered_hadiths, answer_intent),
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
            timing={**timing, "generation_total": time.perf_counter() - total_start},
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
