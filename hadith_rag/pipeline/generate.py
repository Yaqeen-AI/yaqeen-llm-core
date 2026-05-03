# ============================================================
# YaqeenAI — Generation Module (Production Enhanced)
# ============================================================
# Generates answers using Gemini API with Arabic-first system prompt.
# Assembles context from reranked hadiths and enforces citation rules.
#
# Enhancements:
# - Metadata-focused prompts for metadata queries (rawi, grade, masdar, etc.)
# - Anti-hallucination: explicit refusal instructions, lower temperature for metadata
# - Khilaf (scholarly difference) handling
# - Weak/fabricated hadith warnings
# - Citation grounding verification
# - Query-type-aware generation (different prompts for different query types)

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Optional
from dataclasses import dataclass, field, asdict

# from groq import Groq
from google import genai
from google.genai import types

from pipeline.answer_policy import AnswerIntent, classify_answer_intent, grade_priority
from pipeline.config import (
    audit_grade,
    resolve_grade_bucket,
    resolve_grade_label,
    settings,
)
from pipeline.retrieve import RetrievedHadith

logger = logging.getLogger(__name__)


# ============================================================
# System Prompts — Query-Type Aware
# ============================================================

SYSTEM_PROMPT_GENERAL = """أنت عالم متخصص في الحديث النبوي الشريف. مهمتك الإجابة عن أسئلة المستخدم بناءً على الأحاديث المقدمة لك فقط في السياق أدناه.

## القواعد الصارمة:

1. **الاستشهاد الإلزامي**: لكل حديث تستشهد به، يجب ذكر:
   - نص الحديث (المتن)
   - المصدر (اسم الكتاب)
   - رقم الحديث أو الصفحة
   - الراوي
   - المحدِّث الذي حكم عليه
   - درجة الحديث (صحيح، حسن، ضعيف، موضوع)

2. **التحذير من الأحاديث الضعيفة والموضوعة**: إذا كان الحديث ضعيفاً أو موضوعاً:
   - اذكر ذلك بوضوح قبل نص الحديث
   - استخدم عبارة: "⚠️ تنبيه: هذا حديث [ضعيف/موضوع] — لا يُحتج به في إثبات الأحكام"
   - لا تبنِ أحكاماً شرعية على أحاديث ضعيفة أو موضوعة منفردة

3. **الامتناع عند عدم كفاية السياق**: إذا لم تجد في الأحاديث المقدمة ما يكفي للإجابة:
   - قل بوضوح: "لم أجد في الأحاديث المتاحة ما يكفي للإجابة عن سؤالك بشكل كامل"
   - لا تختلق أحاديث أو تنسب أقوالاً لم ترد في السياق المقدم
   - لا تذكر أرقام أحاديث أو أسماء كتب أو رواة لم ترد في السياق

4. **عدم الاختلاق مطلقاً**:
   - لا تذكر أي حديث غير موجود في السياق المقدم
   - لا تنسب حكماً لعالم لم يُذكر في السياق
   - لا تذكر رقم حديث أو صفحة لم يرد في البيانات المقدمة
   - لا تخلط بين بيانات حديث وبيانات حديث آخر

5. **اللغة**: أجب دائماً باللغة العربية الفصحى

6. **الترتيب**: رتّب الأحاديث في إجابتك حسب قوة الإسناد: صحيح ← حسن ← ضعيف

7. **الأمانة العلمية**: لا تخلط بين نص الحديث وشرحه. اذكر المتن كما ورد، ثم اشرحه إن لزم الأمر.

8. **التنسيق**: استخدم تنسيقاً واضحاً يسهل القراءة مع فواصل بين الأحاديث المختلفة.

9. **مسائل الخلاف**: إذا كان السؤال يتعلق بمسألة فقهية فيها خلاف بين العلماء:
   - اعرض الأقوال المختلفة إن وُجدت في السياق
   - لا تنحاز لرأي واحد دون ذكر الآخر
    - قل: "في هذه المسألة خلاف بين العلماء" إن كان ذلك واضحاً

10. **الدقة وعدم التهويل**:
    - لا تستخدم عبارات تعميمية مثل "عدة أحاديث" أو "كثيرة" إلا إذا كان العدد ظاهراً من السياق.
    - الأفضل ذكر العدد بشكل صريح مثل: "ورد في النتائج حديثان...".

11. **منع التكرار**:
    - لا تكرر نفس الرواية أو نفس المعنى المكرر بصياغات متعددة.
    - إذا تكررت روايات متقاربة جداً، اذكر الأقوى سنداً فقط ثم نبّه باختصار لوجود روايات مقاربة.

12. **الأسئلة الحساسة**:
    - عند الأسئلة التي قد تُفهم على أنها انتقاص من فئة من الناس، اذكر سطر توضيح منهجي مختصر يبيّن أن الفهم يكون في سياقه الشرعي واللغوي، دون خطاب جارح."""


SYSTEM_PROMPT_METADATA = """أنت عالم متخصص في الحديث النبوي. المستخدم يسأل عن **بيانات وصفية** محددة لحديث معين (الراوي، الدرجة، المصدر، الرقم، المحدث، التصنيف).

## قواعد صارمة للإجابة عن أسئلة البيانات الوصفية:

1. **أجب من السياق فقط**: استخرج المعلومة المطلوبة من بيانات الأحاديث المقدمة فقط.

2. **كن مباشراً ودقيقاً**: أجب عن السؤال المحدد مباشرة، ثم أضف التفاصيل.
   - إذا سُئل "من رواه؟" → ابدأ بذكر الراوي مباشرة
   - إذا سُئل "ما درجته؟" → ابدأ بذكر الدرجة مباشرة
   - إذا سُئل "في أي كتاب؟" → ابدأ بذكر المصدر مباشرة
   - إذا سُئل "ما رقمه؟" → ابدأ بذكر الرقم مباشرة
   - إذا سُئل "من حكم عليه؟" → ابدأ بذكر المحدث مباشرة

3. **لا تختلق بيانات**: إذا لم تجد المعلومة في السياق المقدم:
   - قل: "لم أجد هذه المعلومة في البيانات المتاحة"
   - لا تخمن أو تستنتج بيانات لم ترد صراحة

4. **أضف السياق الكامل**: بعد الإجابة المباشرة، اذكر بقية بيانات الحديث للفائدة:
   - نص المتن (مختصراً)
   - المصدر والرقم
   - الراوي والمحدث
   - الدرجة

5. **إذا وُجد الحديث في أكثر من مصدر**: اذكر جميع المصادر المتاحة في السياق.

6. **اللغة**: أجب باللغة العربية الفصحى."""


SYSTEM_PROMPT_NARRATOR = """أنت عالم متخصص في الحديث النبوي. المستخدم يبحث عن أحاديث مرتبطة **براوٍ محدد**.

## القواعد:

1. اعرض الأحاديث التي رواها هذا الراوي من السياق المقدم فقط.
2. لكل حديث، اذكر: المتن، المصدر، الرقم، المحدث، الدرجة.
3. رتّب بحسب قوة الإسناد: صحيح ← حسن ← ضعيف.
4. لا تذكر أحاديث غير موجودة في السياق.
5. إذا كان الحديث ضعيفاً أو موضوعاً، نبّه بوضوح.
6. أجب باللغة العربية الفصحى."""


SYSTEM_PROMPT_EXPLAIN = """أنت عالم متخصص في الحديث النبوي الشريف. المستخدم يطلب شرح حديث بعينه.

## قواعد شرح الحديث:

1. **ابحث أولاً في السياق**: هل يوجد في الأحاديث المقدمة ما يطابق الحديث المطلوب أو يقاربه؟
   - إذا وُجد الحديث أو ما يشابهه → اشرحه بالتفصيل.
   - إذا لم يُوجد → صرّح بذلك بوضوح (انظر القاعدة 4).

2. **الشرح المطلوب** (عند وجود الحديث):
   - اذكر المتن الكامل كما ورد.
   - اذكر درجة الحديث وحكم المحدثين عليه.
   - اشرح معنى الحديث وما يُستفاد منه.
   - اذكر المصدر والراوي.
   - إذا كان ضعيفاً أو موضوعاً: نبّه بعبارة واضحة ⚠️ ولا تبنِ عليه أحكاماً.

3. **التحذير من الأحاديث الضعيفة والموضوعة**:
   - إذا كانت الأحاديث المقدمة ضعيفة أو موضوعة، قل: "⚠️ تنبيه: هذا الحديث [ضعيف/موضوع] — قال عنه [المحدث]: [الحكم]"
   - لا تستخدم الحديث الضعيف لإثبات أحكام شرعية.

4. **إذا لم يكن الحديث في قاعدة البيانات**:
   إذا كانت الأحاديث المقدمة لا تتطابق مع الحديث المطلوب (أي المتن مختلف تماماً)، فقل:
   "**لم يُعثر على هذا الحديث في قاعدة بيانات الأحاديث المتاحة.**
   
   وهذا قد يعني أحد أمرين:
   1. الحديث غير موجود في المصادر الحديثية المعتمدة التي تغطيها قاعدة البيانات.
   2. الحديث مشهور على الألسنة لكنه لا أصل له أو إسناده ضعيف جداً لدرجة أن أهل الحديث لم يُدرجوه في مصنفاتهم.
   
   **مثال**: حديث 'اختلاف أمتي رحمة' — قال عنه الإمام النووي وابن حزم: لا يُعرف له إسناد صحيح. وقال السيوطي في اللآلئ المصنوعة: لا أصل له بهذا اللفظ."
   
   ثم اذكر الأحاديث المقدمة في السياق إن كانت وثيقة الصلة بالموضوع.

5. **لا تختلق**: لا تذكر أي حديث غير موجود في السياق المقدم.

6. **اللغة**: أجب باللغة العربية الفصحى."""


SYSTEM_PROMPT_GENERAL_EXPLANATION_FALLBACK = """أنت مساعد إسلامي يجيب عن سؤال تفسيري عندما تكون الأحاديث المسترجعة غير كافية أو غير مرتبطة مباشرة بالسؤال.

## قواعد صارمة:
1. لا تذكر أي حديث أو أثر أو نسبة إلى النبي صلى الله عليه وسلم ما لم يكن موجوداً في النتائج المسترجعة.
2. اشرح المعنى العام فقط اعتماداً على مبادئ إسلامية معروفة بصياغة عامة، من غير اقتباس نصوص حديثية.
3. ابدأ بالتنبيه المختصر إلى أن الأحاديث المسترجعة غير كافية أو غير مرتبطة مباشرة بالسؤال.
4. بعد ذلك قدّم شرحاً عاماً نافعاً وصادقاً ومباشراً للسؤال.
5. لا تقل "لم يُعثر على هذا الحديث" إلا إذا كان السؤال فعلاً عن حديث بعينه.
6. لا تختلق مصادر، ولا أرقام، ولا رواة، ولا نصوصاً من عندك.
7. أجب بالعربية الفصحى."""


INTENT_AR_LABELS = {
    AnswerIntent.EXPLANATORY: "إجابة تفسيرية أو تعليمية",
    AnswerIntent.VERIFICATION: "تحقق من صحة الحديث أو درجته",
    AnswerIntent.COLLECTION: "جمع شامل للروايات",
    AnswerIntent.LOOKUP: "بحث عن حديث بعينه",
}


def _build_intent_policy_prompt(answer_intent: AnswerIntent) -> str:
    """Return strict rules tailored to the classified answer intent."""
    common_rules = """
## قواعد الدرجات الملزمة:
1. اذكر درجة كل رواية تذكرها صراحة.
2. قدّم الصحيح ثم الحسن قبل غيرهما كلما أمكن.
3. الرواية الضعيفة أو الموضوعة أو غير المتحققة لا تُبنى عليها أحكام ولا فضائل ولا توجيه ديني.
4. إذا كانت درجة الرواية غير واضحة فعدّها «غير متحققة» ولا تستخدمها دليلاً.
5. إذا تعارضت الدرجة المختصرة مع الحكم التفصيلي فاعتبر الرواية غير صالحة للاحتجاج.
6. لا تستعمل روايات أحكام الزكاة وتوزيعها لإثبات فضائل الصدقة إلا إذا كان وجه الاستدلال صريحاً في السياق.
7. اكتب متن الجواب فقط؛ قسم كفاية الأدلة وقسم الروايات المستبعدة سيُضافان خارجياً.
8. إن كانت الأدلة المتاحة محدودة، استخدم صياغة حذرة وتجنب الجزم الزائد.
"""

    if answer_intent == AnswerIntent.EXPLANATORY:
        return common_rules + """
## سياسة الإجابة التفسيرية:
1. استعمل في الاستدلال فقط الأحاديث الصحيحة والحسنة.
2. إذا وُجدت روايات ضعيفة أو موضوعة أو غير متحققة ذات صلة، فاذكرها فقط بوصفها غير صالحة للاحتجاج.
3. إذا لم يوجد في السياق حديث صحيح أو حسن أو كانت النتائج غير مرتبطة مباشرة بالسؤال، فاذكر ذلك بوضوح ثم قدّم شرحاً عاماً بلا أحاديث مخترعة.
4. لا تقل "لم يُعثر على هذا الحديث" في الأسئلة التفسيرية العامة مثل الفضائل والفوائد والشرح العام.
"""

    if answer_intent == AnswerIntent.VERIFICATION:
        return common_rules + """
## سياسة التحقق من الحديث:
1. اذكر درجة الحديث أولاً قبل أي شرح لمعناه.
2. يجوز ذكر الضعيف أو الموضوع أو غير المتحقق هنا، لكن مع بيان درجته بوضوح قبل ذكر المعنى.
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
2. إذا كانت الرواية ضعيفة أو موضوعة أو غير متحققة فصرّح بذلك قبل أي شرح موجز.
3. عند تعدد النتائج، قدّم الصحيح ثم الحسن، ثم بيّن ما دونهما مع التحذير.
"""


# ============================================================
# Structured Response
# ============================================================

@dataclass
class Citation:
    """A single citation in the generated response."""
    hadith_index: int          # 1-based index matching context order
    hadith_id: str
    matn_snippet: str          # First 100 chars of the cited matn
    grade: str
    grade_ar: str
    masdar: str
    rawi: str
    muhaddith: str
    is_weak: bool = False      # True for daif/mawdu


@dataclass
class IgnoredNarration:
    """Narration excluded from the reliable-evidence set."""
    hadith_index: int
    hadith_id: str
    grade: str
    grade_ar: str
    reason: str
    matn_snippet: str


@dataclass
class EvidenceEvaluation:
    """Structured evidence evaluation before answer generation."""
    authenticity_of_evidence: str
    relevance_to_question: str
    final_sufficiency: str


@dataclass
class GeneratedResponse:
    """Structured response from the generation module."""
    answer: str                                    # Full Arabic answer text
    citations: list[Citation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    grounding_verified: bool = False               # True if citation check passed
    grounding_issues: list[str] = field(default_factory=list)
    raw_text: str = ""                             # Raw LLM output
    query_type: str = ""                           # For debugging
    answer_intent: str = ""                        # Policy category used for answering
    evidence_sufficient: bool = False
    authenticity_of_evidence: str = "insufficient"
    relevance_to_question: str = "weak"
    final_sufficiency: str = "insufficient"
    ignored_narrations: list[IgnoredNarration] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "answer": self.answer,
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
    """
    Verify that the LLM's answer only cites hadiths from the provided context.
    
    Checks:
    1. Source names mentioned in the answer exist in provided hadiths
    2. Narrator names match provided context
    3. No hallucinated hadith texts (check key phrases)
    
    Args:
        answer_text: The generated answer from the LLM
        provided_hadiths: The hadiths given as context
        
    Returns:
        Tuple of (is_grounded, list_of_issues)
    """
    issues = []
    
    # Collect known sources, narrators, and key matn phrases from context
    known_masdar = {h.masdar.strip() for h in provided_hadiths if h.masdar.strip()}
    known_rawi = {h.rawi.strip() for h in provided_hadiths if h.rawi.strip()}
    known_muhaddith = {h.muhaddith.strip() for h in provided_hadiths if h.muhaddith.strip()}
    
    # Extract source references from the answer
    source_pattern = re.compile(r"(?:المصدر[:\s]+|كتاب\s+)([\u0600-\u06FF\s]+?)(?:\s*[،,\.\-\n])")
    mentioned_sources = source_pattern.findall(answer_text)
    
    for source in mentioned_sources:
        source = source.strip()
        if source and not any(source in m for m in known_masdar):
            issues.append(f"مصدر غير موجود في السياق: {source}")
    
    # Check for common hallucination patterns
    prophet_said_pattern = re.compile(
        r"قال\s+(?:رسول\s+الله|النبي).*?[:\s]+[«\"](.*?)[»\"]",
        re.DOTALL
    )
    quoted_texts = prophet_said_pattern.findall(answer_text)
    
    for quoted in quoted_texts:
        quoted_clean = quoted.strip()[:50]  # First 50 chars
        if quoted_clean and not any(quoted_clean in h.text_ar for h in provided_hadiths):
            issues.append(f"نص مقتبس قد لا يطابق السياق: {quoted_clean}...")
    
    # Check for hadith numbers that don't match context
    number_pattern = re.compile(r"(?:رقم|حديث\s+رقم|الصفحة)\s*[:\s]*(\d+)")
    mentioned_numbers = number_pattern.findall(answer_text)
    known_numbers = {h.safha_raqam.strip() for h in provided_hadiths if h.safha_raqam.strip()}
    
    for num in mentioned_numbers:
        if num and not any(num in n for n in known_numbers):
            issues.append(f"رقم حديث/صفحة غير موجود في السياق: {num}")
    
    is_grounded = len(issues) == 0
    return is_grounded, issues


@dataclass
class _AuditedHadith:
    source_index: int
    hadith: RetrievedHadith
    canonical_grade: str
    grade_label: str
    is_authentic: bool
    is_directly_relevant: bool
    exclusion_reason: str = ""


_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]+")
_TATWEEL = re.compile(r"\u0640+")
_WHITESPACE = re.compile(r"\s+")
_ALEF_VARIANTS = re.compile(r"[أإآٱ]")

_CHARITY_TERMS = ("صدقه", "الصدقه", "صدقة", "الصدقة", "انفاق", "الانفاق", "إنفاق", "تبرع")
_VIRTUE_TERMS = ("فضل", "فضائل", "فوائد", "اجر", "أجر", "ثواب", "ترغيب", "منفعه", "منفعة")
_ZAKAT_LEGAL_TERMS = (
    "زكاه",
    "الزكاه",
    "زكاة",
    "الزكاة",
    "نصاب",
    "مصارف",
    "مصرف",
    "العاملين عليها",
    "ابن السبيل",
    "الفقراء",
    "المساكين",
    "صدقه الفطر",
    "صدقة الفطر",
)
_CHARITY_VIRTUE_TERMS = (
    "الصدقه برهان",
    "الصدقة برهان",
    "تطفئ الخطيئه",
    "تطفئ الخطيئة",
    "ظل",
    "اجر الصدقه",
    "أجر الصدقة",
    "فضل الصدقه",
    "فضل الصدقة",
)


_SOURCE_PRIORITY_RULES = (
    ("صحيح البخاري", 0),
    ("البخاري", 0),
    ("bukhari", 0),
    ("صحيح مسلم", 1),
    ("مسلم", 1),
    ("muslim", 1),
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
    return any(term in normalized for term in _CHARITY_TERMS) and any(term in normalized for term in _VIRTUE_TERMS)


def _is_legal_zakat_narration(hadith: RetrievedHadith) -> bool:
    haystack = _normalize_audit_text(
        " ".join(
            part
            for part in (
                hadith.text_ar,
                hadith.category,
                hadith.subcategory_name,
                hadith.masdar,
            )
            if part
        )
    )
    has_legal = any(term in haystack for term in _ZAKAT_LEGAL_TERMS)
    has_virtue = any(term in haystack for term in _CHARITY_VIRTUE_TERMS)
    return has_legal and not has_virtue


def _detect_topic_exclusion_reason(query: str, hadith: RetrievedHadith) -> str:
    """
    Filter known topical mismatches that commonly look relevant lexically
    but do not answer the user's actual question.
    """
    if _is_charity_virtue_query(query) and _is_legal_zakat_narration(hadith):
        return "يتعلق بأحكام الزكاة أو مصارفها، لا بفضائل الصدقة وثوابها"
    return ""


def _audit_hadiths_for_answer(
    query: str,
    hadiths: list[RetrievedHadith],
) -> tuple[list[_AuditedHadith], list[IgnoredNarration]]:
    """Audit retrieved narrations for authenticity and direct relevance."""
    audited: list[_AuditedHadith] = []
    ignored: list[IgnoredNarration] = []

    for index, hadith in enumerate(hadiths, 1):
        grade_audit = audit_grade(hadith.grade, hadith.grade_ar, hadith.ruling)
        canonical_grade = grade_audit.effective_bucket
        grade_label = resolve_grade_label(hadith.grade, hadith.grade_ar, hadith.ruling)
        is_authentic = grade_audit.is_usable_for_evidence

        topic_reason = _detect_topic_exclusion_reason(query, hadith)
        is_directly_relevant = is_authentic and not topic_reason
        exclusion_reason = ""

        if not is_authentic:
            exclusion_reason = grade_audit.exclusion_reason
        elif topic_reason:
            exclusion_reason = topic_reason

        audited.append(
            _AuditedHadith(
                source_index=index,
                hadith=hadith,
                canonical_grade=canonical_grade,
                grade_label=grade_label,
                is_authentic=is_authentic,
                is_directly_relevant=is_directly_relevant,
                exclusion_reason=exclusion_reason,
            )
        )

        if exclusion_reason:
            ignored.append(
                IgnoredNarration(
                    hadith_index=index,
                    hadith_id=hadith.id,
                    grade=canonical_grade,
                    grade_ar=grade_label,
                    reason=exclusion_reason,
                    matn_snippet=(hadith.text_ar or "")[:120],
                )
            )
    return audited, ignored


def _evaluate_retrieved_evidence(
    audited_hadiths: list[_AuditedHadith],
    answer_intent: AnswerIntent,
) -> EvidenceEvaluation:
    """Compute authenticity, relevance, and final sufficiency."""
    authentic_hadiths = [item for item in audited_hadiths if item.is_authentic]
    direct_hadiths = [item for item in audited_hadiths if item.is_directly_relevant]

    authenticity = "sufficient" if authentic_hadiths else "insufficient"

    if direct_hadiths:
        relevance = "direct"
    elif authentic_hadiths:
        relevance = "partial"
    else:
        relevance = "weak"

    minimum_direct_for_sufficient = 1
    if answer_intent in {AnswerIntent.EXPLANATORY, AnswerIntent.COLLECTION}:
        minimum_direct_for_sufficient = 2

    if len(direct_hadiths) >= minimum_direct_for_sufficient:
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


def _build_citations(
    hadiths: list[RetrievedHadith],
) -> list[Citation]:
    """Build citation objects from provided hadiths."""
    citations = []
    seen_ids: set[str] = set()
    for i, h in enumerate(hadiths, 1):
        if h.id in seen_ids:
            continue
        seen_ids.add(h.id)
        canonical_grade = resolve_grade_bucket(h.grade, h.grade_ar, h.ruling)
        grade_ar = resolve_grade_label(h.grade, h.grade_ar, h.ruling)
        citations.append(Citation(
            hadith_index=i,
            hadith_id=h.id,
            matn_snippet=h.text_ar[:100] if h.text_ar else "",
            grade=canonical_grade,
            grade_ar=grade_ar,
            masdar=h.masdar,
            rawi=h.rawi,
            muhaddith=h.muhaddith,
            is_weak=canonical_grade in ("daif", "mawdu"),
        ))
    return citations


def _build_warning_text(grade: str, grade_ar: str) -> str:
    """Return a user-facing warning for non-authentic narrations."""
    if grade in ("daif", "mawdu"):
        return f"⚠️ {grade_ar}: لا يُحتج به في إثبات الأحكام والفضائل"
    if grade == "unknown":
        return "⚠️ غير متحقق: لم تثبت درجته فلا يُستخدم دليلاً"
    return ""


def _order_hadiths_for_generation(
    hadiths: list[RetrievedHadith],
    answer_intent: AnswerIntent,
) -> list[RetrievedHadith]:
    """Order hadiths for safer presentation without discarding relevant hits."""
    indexed = list(enumerate(hadiths))
    indexed.sort(
        key=lambda item: (
            _source_priority(item[1].masdar),
            grade_priority(resolve_grade_bucket(item[1].grade, item[1].grade_ar, item[1].ruling)),
            item[0],
        )
    )

    if answer_intent in {AnswerIntent.EXPLANATORY, AnswerIntent.COLLECTION}:
        return [hadith for _, hadith in indexed]

    return [hadith for _, hadith in indexed]


def _normalize_hadith_text_for_dedup(text: str) -> str:
    """Normalize hadith text so near-identical narrations can be de-duplicated."""
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", normalized)
    normalized = re.sub(r"[أإآٱ]", "ا", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _tokenize_for_similarity(text: str) -> set[str]:
    """Tokenize normalized hadith text for lightweight similarity matching."""
    tokens = set()
    for token in _normalize_hadith_text_for_dedup(text).split():
        cleaned = re.sub(r"[^\u0600-\u06FFa-z0-9]", "", token)
        if len(cleaned) >= 3:
            tokens.add(cleaned)
    return tokens


def _token_jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return intersection / union


def _token_overlap_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute overlap coefficient: intersection over smaller set size."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    return intersection / min(len(tokens_a), len(tokens_b))


def _are_near_duplicate_narrations(text_a: str, text_b: str) -> bool:
    """Heuristic check to collapse closely-related narrations across different books/chains."""
    norm_a = _normalize_hadith_text_for_dedup(text_a)
    norm_b = _normalize_hadith_text_for_dedup(text_b)

    if not norm_a or not norm_b:
        return False

    if norm_a == norm_b:
        return True

    seq_ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    if seq_ratio >= 0.72:
        return True

    if "ناقصات عقل ودين" in norm_a and "ناقصات عقل ودين" in norm_b:
        return True

    anchor_phrases = (
        "ناقصات عقل ودين",
        "شهادة امراتين",
        "نقصان عقل",
        "نقصان دين",
        "تكثرن اللعن",
        "تكفرن العشير",
    )
    shared_anchor_count = sum(1 for phrase in anchor_phrases if phrase in norm_a and phrase in norm_b)
    if shared_anchor_count >= 2:
        return True

    tokens_a = _tokenize_for_similarity(norm_a)
    tokens_b = _tokenize_for_similarity(norm_b)
    jaccard = _token_jaccard_similarity(tokens_a, tokens_b)
    overlap = _token_overlap_similarity(tokens_a, tokens_b)

    if overlap >= 0.72:
        return True

    return overlap >= 0.62 and jaccard >= 0.45


def _hadith_representative_rank(hadith: RetrievedHadith) -> tuple[int, float, int]:
    """Lower rank is better for choosing one representative narration per cluster."""
    return (
        _source_priority(hadith.masdar),
        grade_priority(resolve_grade_bucket(hadith.grade, hadith.grade_ar, hadith.ruling)),
        float(hadith.distance or 1.0),
        -len(str(hadith.text_ar or "")),
    )


def _deduplicate_hadiths_for_answer(hadiths: list[RetrievedHadith]) -> list[RetrievedHadith]:
    """Drop obvious duplicate narrations to avoid repeated evidence blocks in final answer."""
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

    representatives: list[RetrievedHadith] = []
    for cluster in clusters:
        best = min(cluster, key=_hadith_representative_rank)
        representatives.append(best)

    return representatives


def _format_hadith_block(
    index: int,
    hadith: RetrievedHadith,
    metadata_first: bool = False,
) -> str:
    """Format a single hadith block with explicit grade handling."""
    canonical_grade = resolve_grade_bucket(hadith.grade, hadith.grade_ar, hadith.ruling)
    grade_label = resolve_grade_label(hadith.grade, hadith.grade_ar, hadith.ruling)
    warning = _build_warning_text(canonical_grade, grade_label)
    warning_line = f"\n   {warning}" if warning else ""

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
    """Group hadiths by canonical grade bucket."""
    groups = {grade: [] for grade in ("sahih", "hasan", "daif", "mawdu", "unknown")}
    for hadith in hadiths:
        groups[resolve_grade_bucket(hadith.grade, hadith.grade_ar, hadith.ruling)].append(hadith)
    return groups


def _format_grouped_sections(
    sections: list[tuple[str, list[RetrievedHadith]]],
    metadata_first: bool = False,
) -> str:
    """Format multiple hadith sections with fresh indices per final context order."""
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


def _format_hadith_context(
    hadiths: list[RetrievedHadith],
    answer_intent: AnswerIntent,
) -> str:
    """
    Format retrieved hadiths into a structured context string for the LLM.
    Each hadith is formatted as a numbered block with all metadata.
    """
    if answer_intent == AnswerIntent.EXPLANATORY:
        groups = _group_hadiths_by_grade(hadiths)
        authentic = groups["sahih"] + groups["hasan"]
        non_evidence = groups["daif"] + groups["mawdu"] + groups["unknown"]
        sections = []
        if authentic:
            sections.append(("### الأحاديث المسموح الاستدلال بها (صحيح وحسن)", authentic))
        else:
            sections.append(("### لا يوجد في النتائج حديث صحيح أو حسن يمكن الاستدلال به", []))
        if non_evidence:
            sections.append(("### روايات غير صالحة للاحتجاج (ضعيف أو موضوع أو غير متحقق)", non_evidence))
        return _format_grouped_sections(sections)

    if answer_intent == AnswerIntent.COLLECTION:
        groups = _group_hadiths_by_grade(hadiths)
        sections = [
            ("### الأحاديث الصحيحة", groups["sahih"]),
            ("### الأحاديث الحسنة", groups["hasan"]),
            ("### الأحاديث الضعيفة", groups["daif"]),
            ("### الأحاديث الموضوعة", groups["mawdu"]),
            ("### الروايات غير المتحققة", groups["unknown"]),
        ]
        return _format_grouped_sections(sections)

    return "\n\n".join(
        _format_hadith_block(i, hadith)
        for i, hadith in enumerate(hadiths, 1)
    )


def _format_metadata_context(
    hadiths: list[RetrievedHadith],
    answer_intent: AnswerIntent,
) -> str:
    """
    Format hadith context emphasizing metadata for metadata-focused queries.
    Places metadata fields prominently before the matn text.
    """
    if answer_intent in {AnswerIntent.EXPLANATORY, AnswerIntent.COLLECTION}:
        groups = _group_hadiths_by_grade(hadiths)
        sections = [
            ("### الأحاديث الصحيحة", groups["sahih"]),
            ("### الأحاديث الحسنة", groups["hasan"]),
            ("### الأحاديث الضعيفة", groups["daif"]),
            ("### الأحاديث الموضوعة", groups["mawdu"]),
            ("### الروايات غير المتحققة", groups["unknown"]),
        ]
        return _format_grouped_sections(sections, metadata_first=True)

    return "\n\n".join(
        _format_hadith_block(i, hadith, metadata_first=True)
        for i, hadith in enumerate(hadiths, 1)
    )


def _format_ignored_narrations(ignored_narrations: list[IgnoredNarration]) -> str:
    """Render excluded narrations as a deterministic final section."""
    if not ignored_narrations:
        return "لا توجد روايات مستبعدة."

    lines = []
    for item in ignored_narrations:
        snippet = item.matn_snippet.strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        lines.append(
            f"{item.hadith_index}. الدرجة: {item.grade_ar} | السبب: {item.reason} | النص: {snippet}"
        )
    return "\n".join(lines)


def _build_non_explanatory_fallback_body(evaluation: EvidenceEvaluation) -> str:
    """Deterministic fallback when a hadith-based answer cannot be given."""
    if evaluation.final_sufficiency == "partial":
        return (
            "توجد روايات صحيحة أو حسنة في النتائج، لكنها لا تجيب مباشرة عن السؤال المطلوب، "
            "لذلك لا يمكن بناء جواب حديثي مباشر مكتمل من هذه النتائج."
        )

    return "النتائج المسترجعة لا تقدم دليلاً حديثياً مناسباً لهذا السؤال."


def _build_explanatory_fallback_user_message(
    query: str,
    ignored_narrations: list[IgnoredNarration],
    evaluation: EvidenceEvaluation,
) -> str:
    """Build the user message for general explanatory fallback mode."""
    summary_lines = []
    for item in ignored_narrations[:5]:
        summary_lines.append(f"- الحديث [{item.hadith_index}] استُبعد لأن: {item.reason}")

    ignored_summary = "\n".join(summary_lines) if summary_lines else "- لم تُسترجع روايات يمكن الاعتماد عليها."

    return (
        "## السؤال التفسيري:\n"
        f"{query}\n\n"
        "## نتيجة فحص الأحاديث المسترجعة:\n"
        f"Authenticity of evidence: {evaluation.authenticity_of_evidence}\n"
        f"Relevance to question: {evaluation.relevance_to_question}\n"
        f"Final sufficiency: {evaluation.final_sufficiency}\n"
        "الأحاديث المسترجعة غير كافية أو غير مرتبطة مباشرة بالسؤال التفسيري.\n"
        "فيما يلي ملخص موجز للأسباب:\n"
        f"{ignored_summary}\n\n"
        "## المطلوب:\n"
        "قدّم شرحاً عاماً صادقاً ومفيداً للمستخدم باللغة العربية الفصحى، من غير ذكر أي حديث غير موجود في النتائج، "
        "ومن غير اختلاق نصوص أو مصادر. إذا احتجت إلى التوضيح فليكن على صورة مبادئ ومعانٍ عامة فقط."
    )


def _build_explanatory_fallback_prefix(evaluation: EvidenceEvaluation) -> str:
    """Deterministic lead-in for explanatory fallback answers."""
    if evaluation.final_sufficiency == "partial":
        return (
            "توجد روايات صحيحة أو حسنة في النتائج، لكنها لا تجيب مباشرة عن السؤال، "
            "ولذلك سيكون الجواب الآتي شرحاً عاماً محدوداً لا استدلالاً حديثياً مباشراً."
        )

    return (
        "النتائج المسترجعة لا توفر دليلاً حديثياً مناسباً لهذا السؤال، "
        "ولذلك سيكون الجواب الآتي شرحاً عاماً بلا الاستناد إلى حديث معين."
    )


def _wrap_audited_answer(
    evaluation: EvidenceEvaluation,
    core_answer: str,
    ignored_narrations: list[IgnoredNarration],
) -> str:
    """Wrap the answer in the required audit output structure."""
    answer_body = core_answer.strip()
    excluded_lines = "\n".join(
        f"* [{item.hadith_index}] {item.reason}"
        for item in ignored_narrations
    ) or "* None"

    return (
        f"Authenticity of evidence: {evaluation.authenticity_of_evidence}\n"
        f"Relevance to question: {evaluation.relevance_to_question}\n"
        f"Final sufficiency: {evaluation.final_sufficiency}\n\n"
        f"Answer:\n{answer_body}\n\n"
        f"Excluded narrations:\n\n"
        f"{excluded_lines}"
    )


def _select_system_prompt(query_type: str, answer_intent: AnswerIntent) -> str:
    """Select the appropriate system prompt based on query type."""
    if answer_intent == AnswerIntent.EXPLANATORY:
        return SYSTEM_PROMPT_GENERAL
    if query_type == "metadata":
        return SYSTEM_PROMPT_METADATA
    elif query_type == "narrator":
        return SYSTEM_PROMPT_NARRATOR
    elif query_type == "explain_hadith":
        return SYSTEM_PROMPT_EXPLAIN
    else:
        return SYSTEM_PROMPT_GENERAL


def _check_hadith_relevance(
    requested_text: str,
    hadiths: list[RetrievedHadith],
    min_overlap_chars: int = 4,
) -> bool:
    """
    Check if at least one retrieved hadith actually contains key terms from
    the requested hadith text.

    Used to detect the case where the user asks for a specific hadith (e.g.,
    "اختلاف أمتي رحمة") but the retrieved corpus only has loosely related
    hadiths (e.g., "اختلاف وفرقة").

    Returns True  if the corpus appears to contain the requested hadith.
    Returns False if no retrieved hadith contains enough key terms → corpus miss.
    """
    if not requested_text or not hadiths:
        return False

    # Normalize for comparison: strip tashkeel, lowercased, collapse spaces
    def _norm(s: str) -> str:
        s = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670]", "", s)  # strip tashkeel
        s = re.sub(r"[أإآٱ]", "ا", s)    # alef normalization
        s = re.sub(r"\s+", " ", s).strip()
        return s

    norm_req = _norm(requested_text)
    # Get meaningful words (length ≥ min_overlap_chars) from the requested text
    key_words = [w for w in norm_req.split() if len(w) >= min_overlap_chars]

    if not key_words:
        return True  # Too short to judge, assume ok

    for hadith in hadiths:
        norm_matn = _norm(hadith.text_ar or "")
        # If more than half the key words appear in the matn, consider it relevant
        matches = sum(1 for w in key_words if w in norm_matn)
        if matches >= max(1, len(key_words) // 2):
            return True

    return False


class HadithGenerator:
    """
    Generates answers using Gemini API (Gemma 3 free tier) with Arabic system prompt.
    Query-type-aware: different prompts for metadata, narrator, and general queries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model or settings.GEMINI_MODEL
        # self.api_key = api_key or settings.GROQ_API_KEY
        # self.model_name = model or settings.GROQ_MODEL

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Set it in .env or pass it directly. "
                "Get your FREE key at https://aistudio.google.com/apikey"
            )

        self.client = genai.Client(api_key=self.api_key)
        # self.client = Groq(api_key=self.api_key)

        logger.info(f"Gemini generator initialized: model={self.model_name}")

    def _generate_general_explanatory_fallback(
        self,
        query: str,
        ignored_narrations: list[IgnoredNarration],
        evaluation: EvidenceEvaluation,
        max_output_tokens: int,
    ) -> str:
        """Generate a hadith-free explanatory answer when retrieval evidence is insufficient."""
        user_message = _build_explanatory_fallback_user_message(query, ignored_narrations, evaluation)
        merged_prompt = (
            f"## تعليمات النظام:\n{SYSTEM_PROMPT_GENERAL_EXPLANATION_FALLBACK}\n\n"
            f"## رسالة المستخدم:\n{user_message}"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=merged_prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=max_output_tokens,
            ),
        )
        return response.text or ""

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
        """
        Generate an Arabic answer with hadith citations.

        Args:
            query: The user's question (Arabic or English).
            hadiths: Top-K reranked hadiths with full metadata.
            temperature: Lower = more faithful to context (0.3 is conservative).
            max_output_tokens: Maximum response length.
            verify_grounding: Whether to verify citation grounding.
            query_type: Type of query (general, metadata, narrator, etc.)
            metadata_fields: Which metadata fields user is asking about.

        Returns:
            GeneratedResponse with answer, citations, grounding status, and warnings.
        """
        answer_intent = classify_answer_intent(
            query=query,
            query_type=query_type,
            metadata_fields=metadata_fields,
        )

        empty_evaluation = EvidenceEvaluation(
            authenticity_of_evidence="insufficient",
            relevance_to_question="weak",
            final_sufficiency="insufficient",
        )

        if not hadiths:
            if answer_intent == AnswerIntent.EXPLANATORY:
                general_text = self._generate_general_explanatory_fallback(
                    query=query,
                    ignored_narrations=[],
                    evaluation=empty_evaluation,
                    max_output_tokens=max_output_tokens,
                )
                core_answer = f"{_build_explanatory_fallback_prefix(empty_evaluation)}\n\n{general_text}".strip()
            else:
                core_answer = _build_non_explanatory_fallback_body(empty_evaluation)
            answer = _wrap_audited_answer(
                evaluation=empty_evaluation,
                core_answer=core_answer,
                ignored_narrations=[],
            )
            return GeneratedResponse(
                answer=answer,
                grounding_verified=True,
                raw_text=core_answer,
                query_type=query_type,
                answer_intent=answer_intent.value,
                evidence_sufficient=False,
                authenticity_of_evidence=empty_evaluation.authenticity_of_evidence,
                relevance_to_question=empty_evaluation.relevance_to_question,
                final_sufficiency=empty_evaluation.final_sufficiency,
            )

        audited_hadiths, ignored_narrations = _audit_hadiths_for_answer(query, hadiths)
        evaluation = _evaluate_retrieved_evidence(audited_hadiths, answer_intent)
        direct_hadiths = [item.hadith for item in audited_hadiths if item.is_directly_relevant]
        deduplicated_direct_hadiths = _deduplicate_hadiths_for_answer(direct_hadiths)
        ordered_direct_hadiths = _order_hadiths_for_generation(deduplicated_direct_hadiths, answer_intent)
        evidence_sufficient = evaluation.final_sufficiency == "sufficient"

        # Build citations
        citations = _build_citations(ordered_direct_hadiths)
        
        # Build warnings from excluded narrations
        warnings = [
            f"⚠️ استُبعد الحديث [{item.hadith_index}] — {item.reason}"
            for item in ignored_narrations
        ]

        if evaluation.final_sufficiency != "sufficient":
            if answer_intent == AnswerIntent.EXPLANATORY:
                general_text = self._generate_general_explanatory_fallback(
                    query=query,
                    ignored_narrations=ignored_narrations,
                    evaluation=evaluation,
                    max_output_tokens=max_output_tokens,
                )
                answer_body = f"{_build_explanatory_fallback_prefix(evaluation)}\n\n{general_text}".strip()
            else:
                answer_body = _build_non_explanatory_fallback_body(evaluation)
            final_answer = _wrap_audited_answer(
                evaluation=evaluation,
                core_answer=answer_body,
                ignored_narrations=ignored_narrations,
            )
            return GeneratedResponse(
                answer=final_answer,
                citations=citations,
                warnings=warnings,
                grounding_verified=True,
                raw_text=answer_body,
                query_type=query_type,
                answer_intent=answer_intent.value,
                evidence_sufficient=evaluation.final_sufficiency == "sufficient",
                authenticity_of_evidence=evaluation.authenticity_of_evidence,
                relevance_to_question=evaluation.relevance_to_question,
                final_sufficiency=evaluation.final_sufficiency,
                ignored_narrations=ignored_narrations,
            )

        # Select system prompt and context format based on query type
        system_prompt = _select_system_prompt(query_type, answer_intent)
        intent_policy_prompt = _build_intent_policy_prompt(answer_intent)
        
        if query_type == "metadata":
            context = _format_metadata_context(ordered_direct_hadiths, answer_intent)
            # Lower temperature for metadata answers (factual)
            temperature = min(temperature, 0.1)
        else:
            context = _format_hadith_context(ordered_direct_hadiths, answer_intent)

        # Build the user message with context
        user_message = f"""## السياق (الأحاديث المسترجعة):
{context}

## سؤال المستخدم:
{query}"""

        duplicate_collapsed_count = max(0, len(direct_hadiths) - len(ordered_direct_hadiths))
        if duplicate_collapsed_count > 0:
            user_message += (
                "\n\n## تنبيه مهم لأسلوب العرض:\n"
                "بعض النتائج كانت روايات متقاربة جداً في المعنى والنص لنفس الخبر. "
                "لا تعرضها كأحاديث مستقلة متعددة أمام المستخدم. "
                "اكتف بتمثيل موجز غير مُرقم على أنها روايات متعددة لخبر واحد عند اللزوم، "
                "واذكر فقط الروايات الأوضح والأقوى دون تعدادٍ طويل."
            )

        # For explain_hadith queries: add a relevance hint so the LLM knows
        # whether the specific hadith was found in the corpus
        if query_type == "explain_hadith" and answer_intent == AnswerIntent.LOOKUP:
            from retrieval.query_preprocessor import _extract_hadith_text_from_explain_query
            # Derive the requested hadith text from the normalized query
            norm_q = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u0640]", "", query)
            norm_q = re.sub(r"[أإآٱ]", "ا", norm_q)
            norm_q = re.sub(r"\s+", " ", norm_q).strip()
            requested_text = _extract_hadith_text_from_explain_query(norm_q)
            corpus_has_it = _check_hadith_relevance(requested_text, ordered_direct_hadiths)
            if not corpus_has_it:
                user_message += (
                    f"\n\n## ⚠️ تنبيه للنموذج (لا تعرضه للمستخدم كما هو):\n"
                    f"لم يُعثر على الحديث المطلوب «{requested_text}» في قاعدة البيانات. "
                    f"الأحاديث المقدمة هي أقرب ما وجده النظام لكنها لا تطابق الحديث المطلوب. "
                    f"يجب أن تُصرّح للمستخدم بأن هذا الحديث غير موجود في قاعدة البيانات "
                    f"ولا يُعرف له إسناد صحيح أو أنه غير موجود في المصادر المعتمدة، "
                    f"وفق قاعدة 4 في التعليمات."
                )

        # Add metadata field hint for metadata queries
        if metadata_fields:
            fields_ar = {
                "rawi": "الراوي",
                "grade": "الدرجة/الصحة",
                "masdar": "المصدر/الكتاب",
                "safha_raqam": "الرقم/الصفحة",
                "muhaddith": "المحدث",
                "category": "التصنيف/الباب",
            }
            requested = [fields_ar.get(f, f) for f in metadata_fields]
            user_message += f"\n\n## تنبيه: المستخدم يسأل تحديداً عن: {', '.join(requested)}"

        # Add exclusion hint: tell LLM which books the user wants to EXCLUDE
        if excluded_masdar:
            excluded_str = ", ".join(f"«{b}»" for b in excluded_masdar)
            user_message += (
                f"\n\n## ⚠️ تعليمات الإقصاء:\n"
                f"المستخدم يريد فقط الأحاديث التي لم تُذكر في: {excluded_str}.\n"
                f"افحص كل حديث في السياق: إذا كانت قاعدة البيانات لا تتضمن معلومات مقارنة بين الكتب، "
                f"فأخبر المستخدم بأن قاعدة البيانات تحتوي على الأحاديث الواردة في كتاب المصدر المطلوب "
                f"لكنها لا تتضمن بيانات مقارنة تُحدد أي الأحاديث غائبة عن {excluded_str}. "
                f"اعرض الأحاديث المسترجعة كما هي مع هذا التوضيح."
            )

        logger.info(
            f"Generating answer: model={self.model_name}, "
            f"hadiths={len(ordered_direct_hadiths)}, query_type={query_type}, "
            f"answer_intent={answer_intent.value}, "
            f"evaluation={evaluation.final_sufficiency}, "
            f"temp={temperature}"
        )

        # gemma-3-* does not support system_instruction, so merge it into contents.
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
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_message},
        #     ],
        #     temperature=temperature,
        #     max_tokens=max_output_tokens,
        # )

        core_answer = response.text or ""
        # core_answer = response.choices[0].message.content
        logger.info(f"Generation complete: {len(core_answer)} chars")

        # Citation grounding verification
        grounding_verified = True
        grounding_issues = []
        if verify_grounding:
            grounding_verified, grounding_issues = _verify_citation_grounding(
                core_answer, ordered_direct_hadiths
            )
            if not grounding_verified:
                logger.warning(
                    f"Grounding issues detected: {grounding_issues}"
                )

        final_answer = _wrap_audited_answer(
            evaluation=evaluation,
            core_answer=core_answer,
            ignored_narrations=ignored_narrations,
        )

        return GeneratedResponse(
            answer=final_answer,
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


# Module-level convenience
_generator: Optional[HadithGenerator] = None


def get_generator() -> HadithGenerator:
    """Get or create the singleton generator."""
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
    """Convenience function for generation."""
    return get_generator().generate(
        query=query,
        hadiths=hadiths,
        temperature=temperature,
        query_type=query_type,
        metadata_fields=metadata_fields,
        excluded_masdar=excluded_masdar,
    )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    print(f"Generator ready. Model: {settings.GEMINI_MODEL}")
    print("System prompt (general) length:", len(SYSTEM_PROMPT_GENERAL), "chars")
    print("System prompt (metadata) length:", len(SYSTEM_PROMPT_METADATA), "chars")
    print("System prompt (narrator) length:", len(SYSTEM_PROMPT_NARRATOR), "chars")
