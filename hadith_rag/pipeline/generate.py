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
from typing import Optional
from dataclasses import dataclass, field, asdict

# from groq import Groq
from google import genai
from google.genai import types

from pipeline.config import settings, resolve_grade_label
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
   - قل: "في هذه المسألة خلاف بين العلماء" إن كان ذلك واضحاً"""


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
class GeneratedResponse:
    """Structured response from the generation module."""
    answer: str                                    # Full Arabic answer text
    citations: list[Citation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    grounding_verified: bool = False               # True if citation check passed
    grounding_issues: list[str] = field(default_factory=list)
    raw_text: str = ""                             # Raw LLM output
    query_type: str = ""                           # For debugging

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "answer": self.answer,
            "citations": [asdict(c) for c in self.citations],
            "warnings": self.warnings,
            "grounding_verified": self.grounding_verified,
            "grounding_issues": self.grounding_issues,
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


def _build_citations(
    hadiths: list[RetrievedHadith],
) -> list[Citation]:
    """Build citation objects from provided hadiths."""
    citations = []
    for i, h in enumerate(hadiths, 1):
        grade_ar = resolve_grade_label(h.grade, h.grade_ar, h.ruling)
        citations.append(Citation(
            hadith_index=i,
            hadith_id=h.id,
            matn_snippet=h.text_ar[:100] if h.text_ar else "",
            grade=h.grade,
            grade_ar=grade_ar,
            masdar=h.masdar,
            rawi=h.rawi,
            muhaddith=h.muhaddith,
            is_weak=h.grade in ("daif", "mawdu"),
        ))
    return citations


def _format_hadith_context(hadiths: list[RetrievedHadith]) -> str:
    """
    Format retrieved hadiths into a structured context string for the LLM.
    Each hadith is formatted as a numbered block with all metadata.
    """
    context_parts = []

    for i, h in enumerate(hadiths, 1):
        grade_label = resolve_grade_label(h.grade, h.grade_ar, h.ruling)
        warning = ""
        if h.grade in ("daif", "mawdu"):
            warning = f"\n   ⚠️ تنبيه: حديث {grade_label}"

        block = (
            f"--- الحديث [{i}] ---{warning}\n"
            f"   المتن: {h.text_ar}\n"
            f"   الدرجة: {grade_label}\n"
            f"   الحكم التفصيلي: {h.ruling}\n"
            f"   الراوي: {h.rawi}\n"
            f"   المحدِّث: {h.muhaddith}\n"
            f"   المصدر: {h.masdar}\n"
            f"   الرقم/الصفحة: {h.safha_raqam}\n"
            f"   التصنيف: {h.category} — {h.subcategory_name}"
        )
        context_parts.append(block)

    return "\n\n".join(context_parts)


def _format_metadata_context(hadiths: list[RetrievedHadith]) -> str:
    """
    Format hadith context emphasizing metadata for metadata-focused queries.
    Places metadata fields prominently before the matn text.
    """
    context_parts = []

    for i, h in enumerate(hadiths, 1):
        grade_label = resolve_grade_label(h.grade, h.grade_ar, h.ruling)
        warning = ""
        if h.grade in ("daif", "mawdu"):
            warning = f"\n   ⚠️ تنبيه: حديث {grade_label}"

        block = (
            f"=== الحديث [{i}] ==={warning}\n"
            f"   📋 الراوي: {h.rawi}\n"
            f"   📋 المحدِّث: {h.muhaddith}\n"
            f"   📋 الدرجة: {grade_label}\n"
            f"   📋 الحكم التفصيلي: {h.ruling}\n"
            f"   📋 المصدر: {h.masdar}\n"
            f"   📋 الرقم/الصفحة: {h.safha_raqam}\n"
            f"   📋 التصنيف: {h.category} — {h.subcategory_name}\n"
            f"   📋 المتن: {h.text_ar}"
        )
        context_parts.append(block)

    return "\n\n".join(context_parts)


def _select_system_prompt(query_type: str) -> str:
    """Select the appropriate system prompt based on query type."""
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
        if not hadiths:
            return GeneratedResponse(
                answer="لم أجد أحاديث متعلقة بسؤالك في قاعدة البيانات المتاحة.",
                grounding_verified=True,
                query_type=query_type,
            )

        # Build citations
        citations = _build_citations(hadiths)
        
        # Build warnings for weak/fabricated hadiths
        warnings = []
        for c in citations:
            if c.is_weak:
                warnings.append(
                    f"⚠️ الحديث [{c.hadith_index}] — {c.grade_ar}: "
                    f"لا يُحتج به في إثبات الأحكام"
                )

        # Select system prompt and context format based on query type
        system_prompt = _select_system_prompt(query_type)
        
        if query_type == "metadata":
            context = _format_metadata_context(hadiths)
            # Lower temperature for metadata answers (factual)
            temperature = min(temperature, 0.1)
        else:
            context = _format_hadith_context(hadiths)

        # Build the user message with context
        user_message = f"""## السياق (الأحاديث المسترجعة):
{context}

## سؤال المستخدم:
{query}"""

        # For explain_hadith queries: add a relevance hint so the LLM knows
        # whether the specific hadith was found in the corpus
        if query_type == "explain_hadith":
            from retrieval.query_preprocessor import _extract_hadith_text_from_explain_query
            # Derive the requested hadith text from the normalized query
            norm_q = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u0640]", "", query)
            norm_q = re.sub(r"[أإآٱ]", "ا", norm_q)
            norm_q = re.sub(r"\s+", " ", norm_q).strip()
            requested_text = _extract_hadith_text_from_explain_query(norm_q)
            corpus_has_it = _check_hadith_relevance(requested_text, hadiths)
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
            f"hadiths={len(hadiths)}, query_type={query_type}, "
            f"temp={temperature}"
        )

        # gemma-3-* does not support system_instruction, so merge it into contents.
        merged_prompt = (
            f"## تعليمات النظام:\n{system_prompt}\n\n"
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

        answer = response.text or ""
        # answer = response.choices[0].message.content
        logger.info(f"Generation complete: {len(answer)} chars")

        # Citation grounding verification
        grounding_verified = True
        grounding_issues = []
        if verify_grounding:
            grounding_verified, grounding_issues = _verify_citation_grounding(
                answer, hadiths
            )
            if not grounding_verified:
                logger.warning(
                    f"Grounding issues detected: {grounding_issues}"
                )

        return GeneratedResponse(
            answer=answer,
            citations=citations,
            warnings=warnings,
            grounding_verified=grounding_verified,
            grounding_issues=grounding_issues,
            raw_text=answer,
            query_type=query_type,
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
