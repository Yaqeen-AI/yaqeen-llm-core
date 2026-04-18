"""
Arabic answer generation via LM Studio (Gemma 4 — OpenAI-compatible API).
LM Studio runs locally on http://localhost:1234/v1
"""

import re

from openai import OpenAI

from core.config import LM_STUDIO_API_KEY, LM_STUDIO_BASE_URL, LM_STUDIO_MODEL
from core.retriever import Result

# Module-level client — instantiated once, not per request
_client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
أنت عالم متخصص في الفقه الإسلامي، متمكن من المذاهب الأربعة: الحنفي والمالكي والشافعي والحنبلي.
مهمتك الإجابة على الأسئلة الفقهية استناداً حصراً إلى المقاطع المسترجعة من الموسوعة الفقهية الكويتية.

قواعد الإجابة:
١. أجب بالعربية الفصحى دائماً.
٢. استند إلى نصوص المقاطع المقدمة فقط — لا تضف معلومات من خارجها.
٣. إن وُجد خلاف بين المذاهب، اعرضه بوضوح وأشر إلى رأي كل مذهب صراحةً.
٤. كل حكم تذكره يجب أن يكون مرفقاً بالمصدر [رقم المقطع] مباشرةً في النص — لا تذكر حكماً بلا مصدر.
٥. اختم إجابتك بقائمة المصادر بتنسيق: [ن] م.ف.ك — جX، صY.
٦. إذا لم تكفِ المقاطع للإجابة، صرّح بذلك صراحةً بدلاً من الاجتهاد خارج النص.
٧. كن دقيقاً في النقل وأمناً في العرض.\
"""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(results: list[Result]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        mazhab_line = f"المذاهب: {r.mazhab_tag()}\n" if r.mazhabs else ""
        parts.append(
            f"[{i}] المصدر: {r.short_ref()}\n"
            f"{mazhab_line}"
            f"النص: {r.chunk_text}"
        )
    return ("\n\n" + "─" * 50 + "\n\n").join(parts)


# ---------------------------------------------------------------------------
# Citation validator
# ---------------------------------------------------------------------------

def _validate_citations(answer: str, num_chunks: int) -> str:
    """
    Check that every [n] reference in the answer maps to a real chunk.
    Appends a warning for any out-of-range citations so the user can verify.
    """
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    invalid = sorted(n for n in cited if n < 1 or n > num_chunks)
    if invalid:
        refs = "، ".join(f"[{n}]" for n in invalid)
        answer += (
            f"\n\n---\n⚠️ **تنبيه:** المراجع {refs} لا تقابل مقاطع مسترجعة "
            f"(المتاح: [1]–[{num_chunks}]). يُرجى التحقق."
        )
    return answer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(query: str, results: list[Result]) -> str:
    """Send retrieved context + query to local Gemma 4 via LM Studio."""
    user_message = (
        f"السؤال: {query}\n\n"
        f"المقاطع المسترجعة من الموسوعة الفقهية الكويتية:\n\n"
        f"{_build_context(results)}\n\n"
        f"أجب على السؤال بناءً على هذه المقاطع فقط، مع الاستشهاد بـ [رقم المقطع] لكل حكم."
    )

    response = _client.chat.completions.create(
        model=LM_STUDIO_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=2048,
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    return _validate_citations(answer, len(results))
