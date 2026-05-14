"""
Arabic answer generation via Google Gemini, wrapped as a LlamaIndex CustomLLM.
"""

import concurrent.futures
import re
import time
from typing import Any, Generator

from google import genai
from google.genai import types
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)

from core.config import GEMINI_MODEL, GOOGLE_API_KEY, MAX_OUTPUT_TOKENS
from core.retriever import Result


# ---------------------------------------------------------------------------
# GeminiLLM — LlamaIndex CustomLLM wrapping google-genai SDK
# ---------------------------------------------------------------------------

class GeminiLLM(CustomLLM):
    """Google Gemini as a LlamaIndex CustomLLM (registered via Settings.llm)."""

    context_window: int = 32_768
    num_output: int = MAX_OUTPUT_TOKENS
    model_name: str = GEMINI_MODEL

    _client: Any = PrivateAttr()
    _gen_config: Any = PrivateAttr()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = genai.Client(api_key=GOOGLE_API_KEY)
        self._gen_config = types.GenerateContentConfig(
            max_output_tokens=self.num_output,
            temperature=0.2,
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(
                self._client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=self._gen_config,
            )
            response = future.result(timeout=45)
        return CompletionResponse(text=_extract_text(response))

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        def _gen() -> Generator[CompletionResponse, None, None]:
            full = ""
            deadline = time.monotonic() + 60
            for chunk in self._client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=self._gen_config,
            ):
                if time.monotonic() > deadline:
                    raise TimeoutError("Gemini streaming timed out after 60 seconds")
                delta = getattr(chunk, "text", "") or ""
                if delta:
                    full += delta
                    yield CompletionResponse(text=full, delta=delta)
        return _gen()


# Module-level LLM instance used by generate_answer / generate_answer_stream
_llm = GeminiLLM()


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
# Message builder
# ---------------------------------------------------------------------------

def _build_user_message(query: str, results: list[Result]) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"السؤال: {query}\n\n"
        f"المقاطع المسترجعة من الموسوعة الفقهية الكويتية:\n\n"
        f"{_build_context(results)}\n\n"
        f"أجب على السؤال بناءً على هذه المقاطع فقط، مع الاستشهاد بـ [رقم المقطع] لكل حكم."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(response) -> str:
    """Safely extract text — handles empty/blocked responses."""
    try:
        return response.text or ""
    except Exception:
        pass
    try:
        return response.candidates[0].content.parts[0].text or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public generation API (callers unchanged after refactor)
# ---------------------------------------------------------------------------

_NO_RESULTS_AR = "لم يُعثر على نتائج ذات صلة في الموسوعة الفقهية الكويتية للإجابة على هذا السؤال."


def generate_answer(query: str, results: list[Result]) -> str:
    """Send retrieved context + query to Gemini via LlamaIndex CustomLLM."""
    if not results:
        return _NO_RESULTS_AR
    prompt = _build_user_message(query, results)
    answer = _llm.complete(prompt).text
    return _validate_citations(answer, len(results))


def generate_answer_stream(query: str, results: list[Result]):
    """Yield answer tokens using Gemini streaming via LlamaIndex CustomLLM."""
    if not results:
        yield _NO_RESULTS_AR
        return
    prompt = _build_user_message(query, results)
    full = ""
    for cr in _llm.stream_complete(prompt):
        delta = cr.delta or ""
        if delta:
            full += delta
            yield delta

    validated = _validate_citations(full, len(results))
    if len(validated) > len(full):
        yield validated[len(full):]
