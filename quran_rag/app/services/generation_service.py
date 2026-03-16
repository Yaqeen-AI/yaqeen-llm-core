from __future__ import annotations

import os
import re
from typing import Optional

from loguru import logger

from app.core.config import get_settings
from app.models.schemas import (
    AnswerCitation,
    AnswerRequest,
    AnswerResponse,
    RetrievalRequest,
    RetrievalResult,
)
from app.preprocessing.arabic_normalizer import ArabicTextNormalizer
from app.services.hybrid_retrieval import HybridRetrievalPipeline
from app.services.vector_store_factory import VectorStoreProtocol


class GenerationService:
    """
    Retrieval-grounded answer generation on top of the Quran retriever.

    The class deliberately keeps generation thin:
      1. Retrieve ayah-level evidence through the hybrid pipeline.
      2. Expand each top hit with a small local ayah window for coherence.
      3. Build a strict prompt that forbids unsupported claims.
      4. Call the configured Gemma 3 provider and return citations.

    This design keeps the retrieval system as the source of truth and makes the
    generation layer swappable if the serving provider changes later.
    """

    def __init__(
        self,
        retrieval_pipeline: HybridRetrievalPipeline,
        vector_store: VectorStoreProtocol,
    ):
        self._pipeline = retrieval_pipeline
        self._vector_store = vector_store
        self._settings = get_settings()
        self._normalizer = ArabicTextNormalizer()
        self._client = None
        self._types = None

        provider = self._settings.generation_provider.lower()
        api_key = (
            self._settings.google_api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        if provider != "google_genai":
            raise ValueError(
                f"Unsupported GENERATION_PROVIDER='{self._settings.generation_provider}'. "
                "Only 'google_genai' is implemented in this repo."
            )

        if not api_key:
            logger.warning(
                "Generation service not initialized because GOOGLE_API_KEY/GEMINI_API_KEY is missing."
            )
            return

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            logger.warning("google-genai is not installed: {}", exc)
            return

        self._client = genai.Client(api_key=api_key)
        self._types = types
        logger.info(
            "Generation service ready with model '{}'",
            self._settings.generation_model_name,
        )

    @property
    def is_available(self) -> bool:
        return self._client is not None

    @property
    def model_name(self) -> str:
        return self._settings.generation_model_name

    def answer(self, request: AnswerRequest) -> AnswerResponse:
        """
        Generate an answer grounded in retrieved Quran evidence.

        Raises:
            RuntimeError: if the external generation provider is unavailable.
        """
        if not self.is_available:
            raise RuntimeError(
                "Generation service is unavailable. Install `google-genai` and set "
                "GOOGLE_API_KEY (or GEMINI_API_KEY)."
            )

        retrieval_request = RetrievalRequest(
            query=request.query,
            language=request.language,
            content_type_filter=request.content_type_filter,
            edition_identifier_filter=request.edition_identifier_filter,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            use_reranking=request.use_reranking,
            surah_filter=request.surah_filter,
            juz_filter=request.juz_filter,
        )
        retrieval = self._pipeline.retrieve(retrieval_request)

        if not retrieval.results:
            return AnswerResponse(
                query=request.query,
                answer=self._build_no_evidence_answer(request.query),
                model_name=self._settings.generation_model_name,
                citations=[],
                retrieval=retrieval,
                prompt_preview="",
            )

        if self._should_return_grounded_verse_list(
            query=request.query,
            retrieved_results=retrieval.results,
        ):
            citations = [
                AnswerCitation(
                    chunk_id=result.chunk_id,
                    surah_number=result.metadata.surah_number,
                    ayah_number_in_surah=result.metadata.ayah_number_in_surah,
                    ayah_ref=self._format_ayah_ref(result.metadata),
                    surah_name_english=result.metadata.surah_name_english,
                    content_type=result.metadata.content_type,
                    edition_identifier=result.metadata.edition_identifier,
                    edition_name=result.metadata.edition_name,
                    score=result.score,
                    text=result.text,
                )
                for result in retrieval.results
            ]
            return AnswerResponse(
                query=request.query,
                answer=self._build_grounded_verse_list(retrieval.results),
                model_name=self._settings.generation_model_name,
                citations=citations,
                retrieval=retrieval,
                prompt_preview="",
            )

        context_window = request.context_window or self._settings.retrieval_context_window
        prompt = self._build_prompt(
            query=request.query,
            retrieved_results=retrieval.results,
            context_window=context_window,
        )
        answer_text = self._generate(prompt)

        citations = [
            AnswerCitation(
                chunk_id=result.chunk_id,
                surah_number=result.metadata.surah_number,
                ayah_number_in_surah=result.metadata.ayah_number_in_surah,
                ayah_ref=self._format_ayah_ref(result.metadata),
                surah_name_english=result.metadata.surah_name_english,
                content_type=result.metadata.content_type,
                edition_identifier=result.metadata.edition_identifier,
                edition_name=result.metadata.edition_name,
                score=result.score,
                text=result.text,
            )
            for result in retrieval.results
        ]

        return AnswerResponse(
            query=request.query,
            answer=answer_text,
            model_name=self._settings.generation_model_name,
            citations=citations,
            retrieval=retrieval,
            prompt_preview=prompt[:4000],
        )

    def _generate(self, prompt: str) -> str:
        config = self._types.GenerateContentConfig(
            temperature=self._settings.generation_temperature,
            top_p=self._settings.generation_top_p,
            max_output_tokens=self._settings.generation_max_output_tokens,
        )
        response = self._client.models.generate_content(
            model=self._settings.generation_model_name,
            contents=prompt,
            config=config,
        )
        text = getattr(response, "text", "") or ""
        if not text.strip():
            raise RuntimeError("Generation provider returned an empty response.")
        return text.strip()

    def _build_no_evidence_answer(self, query: str) -> str:
        answer_language = self._normalizer.detect_language(query)
        if answer_language == "ar":
            return (
                "لم يتم العثور على آيات مطابقة في النتائج المسترجعة، لذلك لن أقدّم إجابة غير "
                "مدعومة. جرّب إزالة بعض الفلاتر أو إعادة صياغة السؤال."
            )
        return (
            "No supporting Quran evidence was retrieved, so I will not generate an "
            "unsupported answer. Try removing restrictive filters or rephrasing the query."
        )

    def _should_return_grounded_verse_list(
        self,
        *,
        query: str,
        retrieved_results: list[RetrievalResult],
    ) -> bool:
        if not retrieved_results:
            return False

        if any(result.metadata.content_type.value != "quran_ayah" for result in retrieved_results):
            return False

        detected_language = self._normalizer.detect_language(query)
        if detected_language == "ar":
            return bool(re.search(r"\b(اية|اية|ايات|آيات)\b", query))

        return bool(re.search(r"\b(verse|verses|ayah|ayat)\b", query.lower()))

    def _build_grounded_verse_list(
        self,
        retrieved_results: list[RetrievalResult],
    ) -> str:
        lines = ["الآيات الأقرب إلى سؤالك بحسب النتائج المسترجعة:"]
        for result in retrieved_results:
            ayah_ref = self._format_ayah_ref(result.metadata) or "unknown"
            lines.append(f"- [{ayah_ref}] {result.text}")
        lines.append("الجواب أعلاه مقتصر على الآيات المسترجعة فقط.")
        return "\n".join(lines)

    def _build_prompt(
        self,
        *,
        query: str,
        retrieved_results: list[RetrievalResult],
        context_window: int,
    ) -> str:
        answer_language = self._normalizer.detect_language(query)
        context_blocks = self._build_context_blocks(
            retrieved_results=retrieved_results,
            context_window=context_window,
        )
        joined_context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."
        content_types = {
            result.metadata.content_type.value for result in retrieved_results
        }
        includes_tafsir = "tafsir" in content_types

        if answer_language == "ar" and includes_tafsir:
            return (
                "أنت مساعد متخصص في الإجابة المعتمدة على النصوص المسترجعة من القرآن والتفسير فقط.\n"
                "لا تضف ادعاءات غير مدعومة، وإذا كان الدليل غير كافٍ فقل ذلك بوضوح.\n"
                "استشهد بالقرآن بصيغة [السورة:الآية]، وبالتفسير بصيغة [السورة:الآية | المصدر].\n\n"
                f"السؤال:\n{query}\n\n"
                f"السياق المسترجع:\n{joined_context}\n\n"
                "التعليمات:\n"
                "1. أجب بإيجاز ووضوح.\n"
                "2. ميّز بين النص القرآني وبين الشرح التفسيري.\n"
                "3. عند الاعتماد على تفسير، اذكر اسم المصدر في الاستشهاد.\n"
                "4. إذا احتاج السؤال إلى معلومات خارج السياق فاذكر أن السياق غير كافٍ.\n"
            )

        if answer_language == "ar":
            return (
                "أنت مساعد متخصص في الإجابة المعتمدة على آيات القرآن فقط.\n"
                "التزم بالنصوص المسترجعة ولا تضف ادعاءات غير مدعومة.\n"
                "إذا كان الدليل غير كافٍ فقل ذلك بوضوح.\n"
                "اذكر الاستشهادات بصيغة [السورة:الآية].\n\n"
                f"السؤال:\n{query}\n\n"
                f"السياق المسترجع:\n{joined_context}\n\n"
                "التعليمات:\n"
                "1. أجب بإيجاز ووضوح.\n"
                "2. فرّق بين النص القرآني نفسه وبين أي شرح تستنتجه.\n"
                "3. لا تنسب معنى غير موجود في الآيات.\n"
                "4. إذا احتاج السؤال إلى تفسير موسع غير موجود في السياق فاذكر ذلك.\n"
            )

        if includes_tafsir:
            return (
                "You are a retrieval-grounded Quran and tafsir answering assistant.\n"
                "Only use the retrieved passages below.\n"
                "Do not fabricate claims that are not supported by the cited sources.\n"
                "If the evidence is insufficient, say so explicitly.\n"
                "Cite Quran evidence as [surah:ayah] and tafsir evidence as [surah:ayah | source].\n\n"
                f"Question:\n{query}\n\n"
                f"Retrieved context:\n{joined_context}\n\n"
                "Instructions:\n"
                "1. Answer clearly and directly.\n"
                "2. Distinguish between Quran text and tafsir explanation.\n"
                "3. Name the tafsir source when you rely on it.\n"
            )

        return (
            "You are a Quran-grounded answering assistant.\n"
            "Only use the retrieved Quran passages below.\n"
            "Do not fabricate interpretation that is not supported by the cited ayahs.\n"
            "If the retrieved evidence is insufficient, say so explicitly.\n"
            "Cite evidence in the format [surah:ayah].\n\n"
            f"Question:\n{query}\n\n"
            f"Retrieved context:\n{joined_context}\n\n"
            "Instructions:\n"
            "1. Answer clearly and directly.\n"
            "2. Distinguish between direct Quran evidence and your short explanation.\n"
            "3. Keep the answer grounded in the provided context.\n"
        )

    def _build_context_blocks(
        self,
        *,
        retrieved_results: list[RetrievalResult],
        context_window: int,
    ) -> list[str]:
        blocks = []
        seen_keys = set()

        for rank, result in enumerate(retrieved_results, start=1):
            metadata = result.metadata
            if metadata.surah_number is None or metadata.ayah_number_in_surah is None:
                continue

            key = (
                metadata.surah_number,
                metadata.ayah_number_in_surah,
                metadata.language.value,
                metadata.content_type.value,
                metadata.edition_identifier,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)

            window = self._vector_store.get_adjacent_chunks(
                surah_number=metadata.surah_number,
                ayah_number_in_surah=metadata.ayah_number_in_surah,
                language=metadata.language.value,
                content_type=metadata.content_type.value,
                edition_identifier=metadata.edition_identifier,
                window_size=context_window,
            )
            block_lines = [
                f"[Rank {rank}] {self._format_source_label(metadata)} | "
                f"Surah {metadata.surah_number} "
                f"({metadata.surah_name_english or metadata.surah_name_arabic or 'Unknown'})"
            ]
            for item in window or [result]:
                item_meta = item.metadata
                block_lines.append(
                    f"[{self._format_inline_citation(item_meta)}] {item.text}"
                )

            blocks.append("\n".join(block_lines))

        return blocks

    @staticmethod
    def _format_ayah_ref(metadata) -> Optional[str]:
        if metadata.ayah_ref:
            return metadata.ayah_ref
        if metadata.surah_number is None or metadata.ayah_number_in_surah is None:
            return None
        return f"{metadata.surah_number}:{metadata.ayah_number_in_surah}"

    def _format_inline_citation(self, metadata) -> str:
        ayah_ref = self._format_ayah_ref(metadata) or "unknown"
        if metadata.content_type.value == "tafsir":
            source_name = metadata.edition_name or metadata.edition_identifier or "tafsir"
            return f"{ayah_ref} | {source_name}"
        if (
            metadata.edition_identifier
            and metadata.edition_identifier != self._settings.quran_default_edition
        ):
            return f"{ayah_ref} | {metadata.edition_identifier}"
        return ayah_ref

    def _format_source_label(self, metadata) -> str:
        if metadata.content_type.value == "tafsir":
            return f"Tafsir::{metadata.edition_name or metadata.edition_identifier or 'Unknown'}"
        return f"Quran::{metadata.edition_name or metadata.edition_identifier or self._settings.quran_default_edition}"
