# ============================================================
# YaqeenAI — Full RAG Pipeline Orchestrator [LOCAL]
# ============================================================
# Wires together all pipeline stages:
#   Query → Preprocess → Hybrid Retrieve → Rerank → Generate → Answer
#
# Supports two modes:
#   1. Hybrid mode: Dense (Jina/Qdrant) + Sparse (TF-IDF/BM25) + RRF fusion
#   2. Dense-only mode: Falls back if TF-IDF index not available
#
# Features:
#   - LRU caching for embeddings and responses
#   - Canonical group deduplication
#   - Citation grounding verification
#   - Structured JSON response
#   - Query-type-aware routing (greeting, out-of-scope, metadata, general)
#   - Metadata-focused generation for metadata queries

import logging
import copy
import re
import time
from collections import OrderedDict
from typing import Optional, Literal
from dataclasses import dataclass, field

from pipeline.config import settings, resolve_grade_label
from pipeline.embed_query import JinaQueryEmbedder
from pipeline.retrieve import HadithRetriever, RetrievedHadith
from pipeline.rerank import HadithReranker
from pipeline.generate import HadithGenerator, GeneratedResponse
from retrieval.hybrid_retriever import HybridRetriever, HybridResult
from retrieval.query_preprocessor import preprocess_query, QueryType

logger = logging.getLogger(__name__)

_PRIMARY_SOURCE_BOOKS = ("صحيح البخاري", "صحيح مسلم")
_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]+")
_TATWEEL = re.compile(r"\u0640+")
_WHITESPACE = re.compile(r"\s+")
_ALEF_VARIANTS = re.compile(r"[أإآٱ]")
_EXACT_LOOKUP_RERANK_MIN_OVERLAP = 0.45
_TIMING_FLAG_KEYS = {
    "generate_cache_hit",
    "generation_fast_path",
    "rerank_skipped_exact_lookup",
    "generation_thinking_config_retry",
}
_TIMING_METADATA_KEYS = {
    "generation_prompt_chars",
    "generation_max_output_tokens",
    "generation_thinking_budget",
}


def _normalize_source_name(source: str) -> str:
    normalized = _TASHKEEL.sub("", str(source or "").strip().lower())
    normalized = _TATWEEL.sub("", normalized)
    normalized = _ALEF_VARIANTS.sub("ا", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def _source_priority(masdar: str) -> int:
    normalized = _normalize_source_name(masdar)
    if "صحيح البخاري" in normalized or normalized == "البخاري" or "bukhari" in normalized:
        return 0
    if "صحيح مسلم" in normalized or normalized == "مسلم" or "muslim" in normalized:
        return 1
    return 2


def _is_primary_source(hadith: RetrievedHadith) -> bool:
    return _source_priority(hadith.masdar) < 2


def _promote_primary_sources(hadiths: list[RetrievedHadith]) -> list[RetrievedHadith]:
    indexed = list(enumerate(hadiths))
    indexed.sort(key=lambda item: (_source_priority(item[1].masdar), item[0]))
    return [hadith for _, hadith in indexed]


def _merge_with_primary_candidates(
    base_hadiths: list[RetrievedHadith],
    primary_hadiths: list[RetrievedHadith],
    limit: int,
) -> list[RetrievedHadith]:
    if not primary_hadiths:
        return base_hadiths[:limit]

    merged: list[RetrievedHadith] = []
    seen_ids: set[str] = set()

    for hadith in _promote_primary_sources(primary_hadiths) + base_hadiths:
        if hadith.id in seen_ids:
            continue
        merged.append(hadith)
        seen_ids.add(hadith.id)
        if len(merged) >= limit:
            break

    return merged


def _ensure_primary_in_reranked(
    reranked_hadiths: list[RetrievedHadith],
    retrieval_candidates: list[RetrievedHadith],
) -> list[RetrievedHadith]:
    if not reranked_hadiths:
        return reranked_hadiths

    if any(_is_primary_source(hadith) for hadith in reranked_hadiths):
        return _promote_primary_sources(reranked_hadiths)

    fallback_primary = next((hadith for hadith in retrieval_candidates if _is_primary_source(hadith)), None)
    if fallback_primary is None:
        return reranked_hadiths

    forced = [fallback_primary]
    for hadith in reranked_hadiths:
        if hadith.id == fallback_primary.id:
            continue
        forced.append(hadith)
        if len(forced) >= len(reranked_hadiths):
            break
    return forced


def _tokenize_lookup_text(text: str) -> set[str]:
    normalized = _normalize_source_name(text).replace("ة", "ه")
    tokens = re.findall(r"[\u0600-\u06FF]{3,}", normalized)
    stopwords = {
        "حديث", "اشرح", "شرح", "فسر", "معنى", "قال", "عن", "على", "الى",
        "إلى", "هذا", "هذه", "الحديث", "النبي", "رسول", "الله", "صلى",
        "عليه", "وسلم", "الراوي", "الدرجه", "المتن",
    }
    return {token for token in tokens if token not in stopwords}


def _lookup_overlap(query_text: str, hadith: RetrievedHadith) -> float:
    query_tokens = _tokenize_lookup_text(query_text)
    hadith_tokens = _tokenize_lookup_text(hadith.text_ar)
    if not query_tokens or not hadith_tokens:
        return 0.0
    return len(query_tokens & hadith_tokens) / len(query_tokens)


def _can_skip_rerank_for_exact_lookup(
    processed_query_type: QueryType,
    lookup_text: str,
    retrieved_hadiths: list[RetrievedHadith],
) -> bool:
    if processed_query_type not in {QueryType.EXPLAIN_HADITH, QueryType.HADITH_LOOKUP}:
        return False
    if not retrieved_hadiths:
        return False
    best_overlap = max(_lookup_overlap(lookup_text, hadith) for hadith in retrieved_hadiths[:5])
    return best_overlap >= _EXACT_LOOKUP_RERANK_MIN_OVERLAP


def _format_seconds(duration: object) -> str:
    try:
        value = float(duration)
    except (TypeError, ValueError):
        return str(duration)
    if value <= 0 or value < 0.001:
        return "<0.001s"
    if value < 1:
        return f"{value:.3f}s"
    return f"{value:.2f}s"


def _format_timing_entry(key: str, value: object) -> str:
    if key in _TIMING_FLAG_KEYS:
        return "yes" if bool(value) else "no"
    if key == "generation_prompt_chars":
        return f"{int(value):,} chars"
    if key == "generation_max_output_tokens":
        return f"{int(value):,} tokens"
    if key == "generation_thinking_budget":
        return "disabled" if int(value) < 0 else str(int(value))
    if key in _TIMING_METADATA_KEYS:
        return str(value)
    return _format_seconds(value)


@dataclass
class RAGResponse:
    """Complete response from the Hadith RAG pipeline."""

    query: str
    answer: str
    retrieved_hadiths: list[RetrievedHadith] = field(default_factory=list)
    reranked_hadiths: list[RetrievedHadith] = field(default_factory=list)
    generation: Optional[GeneratedResponse] = None
    timing: dict = field(default_factory=dict)
    query_type: str = "general"  # For API consumers to know how the query was handled
    answer_intent: str = ""
    evidence_sufficient: bool = False
    authenticity_of_evidence: str = "insufficient"
    relevance_to_question: str = "weak"
    final_sufficiency: str = "insufficient"

    def __str__(self) -> str:
        header = f"📖 سؤال: {self.query}\n{'=' * 60}\n"
        citations = "\n\n📚 المصادر المستخدمة:\n"
        for i, h in enumerate(self.reranked_hadiths, 1):
            grade_ar = resolve_grade_label(h.grade, h.grade_ar, h.ruling)
            citations += (
                f"  [{i}] {h.masdar} — {h.safha_raqam} "
                f"(الراوي: {h.rawi}، الدرجة: {grade_ar})\n"
            )

        timing_str = "\n⏱️ التوقيت:\n"
        for stage, duration in self.timing.items():
            timing_str += f"  {stage}: {_format_timing_entry(stage, duration)}\n"

        # Grounding status
        grounding = ""
        if self.generation:
            status = "✅" if self.generation.grounding_verified else "⚠️"
            grounding = f"\n{status} التوثيق: {'موثق' if self.generation.grounding_verified else 'يحتاج مراجعة'}\n"
            if self.generation.warnings:
                grounding += "\n".join(self.generation.warnings) + "\n"

        return header + self.answer + citations + grounding + timing_str

    def to_dict(self) -> dict:
        """Serialize to dict for API responses."""
        return {
            "query": self.query,
            "answer": self.answer,
            "query_type": self.query_type,
            "answer_intent": self.answer_intent,
            "evidence_sufficient": self.evidence_sufficient,
            "authenticity_of_evidence": self.authenticity_of_evidence,
            "relevance_to_question": self.relevance_to_question,
            "final_sufficiency": self.final_sufficiency,
            "citations": self.generation.to_dict()["citations"] if self.generation else [],
            "ignored_narrations": self.generation.to_dict()["ignored_narrations"] if self.generation else [],
            "warnings": self.generation.warnings if self.generation else [],
            "grounding_verified": self.generation.grounding_verified if self.generation else False,
            "hadiths": [
                {
                    "id": h.id,
                    "text_ar": h.text_ar,
                    "grade": h.grade,
                    "rawi": h.rawi,
                    "muhaddith": h.muhaddith,
                    "masdar": h.masdar,
                    "safha_raqam": h.safha_raqam,
                    "category": h.category,
                }
                for h in self.reranked_hadiths
            ],
            "timing": self.timing,
        }


class HadithRAGPipeline:
    """
    Full Hadith RAG pipeline orchestrator.

    Stages:
    0. [LOCAL] Query-type routing (greeting/out-of-scope → early exit)
    1. [LOCAL] Query preprocessing (normalize, classify, expand, transliterate)
    2. [LOCAL] Hybrid retrieval: Dense (Jina -> Qdrant) + Sparse (TF-IDF/BM25)
    3. [LOCAL] RRF fusion + canonical group deduplication
    4. [LOCAL→API] Jina reranking (20 → 5)
    5. [LOCAL→API] Query-type-aware generation with citation grounding

    All stages run on CPU. GPU is only needed for the Colab
    indexing phase (already completed).
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[HadithReranker] = None,
        generator: Optional[HadithGenerator] = None,
        cache_size: int = 1000,
    ):
        logger.info("Initializing Hadith RAG Pipeline...")

        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            embedding_cache_size=cache_size,
            embedding_cache_ttl_seconds=settings.EMBEDDING_CACHE_TTL_SECONDS,
        )
        self.reranker = reranker or HadithReranker()
        self.generator = generator or HadithGenerator()
        self._embedding_cache = self.hybrid_retriever.embedding_cache
        self._response_cache: OrderedDict[tuple, GeneratedResponse] = OrderedDict()
        self._response_cache_size = max(0, settings.RESPONSE_CACHE_SIZE)
        self._retrieval_cache: OrderedDict[tuple, tuple[float, HybridResult]] = OrderedDict()
        self._retrieval_cache_size = max(0, settings.RETRIEVAL_CACHE_SIZE)
        self._retrieval_cache_ttl_seconds = max(1, settings.RETRIEVAL_CACHE_TTL_SECONDS)

        logger.info("✅ Hadith RAG Pipeline initialized successfully")

    def _build_generation_cache_key(
        self,
        query: str,
        hadiths: list[RetrievedHadith],
        query_type: str,
        metadata_fields: list[str] | None,
        excluded_masdar: list[str] | None,
        temperature: float,
    ) -> tuple:
        return (
            query.strip(),
            query_type,
            tuple(metadata_fields or []),
            tuple(excluded_masdar or []),
            round(float(temperature), 3),
            settings.GEMINI_MODEL,
            tuple((h.id, h.grade, h.ruling) for h in hadiths),
        )

    def _get_cached_generation(self, key: tuple) -> GeneratedResponse | None:
        if self._response_cache_size <= 0:
            return None
        generation = self._response_cache.get(key)
        if generation is None:
            return None
        self._response_cache.move_to_end(key)
        return generation

    def _cache_generation(self, key: tuple, generation: GeneratedResponse) -> None:
        if self._response_cache_size <= 0:
            return
        self._response_cache[key] = generation
        self._response_cache.move_to_end(key)
        while len(self._response_cache) > self._response_cache_size:
            self._response_cache.popitem(last=False)

    def _build_retrieval_cache_key(
        self,
        processed,
        grade_filter: Optional[str | list[str]],
        masdar_filter: Optional[str | list[str]],
        retrieval_top_k: int,
        retrieval_mode: str,
        enable_dedup: bool = True,
    ) -> tuple:
        if isinstance(grade_filter, list):
            grade_key = tuple(grade_filter)
        else:
            grade_key = grade_filter

        if isinstance(masdar_filter, list):
            masdar_key = tuple(masdar_filter)
        else:
            masdar_key = masdar_filter

        return (
            processed.query_type.value,
            str(getattr(processed, "normalized", "")).strip(),
            str(getattr(processed, "dense_query", "")).strip(),
            str(getattr(processed, "sparse_query", "")).strip(),
            tuple(getattr(processed, "multi_queries", []) or []),
            grade_key,
            masdar_key,
            int(retrieval_top_k),
            retrieval_mode,
            bool(enable_dedup),
            settings.DENSE_TOP_K,
            settings.SPARSE_TOP_K,
            settings.RRF_K,
        )

    def _get_cached_retrieval(self, key: tuple) -> HybridResult | None:
        if self._retrieval_cache_size <= 0:
            return None

        cached = self._retrieval_cache.get(key)
        if cached is None:
            return None

        expires_at, result = cached
        if expires_at <= time.monotonic():
            del self._retrieval_cache[key]
            return None

        self._retrieval_cache.move_to_end(key)
        result_copy = copy.deepcopy(result)
        result_copy.timing = {
            **result_copy.timing,
            "cache_hit": 1.0,
            "total": 0.0,
        }
        return result_copy

    def _cache_retrieval(self, key: tuple, result: HybridResult) -> None:
        if self._retrieval_cache_size <= 0:
            return

        self._retrieval_cache[key] = (
            time.monotonic() + self._retrieval_cache_ttl_seconds,
            copy.deepcopy(result),
        )
        self._retrieval_cache.move_to_end(key)
        while len(self._retrieval_cache) > self._retrieval_cache_size:
            self._retrieval_cache.popitem(last=False)

    def _retrieve_with_cache(
        self,
        *,
        user_query: str,
        processed,
        retrieval_top_k: int,
        grade_filter: Optional[str | list[str]],
        masdar_filter: Optional[str | list[str]],
        retrieval_mode: str,
        query_embedding: Optional[list[float]] = None,
    ) -> HybridResult:
        cache_key = self._build_retrieval_cache_key(
            processed=processed,
            grade_filter=grade_filter,
            masdar_filter=masdar_filter,
            retrieval_top_k=retrieval_top_k,
            retrieval_mode=retrieval_mode,
        )

        cached = self._get_cached_retrieval(cache_key)
        if cached is not None:
            return cached

        result = self.hybrid_retriever.retrieve(
            query=user_query,
            fused_top_k=retrieval_top_k,
            grade_filter=grade_filter,
            masdar_filter=masdar_filter,
            retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
            processed_query=processed,
            query_embedding=query_embedding,
        )
        result.timing["cache_hit"] = 0.0
        self._cache_retrieval(cache_key, result)
        return result

    def query(
        self,
        user_query: str,
        grade_filter: Optional[str | list[str]] = None,
        masdar_filter: Optional[str] = None,
        retrieval_top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        retrieval_mode: Literal["tfidf", "bm25", "both"] = "both",
        temperature: float = 0.3,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline.

        Handles query-type routing:
        - GREETING / OUT_OF_SCOPE → immediate response (no retrieval/generation)
        - METADATA → retrieval + metadata-focused generation
        - NARRATOR → retrieval + narrator-focused generation
        - Others → standard retrieval + general generation

        Args:
            user_query: The user's question (Arabic or English).
            grade_filter: Optional grade filter(s): 'sahih', 'hasan', 'daif', 'mawdu', 'unknown'.
            masdar_filter: Optional source book filter (Arabic name).
            retrieval_top_k: Override fused result count (default: 20).
            rerank_top_k: Override number of reranked results (default: 5).
            retrieval_mode: Sparse retrieval mode (tfidf, bm25, both).
            temperature: LLM temperature (lower = more conservative, default: 0.3).

        Returns:
            RAGResponse with answer, citations, grounding, and timing.
        """
        timing = {}
        total_start = time.time()

        # ──────────────────────────────────────────────
        # Stage 0: Quick preprocess to check for early-exit query types
        # ──────────────────────────────────────────────
        t0 = time.time()
        processed = preprocess_query(user_query)
        timing["preprocess"] = time.time() - t0

        # ── Early exit: Greeting ──
        if processed.query_type == QueryType.GREETING and processed.skip_retrieval:
            timing["total"] = time.time() - total_start
            logger.info(f"Greeting detected, returning direct response")
            return RAGResponse(
                query=user_query,
                answer=processed.direct_response,
                timing=timing,
                query_type="greeting",
                answer_intent="",
                evidence_sufficient=False,
            )

        # ── Early exit: Out of scope ──
        if processed.query_type == QueryType.OUT_OF_SCOPE and processed.skip_retrieval:
            timing["total"] = time.time() - total_start
            logger.info(f"Out-of-scope query detected, returning direct response")
            return RAGResponse(
                query=user_query,
                answer=processed.direct_response,
                timing=timing,
                query_type="out_of_scope",
                answer_intent="",
                evidence_sufficient=False,
            )

        # ── Early exit: Dataset statistics ──
        if processed.query_type == QueryType.DATASET_STATS and processed.skip_retrieval:
            timing["total"] = time.time() - total_start
            logger.info(f"Dataset stats query detected, returning pre-computed stats")
            return RAGResponse(
                query=user_query,
                answer=processed.direct_response,
                timing=timing,
                query_type="dataset_stats",
                answer_intent="",
                evidence_sufficient=False,
            )

        # ──────────────────────────────────────────────
        # Stages 1-3: Hybrid Retrieval (dense + sparse + RRF + dedup)
        # The hybrid retriever calls preprocess_query internally too,
        # but since preprocessing is cheap (<1ms), this is fine.
        # ──────────────────────────────────────────────
        t0 = time.time()
        logger.info(f"Stages 1-3: Hybrid retrieval for: '{user_query[:80]}...'")

        retrieval_limit = retrieval_top_k or settings.RETRIEVAL_TOP_K
        hybrid_result = self._retrieve_with_cache(
            user_query=user_query,
            processed=processed,
            retrieval_top_k=retrieval_limit,
            grade_filter=grade_filter,
            masdar_filter=masdar_filter,
            retrieval_mode=retrieval_mode,
        )
        retrieved_hadiths = hybrid_result.hadiths

        if not masdar_filter and not any(_is_primary_source(hadith) for hadith in retrieved_hadiths):
            try:
                primary_result = self._retrieve_with_cache(
                    user_query=user_query,
                    processed=processed,
                    retrieval_top_k=max(retrieval_limit, settings.RERANK_TOP_K * 2),
                    grade_filter=grade_filter,
                    masdar_filter=list(_PRIMARY_SOURCE_BOOKS),
                    retrieval_mode=retrieval_mode,
                    query_embedding=hybrid_result.query_embedding,
                )
                if primary_result.hadiths:
                    retrieved_hadiths = _merge_with_primary_candidates(
                        base_hadiths=retrieved_hadiths,
                        primary_hadiths=primary_result.hadiths,
                        limit=retrieval_limit,
                    )
                    logger.info(
                        f"Primary-source injection added {len(primary_result.hadiths)} "
                        "candidate(s) from Sahih Bukhari/Muslim"
                    )
            except Exception as exc:
                logger.warning(f"Primary-source injection skipped due to retrieval error: {exc}")

        timing["hybrid_retrieval"] = time.time() - t0
        timing.update({f"retrieval_{k}": v for k, v in hybrid_result.timing.items()})

        gen_query_type = "general"
        if processed.query_type == QueryType.METADATA:
            gen_query_type = "metadata"
        elif processed.query_type == QueryType.NARRATOR:
            gen_query_type = "narrator"
        elif processed.query_type == QueryType.EXPLAIN_HADITH:
            gen_query_type = "explain_hadith"
        elif processed.query_type == QueryType.HADITH_LOOKUP:
            gen_query_type = "hadith_lookup"

        if not retrieved_hadiths:
            logger.warning("No hadiths retrieved")
            generation = self.generator.generate(
                query=user_query,
                hadiths=[],
                temperature=temperature,
                verify_grounding=True,
                query_type=gen_query_type,
                metadata_fields=processed.metadata_fields if processed.metadata_fields else None,
                excluded_masdar=processed.excluded_masdar if processed.excluded_masdar else None,
            )
            timing.update(generation.timing)
            timing["total"] = time.time() - total_start
            return RAGResponse(
                query=user_query,
                answer=generation.answer,
                generation=generation,
                timing=timing,
                query_type=gen_query_type,
                answer_intent=generation.answer_intent,
                evidence_sufficient=generation.evidence_sufficient,
                authenticity_of_evidence=generation.authenticity_of_evidence,
                relevance_to_question=generation.relevance_to_question,
                final_sufficiency=generation.final_sufficiency,
            )

        # ──────────────────────────────────────────────
        # Stage 4: Reranking [LOCAL→API — Jina]
        # ──────────────────────────────────────────────
        t0 = time.time()
        logger.info(f"Stage 4: Reranking {len(retrieved_hadiths)} candidates...")

        rerank_limit = rerank_top_k or settings.RERANK_TOP_K
        if _can_skip_rerank_for_exact_lookup(
            processed_query_type=processed.query_type,
            lookup_text=getattr(processed, "dense_query", "") or getattr(processed, "normalized", user_query),
            retrieved_hadiths=retrieved_hadiths,
        ):
            reranked_hadiths = retrieved_hadiths[:rerank_limit]
            timing["rerank_skipped_exact_lookup"] = 1.0
            logger.info("Stage 4: Skipping reranker for high-overlap exact lookup.")
        else:
            reranked_hadiths = self.reranker.rerank(
                query=user_query,
                candidates=retrieved_hadiths,
                top_k=rerank_top_k,
            )
            timing["rerank_skipped_exact_lookup"] = 0.0
        reranked_hadiths = _ensure_primary_in_reranked(reranked_hadiths, retrieved_hadiths)
        timing["rerank"] = time.time() - t0

        # ──────────────────────────────────────────────
        # Stage 5: Generation with Citation Grounding
        # Query-type-aware: passes query_type and metadata_fields to generator
        # ──────────────────────────────────────────────
        t0 = time.time()

        logger.info(
            f"Stage 5: Generating answer from {len(reranked_hadiths)} hadiths "
            f"(type={gen_query_type}, metadata_fields={processed.metadata_fields})"
        )

        metadata_fields = processed.metadata_fields if processed.metadata_fields else None
        excluded_masdar = processed.excluded_masdar if processed.excluded_masdar else None
        cache_key = self._build_generation_cache_key(
            query=user_query,
            hadiths=reranked_hadiths,
            query_type=gen_query_type,
            metadata_fields=metadata_fields,
            excluded_masdar=excluded_masdar,
            temperature=temperature,
        )
        cached_generation = self._get_cached_generation(cache_key)
        if cached_generation is not None:
            generation = cached_generation
            timing["generate_cache_hit"] = 1.0
            logger.info("Stage 5: Generation cache hit")
        else:
            generation = self.generator.generate(
                query=user_query,
                hadiths=reranked_hadiths,
                temperature=temperature,
                verify_grounding=True,
                query_type=gen_query_type,
                metadata_fields=metadata_fields,
                excluded_masdar=excluded_masdar,
            )
            self._cache_generation(cache_key, generation)
            timing["generate_cache_hit"] = 0.0
        timing["generate"] = time.time() - t0
        if cached_generation is None:
            timing.update(generation.timing)
        else:
            timing["generation_total"] = timing["generate"]

        timing["total"] = time.time() - total_start

        logger.info(
            f"✅ Pipeline complete in {timing['total']:.2f}s "
            f"(type={gen_query_type}, grounding: {'✅' if generation.grounding_verified else '⚠️'})"
        )

        return RAGResponse(
            query=user_query,
            answer=generation.answer,
            retrieved_hadiths=retrieved_hadiths,
            reranked_hadiths=reranked_hadiths,
            generation=generation,
            timing=timing,
            query_type=gen_query_type,
            answer_intent=generation.answer_intent,
            evidence_sufficient=generation.evidence_sufficient,
            authenticity_of_evidence=generation.authenticity_of_evidence,
            relevance_to_question=generation.relevance_to_question,
            final_sufficiency=generation.final_sufficiency,
        )


def main():
    """CLI entry point for testing the pipeline."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Default test queries
    default_queries = [
        "ما صحة حديث من غشنا فليس منا",
        "أحاديث عن فضل الصلاة",
        "ما هي أحاديث النية والإخلاص",
    ]

    # Use CLI argument or default
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = default_queries[:1]  # Just first query by default

    # Initialize pipeline
    pipeline = HadithRAGPipeline()

    for query in queries:
        print(f"\n{'=' * 70}")
        response = pipeline.query(query)
        print(response)


if __name__ == "__main__":
    main()
