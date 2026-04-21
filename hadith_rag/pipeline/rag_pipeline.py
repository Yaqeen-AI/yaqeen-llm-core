# ============================================================
# YaqeenAI — Full RAG Pipeline Orchestrator [LOCAL]
# ============================================================
# Wires together all pipeline stages:
#   Query → Preprocess → Hybrid Retrieve → Rerank → Generate → Answer
#
# Supports two modes:
#   1. Hybrid mode: Dense (Jina/ChromaDB) + Sparse (TF-IDF char n-gram) + RRF fusion
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
import time
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
            timing_str += f"  {stage}: {duration:.2f}s\n"

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
    2. [LOCAL] Hybrid retrieval: Dense (Jina→ChromaDB) + Sparse (TF-IDF char n-gram)
    3. [LOCAL] RRF fusion + canonical group deduplication
    4. [LOCAL] BGE reranker cross-encoder scoring (20 → 5)
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

        logger.info("✅ Hadith RAG Pipeline initialized successfully")

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

        hybrid_result = self.hybrid_retriever.retrieve(
            query=user_query,
            fused_top_k=retrieval_top_k or settings.RETRIEVAL_TOP_K,
            grade_filter=grade_filter,
            masdar_filter=masdar_filter,
            retrieval_mode=retrieval_mode,
        )
        retrieved_hadiths = hybrid_result.hadiths
        timing["hybrid_retrieval"] = time.time() - t0
        timing.update({f"retrieval_{k}": v for k, v in hybrid_result.timing.items()})

        gen_query_type = "general"
        if processed.query_type == QueryType.METADATA:
            gen_query_type = "metadata"
        elif processed.query_type == QueryType.NARRATOR:
            gen_query_type = "narrator"
        elif processed.query_type == QueryType.EXPLAIN_HADITH:
            gen_query_type = "explain_hadith"

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
        # Stage 4: Reranking [LOCAL — CPU]
        # ──────────────────────────────────────────────
        t0 = time.time()
        logger.info(f"Stage 4: Reranking {len(retrieved_hadiths)} candidates...")

        reranked_hadiths = self.reranker.rerank(
            query=user_query,
            candidates=retrieved_hadiths,
            top_k=rerank_top_k,
        )
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

        generation = self.generator.generate(
            query=user_query,
            hadiths=reranked_hadiths,
            temperature=temperature,
            verify_grounding=True,
            query_type=gen_query_type,
            metadata_fields=processed.metadata_fields if processed.metadata_fields else None,
            excluded_masdar=processed.excluded_masdar if processed.excluded_masdar else None,
        )
        timing["generate"] = time.time() - t0

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
