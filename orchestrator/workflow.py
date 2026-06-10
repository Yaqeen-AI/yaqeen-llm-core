from __future__ import annotations

import asyncio
import logging
import os

from agents import (
    AggregationAgent,
    CitationAgent,
    GenerationAgent,
    QueryRewriterAgent,
    QueryUnderstandingAgent,
    RagSelectorAgent,
    RetrievalConfigurationAgent,
)
from cache.semantic_cache import SemanticCache
from orchestrator.models import AskResponse, RagSource, RetrievedDocument
from orchestrator.state import WorkflowState
from rag_adapters import FiqhAdapter, HadithAdapter, QuranAdapter, RagAdapter

logger = logging.getLogger(__name__)
_CACHE_CHECK_TIMEOUT_SECONDS = float(os.getenv("YAQEEN_CACHE_CHECK_TIMEOUT_SECONDS", "1.2"))


class MultiAgentRagWorkflow:
    def __init__(
        self,
        cache: SemanticCache | None = None,
        query_understanding_agent: QueryUnderstandingAgent | None = None,
        query_rewriter_agent: QueryRewriterAgent | None = None,
        rag_selector_agent: RagSelectorAgent | None = None,
        retrieval_config_agent: RetrievalConfigurationAgent | None = None,
        aggregation_agent: AggregationAgent | None = None,
        generation_agent: GenerationAgent | None = None,
        citation_agent: CitationAgent | None = None,
        adapters: dict[RagSource, RagAdapter] | None = None,
    ) -> None:
        self.cache = cache or SemanticCache()
        self.query_understanding_agent = query_understanding_agent or QueryUnderstandingAgent()
        self.query_rewriter_agent = query_rewriter_agent or QueryRewriterAgent()
        self.rag_selector_agent = rag_selector_agent or RagSelectorAgent()
        self.retrieval_config_agent = retrieval_config_agent or RetrievalConfigurationAgent()
        self.aggregation_agent = aggregation_agent or AggregationAgent()
        self.generation_agent = generation_agent or GenerationAgent()
        self.citation_agent = citation_agent or CitationAgent()
        self.adapters = adapters or {
            RagSource.QURAN: QuranAdapter(),
            RagSource.HADITH: HadithAdapter(),
            RagSource.FIQH: FiqhAdapter(),
        }

    async def ask(self, query: str) -> AskResponse:
        state = WorkflowState(original_query=query)

        cached = await self._check_cache_safe(query)
        if cached.hit and cached.response:
            state.cache_hit = True
            state.cached_response = cached.response.model_copy(
                update={
                    "cache_hit": True,
                    "metadata": {**cached.response.metadata, "cache_similarity": cached.similarity},
                }
            )
            return state.cached_response

        state.understanding = await self.query_understanding_agent.run(query)
        state.rewrite = await self.query_rewriter_agent.run(query, state.understanding)
        state.normalized_query = state.rewrite.normalized_query
        state.selection = await self.rag_selector_agent.run(query, state.understanding, state.rewrite)

        if state.selection.route_type == "out_of_scope" or not state.selection.selected_sources:
            response = AskResponse(
                answer="I can only answer questions grounded in the available Quran, Hadith, and Fiqh sources.",
                citations=[],
                follow_up_questions=[],
                sources=[],
                cache_hit=False,
                metadata={"route_type": "out_of_scope", "reason": state.selection.reason},
            )
            state.response = response
            return response

        state.retrieval_plan = await self.retrieval_config_agent.run(state.selection, state.rewrite, state.understanding)
        state.retrieved_documents = await self._retrieve_parallel(state)
        final_top_k = state.retrieval_plan.final_top_k if state.retrieval_plan else 8
        state.evidence = await self.aggregation_agent.run(state.retrieved_documents, final_top_k=final_top_k)
        state.generated = await self.generation_agent.run(query, state.rewrite, state.evidence)
        citations = await self.citation_agent.run(state.evidence.documents)

        response = AskResponse(
            answer=state.generated.answer,
            citations=state.generated.citations or citations,
            follow_up_questions=state.generated.follow_up_questions,
            sources=state.selection.selected_sources,
            cache_hit=False,
            metadata={
                "route_type": state.selection.route_type,
                "selection_confidence": state.selection.confidence,
                "documents_retrieved": len(state.retrieved_documents),
                "documents_used": len(state.evidence.documents),
            },
        )
        state.response = response
        asyncio.create_task(self._store_cache_safe(query, response))
        return response

    async def _retrieve_parallel(self, state: WorkflowState) -> list[RetrievedDocument]:
        assert state.selection is not None
        assert state.rewrite is not None
        assert state.retrieval_plan is not None

        tasks = []
        logger.debug(
            "Retrieval plan configs: keys=%s",
            list(state.retrieval_plan.configs.keys()),
        )
        for source in state.selection.selected_sources:
            adapter = self.adapters.get(source)
            config = state.retrieval_plan.configs.get(source)
            if adapter is None or config is None:
                logger.warning("Skipping %s because adapter or retrieval config is missing.", source)
                continue
            tasks.append(adapter.retrieve(_query_for_source(state, source), config))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        documents: list[RetrievedDocument] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("RAG retrieval failed: %s", result, exc_info=result)
                continue
            documents.extend(result)
        return documents

    async def _store_cache_safe(self, query: str, response: AskResponse) -> None:
        try:
            await self.cache.store(query, response)
        except Exception:
            logger.warning("Failed to store semantic cache entry.", exc_info=True)

    async def _check_cache_safe(self, query: str):
        try:
            return await asyncio.wait_for(self.cache.check(query), timeout=_CACHE_CHECK_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.info("Semantic cache lookup timed out after %.1fs; continuing with cache miss.", _CACHE_CHECK_TIMEOUT_SECONDS)
            from cache.semantic_cache import CacheLookup

            return CacheLookup(hit=False)
        except Exception:
            logger.warning("Semantic cache lookup failed; continuing with cache miss.", exc_info=True)
            from cache.semantic_cache import CacheLookup

            return CacheLookup(hit=False)


def build_default_workflow() -> MultiAgentRagWorkflow:
    return MultiAgentRagWorkflow()


def _query_for_source(state: WorkflowState, source: RagSource) -> str:
    assert state.rewrite is not None
    source_query = state.rewrite.source_queries.get(source.value)
    if source_query:
        return source_query
    return state.rewrite.expanded_query or state.rewrite.rewritten_query or state.original_query
