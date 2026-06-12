from __future__ import annotations

import csv
import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.judge import LlmEvaluationJudge
from evaluation.metrics import (
    CITATION_COLUMNS,
    RELEVANCE_COLUMNS,
    citation_accuracy,
    documents_to_citations,
    documents_to_ids,
    extract_truth_items,
    mean_or_none,
    parse_expected_sources,
    percentile,
    ranking_metrics,
    retrieved_chunks_accuracy,
)
from orchestrator.models import (
    Citation,
    RagSelection,
    RagSource,
    RetrievedDocument,
    SourceRetrievalConfig,
)
from orchestrator.state import WorkflowState
from orchestrator.workflow import MultiAgentRagWorkflow, _final_citations, _query_for_source, _sources_from_citations
from utils.rate_limits import AsyncSlidingWindowRateLimiter


DEFAULT_QUESTIONS_PATH = Path("yaqeen_rag_evaluation_questions.xlsx")
DEFAULT_OUTPUT_DIR = Path("evaluation_results")


@dataclass
class EvaluationOptions:
    questions_path: Path = DEFAULT_QUESTIONS_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    top_k: int = 5
    limit: int | None = None
    write_csv: bool = True
    use_llm_judge: bool = True
    max_similarity_top_k: int = 10
    max_rerank_top_n: int = 5
    proactive_jina_rate_limit: bool = False
    jina_tpm_limit: int = 100_000
    jina_tpm_safety: float = 0.8
    jina_tokens_per_retrieval: int = 5_000
    jina_retry_wait_seconds: float = 70.0
    jina_max_retries: int = 2
    use_workflow_rewrite_for_retrieval_eval: bool = False
    retrieval_timeout_seconds: float = 45.0
    judge_timeout_seconds: float = 20.0


@dataclass
class RetrievalRunResult:
    source: RagSource
    query: str
    retrieval_query: str
    documents: list[RetrievedDocument]
    latency_ms: float
    rate_limit_wait_ms: float = 0.0


@dataclass
class RetrievalEvaluationResult:
    source: RagSource
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    output_files: list[str] = field(default_factory=list)


@dataclass
class FullEvaluationResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    output_files: list[str] = field(default_factory=list)


async def run_single_source_retrieval(
    workflow: MultiAgentRagWorkflow,
    source: RagSource,
    query: str,
    *,
    config: SourceRetrievalConfig | None = None,
    top_k: int = 5,
    use_workflow_rewrite: bool = True,
    options: EvaluationOptions | None = None,
    jina_limiter: AsyncSlidingWindowRateLimiter | None = None,
) -> RetrievalRunResult:
    adapter = workflow.adapters[source]
    retrieval_query = query
    retrieval_config = config or SourceRetrievalConfig(top_k=top_k, rerank_top_n=max(top_k, 5))

    if use_workflow_rewrite and config is None:
        understanding = await workflow.query_understanding_agent.run(query)
        rewrite = await workflow.query_rewriter_agent.run(query, understanding)
        selection = RagSelection(
            selected_sources=[source],
            route_type="single_source",
            confidence="high",
            reason="Forced single-source evaluation retrieval.",
        )
        plan = await workflow.retrieval_config_agent.run(selection, rewrite, understanding)
        retrieval_config = plan.configs.get(source, retrieval_config)
        retrieval_config = retrieval_config.model_copy(
            update={
                "top_k": top_k,
                "rerank_top_n": max(retrieval_config.rerank_top_n, top_k),
            }
        )
        retrieval_query = rewrite.source_queries.get(source.value) or rewrite.expanded_query or query

    if options is not None:
        retrieval_config = _cap_retrieval_config(retrieval_config, options)
    if options is None:
        start = time.perf_counter()
        documents = await adapter.retrieve(retrieval_query, retrieval_config)
        latency_ms = (time.perf_counter() - start) * 1000
        rate_limit_wait_ms = 0.0
    else:
        documents, rate_limit_wait_ms, latency_ms = await _retrieve_with_jina_limits(
            adapter=adapter,
            query=retrieval_query,
            config=retrieval_config,
            options=options,
            jina_limiter=jina_limiter,
            source_count=1,
        )
    return RetrievalRunResult(
        source=source,
        query=query,
        retrieval_query=retrieval_query,
        documents=documents[:top_k],
        latency_ms=latency_ms,
        rate_limit_wait_ms=rate_limit_wait_ms,
    )


async def run_retrieval_evaluation(
    workflow: MultiAgentRagWorkflow,
    source: RagSource,
    options: EvaluationOptions | None = None,
) -> RetrievalEvaluationResult:
    options = options or EvaluationOptions()
    judge = LlmEvaluationJudge() if options.use_llm_judge else None
    jina_limiter = _build_jina_limiter(options)
    questions = _load_questions(options.questions_path, options.limit)
    rows: list[dict[str, Any]] = []

    for question in questions:
        expected_sources = parse_expected_sources(question.get("Expected Route"))
        if not _is_single_source_category(question, source):
            continue

        row = _base_row(question, expected_sources)
        row["evaluated_source"] = source.value
        relevant_items = extract_truth_items(question, RELEVANCE_COLUMNS)
        expected_citations = extract_truth_items(question, CITATION_COLUMNS)
        try:
            result = await run_single_source_retrieval(
                workflow,
                source,
                str(question["Question"]),
                top_k=options.top_k,
                use_workflow_rewrite=options.use_workflow_rewrite_for_retrieval_eval,
                options=options,
                jina_limiter=jina_limiter,
            )
            metric_values = ranking_metrics(result.documents, relevant_items, k=options.top_k)
            citations = documents_to_citations(result.documents)
            citation_score = citation_accuracy(citations, expected_citations)
            judge_values = await _judge_retrieval_if_needed(
                judge,
                question=str(question["Question"]),
                source=source,
                documents=result.documents,
                needs_ranking=metric_values["recall_at_k"] is None,
                needs_citation=citation_score is None,
                timeout_seconds=options.judge_timeout_seconds,
            )
            row.update(
                {
                    "retrieval_query": result.retrieval_query,
                    "returned_ids": " | ".join(documents_to_ids(result.documents)),
                    "returned_citations": " | ".join(citations),
                    "recall_at_5": _metric_or_judge(metric_values["recall_at_k"], judge_values, "recall_at_5"),
                    "mrr": _metric_or_judge(metric_values["mrr"], judge_values, "mrr"),
                    "ndcg_at_5": _metric_or_judge(metric_values["ndcg_at_k"], judge_values, "ndcg_at_5"),
                    "citation_accuracy": _metric_or_judge(citation_score, judge_values, "citation_accuracy"),
                    "retrieved_chunks_accuracy": _metric_or_judge(
                        _retrieved_chunks_accuracy(result.documents, relevant_items),
                        judge_values,
                        "retrieved_chunks_accuracy",
                    ),
                    "metric_source": _metric_source(metric_values["recall_at_k"], citation_score, judge_values),
                    "judge_relevance_grades": judge_values.get("judge_relevance_grades", ""),
                    "judge_reason": judge_values.get("judge_reason", ""),
                    "judge_error": judge_values.get("judge_error", ""),
                    "latency_ms": round(result.latency_ms, 2),
                    "rate_limit_wait_ms": round(result.rate_limit_wait_ms, 2),
                    "error": "",
                }
            )
        except Exception as exc:
            row.update(_error_metrics(exc))
        rows.append(row)

    summary = _retrieval_summary(source, rows)
    output_files = _write_result_files(options, f"retrieval_{source.value}", rows, summary)
    return RetrievalEvaluationResult(source=source, rows=rows, summary=summary, output_files=output_files)


async def run_full_evaluation(
    workflow: MultiAgentRagWorkflow,
    options: EvaluationOptions | None = None,
) -> FullEvaluationResult:
    options = options or EvaluationOptions()
    judge = LlmEvaluationJudge() if options.use_llm_judge else None
    jina_limiter = _build_jina_limiter(options)
    questions = _load_questions(options.questions_path, options.limit)
    rows: list[dict[str, Any]] = []

    for question in questions:
        expected_sources = parse_expected_sources(question.get("Expected Route"))
        row = _base_row(question, expected_sources)
        relevant_items = extract_truth_items(question, RELEVANCE_COLUMNS)
        expected_citations = extract_truth_items(question, CITATION_COLUMNS)
        try:
            state, response, latency_ms = await _run_full_trace(
                workflow,
                str(question["Question"]),
                options=options,
                jina_limiter=jina_limiter,
            )
            retrieved = state.evidence.documents if state.evidence else state.retrieved_documents[: options.top_k]
            retrieved = retrieved[: options.top_k]
            metric_values = ranking_metrics(retrieved, relevant_items, k=options.top_k)
            citation_score = citation_accuracy(response.citations, expected_citations)
            judge_values = await _judge_full_if_needed(
                judge,
                question=str(question["Question"]),
                answer=response.answer,
                citations=response.citations,
                documents=retrieved,
                needs_ranking=metric_values["recall_at_k"] is None,
                needs_citation=citation_score is None,
                timeout_seconds=options.judge_timeout_seconds,
            )
            predicted_sources = state.selection.selected_sources if state.selection else []
            row.update(
                {
                    "predicted_sources": " + ".join(source.value for source in predicted_sources),
                    "router_correct": _same_sources(expected_sources, predicted_sources),
                    "router_accuracy": 1.0 if _same_sources(expected_sources, predicted_sources) else 0.0,
                    "route_type": state.selection.route_type if state.selection else "",
                    "returned_ids": " | ".join(documents_to_ids(retrieved)),
                    "returned_citations": " | ".join(citation.label for citation in response.citations),
                    "recall_at_5": _metric_or_judge(metric_values["recall_at_k"], judge_values, "recall_at_5"),
                    "mrr": _metric_or_judge(metric_values["mrr"], judge_values, "mrr"),
                    "ndcg_at_5": _metric_or_judge(metric_values["ndcg_at_k"], judge_values, "ndcg_at_5"),
                    "citation_accuracy": _metric_or_judge(citation_score, judge_values, "citation_accuracy"),
                    "retrieved_chunks_accuracy": _metric_or_judge(
                        _retrieved_chunks_accuracy(retrieved, relevant_items),
                        judge_values,
                        "retrieved_chunks_accuracy",
                    ),
                    "metric_source": _metric_source(metric_values["recall_at_k"], citation_score, judge_values),
                    "judge_relevance_grades": judge_values.get("judge_relevance_grades", ""),
                    "judge_reason": judge_values.get("judge_reason", ""),
                    "judge_error": judge_values.get("judge_error", ""),
                    "latency_ms": round(latency_ms, 2),
                    "answer": response.answer,
                    "error": "",
                }
            )
        except Exception as exc:
            row.update(_error_metrics(exc))
            row.update({"predicted_sources": "", "router_correct": False, "router_accuracy": 0.0, "route_type": ""})
        rows.append(row)

    summary = _full_summary(rows)
    output_files = _write_result_files(options, "full_multi_agent_multi_rag", rows, summary)
    return FullEvaluationResult(rows=rows, summary=summary, output_files=output_files)


async def _run_full_trace(
    workflow: MultiAgentRagWorkflow,
    query: str,
    *,
    options: EvaluationOptions,
    jina_limiter: AsyncSlidingWindowRateLimiter | None,
) -> tuple[WorkflowState, Any, float]:
    state = WorkflowState(original_query=query)
    start = time.perf_counter()
    state.understanding = await workflow.query_understanding_agent.run(query)
    state.rewrite = await workflow.query_rewriter_agent.run(query, state.understanding)
    state.normalized_query = state.rewrite.normalized_query
    state.selection = await workflow.rag_selector_agent.run(query, state.understanding, state.rewrite)

    if state.selection.route_type == "out_of_scope" or not state.selection.selected_sources:
        from orchestrator.models import AskResponse

        response = AskResponse(
            answer="I can only answer questions grounded in the available Quran, Hadith, and Fiqh sources.",
            citations=[],
            follow_up_questions=[],
            sources=[],
            cache_hit=False,
            metadata={"route_type": "out_of_scope", "reason": state.selection.reason},
        )
        return state, response, (time.perf_counter() - start) * 1000

    state.retrieval_plan = await workflow.retrieval_config_agent.run(state.selection, state.rewrite, state.understanding)
    _cap_retrieval_plan(state, options)
    state.retrieved_documents = await _retrieve_eval_sources(workflow, state, options, jina_limiter)
    final_top_k = state.retrieval_plan.final_top_k if state.retrieval_plan else 8
    state.evidence = await workflow.aggregation_agent.run(state.retrieved_documents, final_top_k=final_top_k)
    state.generated = await workflow.generation_agent.run(query, state.rewrite, state.evidence)
    evidence_citations = await workflow.citation_agent.run(state.evidence.documents)
    citations: list[Citation] = _final_citations(state.generated.answer, evidence_citations)

    from orchestrator.models import AskResponse

    response = AskResponse(
        answer=state.generated.answer,
        citations=citations,
        follow_up_questions=state.generated.follow_up_questions,
        sources=_sources_from_citations(citations),
        cache_hit=False,
        metadata={
            "route_type": state.selection.route_type,
            "selection_confidence": state.selection.confidence,
            "documents_retrieved": len(state.retrieved_documents),
            "documents_used": len(state.evidence.documents),
        },
    )
    state.response = response
    return state, response, (time.perf_counter() - start) * 1000


def _load_questions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    dataframe = pd.read_excel(path, sheet_name="Evaluation Questions")
    if limit is not None:
        dataframe = dataframe.head(limit)
    return dataframe.where(pd.notnull(dataframe), None).to_dict(orient="records")


def _base_row(question: dict[str, Any], expected_sources: list[RagSource]) -> dict[str, Any]:
    return {
        "id": question.get("ID"),
        "category": question.get("Category"),
        "difficulty": question.get("Difficulty"),
        "question": question.get("Question"),
        "expected_route": question.get("Expected Route"),
        "expected_sources": " + ".join(source.value for source in expected_sources),
    }


def _is_single_source_category(question: dict[str, Any], source: RagSource) -> bool:
    category = str(question.get("Category") or "").strip().casefold()
    return category == source.value


def _retrieval_summary(source: RagSource, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source": source.value,
        "questions_evaluated": len(rows),
        "recall_at_5_mean": _rounded_mean(rows, "recall_at_5"),
        "mrr_mean": _rounded_mean(rows, "mrr"),
        "ndcg_at_5_mean": _rounded_mean(rows, "ndcg_at_5"),
        "citation_accuracy_mean": _rounded_mean(rows, "citation_accuracy"),
        "retrieved_chunks_accuracy_mean": _rounded_mean(rows, "retrieved_chunks_accuracy"),
        "latency_p95_ms": _rounded_percentile(rows, "latency_ms", 95),
        "rate_limit_wait_p95_ms": _rounded_percentile(rows, "rate_limit_wait_ms", 95),
        "missing_relevance_count": sum(1 for row in rows if row.get("recall_at_5") is None),
        "missing_citation_truth_count": sum(1 for row in rows if row.get("citation_accuracy") is None),
        "llm_judged_count": sum(1 for row in rows if row.get("metric_source") == "llm_judge"),
        "errors": sum(1 for row in rows if row.get("error")),
    }


def _full_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "questions_evaluated": len(rows),
        "router_accuracy": _rounded_mean(rows, "router_accuracy"),
        "recall_at_5_mean": _rounded_mean(rows, "recall_at_5"),
        "mrr_mean": _rounded_mean(rows, "mrr"),
        "ndcg_at_5_mean": _rounded_mean(rows, "ndcg_at_5"),
        "citation_accuracy_mean": _rounded_mean(rows, "citation_accuracy"),
        "retrieved_chunks_accuracy_mean": _rounded_mean(rows, "retrieved_chunks_accuracy"),
        "latency_p95_ms": _rounded_percentile(rows, "latency_ms", 95),
        "missing_relevance_count": sum(1 for row in rows if row.get("recall_at_5") is None),
        "missing_citation_truth_count": sum(1 for row in rows if row.get("citation_accuracy") is None),
        "llm_judged_count": sum(1 for row in rows if row.get("metric_source") == "llm_judge"),
        "errors": sum(1 for row in rows if row.get("error")),
    }


def _build_jina_limiter(options: EvaluationOptions) -> AsyncSlidingWindowRateLimiter:
    effective_limit = max(1, int(options.jina_tpm_limit * options.jina_tpm_safety))
    return AsyncSlidingWindowRateLimiter(
        limit=effective_limit,
        window_seconds=60,
        name="jina-eval-tpm",
        enabled=options.proactive_jina_rate_limit,
    )


async def _retrieve_eval_sources(
    workflow: MultiAgentRagWorkflow,
    state: WorkflowState,
    options: EvaluationOptions,
    jina_limiter: AsyncSlidingWindowRateLimiter | None,
) -> list[RetrievedDocument]:
    assert state.selection is not None
    assert state.rewrite is not None
    assert state.retrieval_plan is not None

    documents: list[RetrievedDocument] = []
    for source in state.selection.selected_sources:
        adapter = workflow.adapters.get(source)
        config = state.retrieval_plan.configs.get(source)
        if adapter is None or config is None:
            continue
        source_documents, _, _ = await _retrieve_with_jina_limits(
            adapter=adapter,
            query=_query_for_source(state, source),
            config=config,
            options=options,
            jina_limiter=jina_limiter,
            source_count=1,
        )
        documents.extend(source_documents)
    return documents


async def _retrieve_with_jina_limits(
    *,
    adapter: Any,
    query: str,
    config: SourceRetrievalConfig,
    options: EvaluationOptions,
    jina_limiter: AsyncSlidingWindowRateLimiter | None,
    source_count: int,
) -> tuple[list[RetrievedDocument], float, float]:
    attempts = max(1, options.jina_max_retries + 1)
    total_wait_ms = 0.0
    for attempt in range(attempts):
        if jina_limiter is not None:
            wait_start = time.perf_counter()
            await jina_limiter.acquire(options.jina_tokens_per_retrieval * max(1, source_count))
            total_wait_ms += (time.perf_counter() - wait_start) * 1000
        try:
            retrieval_start = time.perf_counter()
            documents = await asyncio.wait_for(
                adapter.retrieve(query, config),
                timeout=options.retrieval_timeout_seconds,
            )
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
            return documents, total_wait_ms, retrieval_ms
        except Exception as exc:
            if not _is_jina_rate_limit_error(exc) or attempt >= attempts - 1:
                raise
            total_wait_ms += options.jina_retry_wait_seconds * 1000
            await asyncio.sleep(options.jina_retry_wait_seconds)
    return [], total_wait_ms, 0.0


def _cap_retrieval_plan(state: WorkflowState, options: EvaluationOptions) -> None:
    if state.retrieval_plan is None:
        return
    capped = {
        source: _cap_retrieval_config(config, options)
        for source, config in state.retrieval_plan.configs.items()
    }
    state.retrieval_plan = state.retrieval_plan.model_copy(update={"configs": capped})


def _cap_retrieval_config(config: SourceRetrievalConfig, options: EvaluationOptions) -> SourceRetrievalConfig:
    top_k = min(config.top_k, options.top_k)
    rerank_cap = max(top_k, options.max_rerank_top_n)
    return config.model_copy(
        update={
            "top_k": top_k,
            "similarity_top_k": min(config.similarity_top_k, max(top_k, options.max_similarity_top_k)),
            "rerank_top_n": min(config.rerank_top_n, rerank_cap),
        }
    )


def _is_jina_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).casefold()
    if "token rate limit exceeded" in text:
        return True
    return "jina" in text and ("429" in text or "too many requests" in text or "rate limit" in text)


def _write_result_files(options: EvaluationOptions, name: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> list[str]:
    if not options.write_csv:
        return []
    options.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = options.output_dir / f"{name}_details.csv"
    summary_path = options.output_dir / f"{name}_summary.csv"
    _write_csv(detail_path, rows)
    _write_csv(summary_path, [summary])
    return [str(detail_path), str(summary_path)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def _judge_retrieval_if_needed(
    judge: LlmEvaluationJudge | None,
    *,
    question: str,
    source: RagSource,
    documents: list[RetrievedDocument],
    needs_ranking: bool,
    needs_citation: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    if judge is None or not (needs_ranking or needs_citation):
        return {}
    try:
        result = await asyncio.wait_for(
            judge.evaluate_retrieval(question=question, documents=documents, source=source.value),
            timeout=timeout_seconds,
        )
        return _judge_values(result)
    except Exception as exc:
        return {"judge_error": str(exc)}


async def _judge_full_if_needed(
    judge: LlmEvaluationJudge | None,
    *,
    question: str,
    answer: str,
    citations: list[Citation],
    documents: list[RetrievedDocument],
    needs_ranking: bool,
    needs_citation: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    if judge is None or not (needs_ranking or needs_citation):
        return {}
    try:
        result = await asyncio.wait_for(
            judge.evaluate_full(question=question, answer=answer, citations=citations, documents=documents),
            timeout=timeout_seconds,
        )
        return _judge_values(result)
    except Exception as exc:
        return {"judge_error": str(exc)}


def _judge_values(result: Any) -> dict[str, Any]:
    return {
        "recall_at_5": result.recall_at_5,
        "mrr": result.mrr,
        "ndcg_at_5": result.ndcg_at_5,
        "citation_accuracy": result.citation_accuracy,
        "retrieved_chunks_accuracy": result.retrieved_chunks_accuracy,
        "judge_relevance_grades": " | ".join(
            f"{item.rank}:{item.relevance_grade}" for item in result.judged_documents
        ),
        "judge_reason": result.reason,
    }


def _metric_or_judge(value: float | None, judge_values: dict[str, Any], key: str) -> float | None:
    if value is not None:
        return value
    judged = judge_values.get(key)
    return round(float(judged), 4) if judged is not None else None


def _retrieved_chunks_accuracy(documents: list[RetrievedDocument], relevant_items: list[str]) -> float | None:
    return retrieved_chunks_accuracy(documents, relevant_items)


def _metric_source(
    ranking_value: float | None,
    citation_value: float | None,
    judge_values: dict[str, Any],
) -> str:
    if ranking_value is not None or citation_value is not None:
        if any(key in judge_values for key in ("recall_at_5", "citation_accuracy")):
            return "gold_and_llm_judge"
        return "gold"
    if any(key in judge_values for key in ("recall_at_5", "citation_accuracy")):
        return "llm_judge"
    if judge_values.get("judge_error"):
        return "llm_judge_error"
    return "missing"


def _same_sources(expected: list[RagSource], predicted: list[RagSource]) -> bool:
    return set(expected) == set(predicted)


def _rounded_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    value = mean_or_none(row.get(key) for row in rows)
    return round(value, 4) if value is not None else None


def _rounded_percentile(rows: list[dict[str, Any]], key: str, percent: float) -> float | None:
    value = percentile([row[key] for row in rows if row.get(key) is not None], percent)
    return round(value, 2) if value is not None else None


def _error_metrics(exc: Exception) -> dict[str, Any]:
    message = str(exc) or exc.__class__.__name__
    return {
        "retrieval_query": "",
        "returned_ids": "",
        "returned_citations": "",
        "recall_at_5": None,
        "mrr": None,
        "ndcg_at_5": None,
        "citation_accuracy": None,
        "retrieved_chunks_accuracy": None,
        "metric_source": "error",
        "judge_relevance_grades": "",
        "judge_reason": "",
        "judge_error": "",
        "latency_ms": None,
        "rate_limit_wait_ms": None,
        "answer": "",
        "error": message,
    }
