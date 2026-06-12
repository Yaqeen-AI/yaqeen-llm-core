from __future__ import annotations

import json
import math
import os
from typing import Any

from pydantic import BaseModel, Field

from agents.base import GeminiLLMClient, LLMClient
from orchestrator.models import Citation, RetrievedDocument


SYSTEM_PROMPT = """You are an evaluation judge for the Yaqeen AI Islamic RAG system.
Evaluate only the supplied query, answer, citations, and retrieved documents.
Return one valid JSON object matching the requested schema.

Rules:
- Do not answer the user's religious question.
- Judge relevance and citation support conservatively.
- A document is relevant if it contains evidence that helps answer the query, not merely matching generic words.
- Use graded relevance:
  0 = irrelevant or misleading
  1 = weakly related but insufficient
  2 = relevant and useful
  3 = directly answers or strongly supports the answer
- Citation accuracy is the fraction of cited labels that are present in the evidence and genuinely support nearby answer claims.
- If evaluating retrieval only, citation accuracy means the returned citation labels correctly identify their evidence and are relevant to the query.
- If the documents are insufficient, give low recall even if one document is partially relevant.
- Be repeatable and avoid generous grading."""

_MAX_DOC_TEXT_CHARS = int(os.getenv("YAQEEN_EVAL_JUDGE_DOC_CHAR_LIMIT", "1600"))


class JudgedDocument(BaseModel):
    rank: int
    relevant: bool
    relevance_grade: int = Field(ge=0, le=3)
    reason: str = ""


class JudgeEvaluation(BaseModel):
    recall_at_5: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg_at_5: float = Field(ge=0.0, le=1.0)
    citation_accuracy: float = Field(ge=0.0, le=1.0)
    retrieved_chunks_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    judged_documents: list[JudgedDocument] = Field(default_factory=list)
    reason: str = ""


class LlmEvaluationJudge:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or GeminiLLMClient()

    async def evaluate_retrieval(
        self,
        *,
        question: str,
        documents: list[RetrievedDocument],
        source: str,
    ) -> JudgeEvaluation:
        payload = {
            "task": "retrieval_only_evaluation",
            "question": question,
            "source_being_evaluated": source,
            "documents": [_document_payload(document, rank) for rank, document in enumerate(documents[:5], start=1)],
            "metric_definitions": _metric_definitions(),
        }
        return await self._evaluate(payload)

    async def evaluate_full(
        self,
        *,
        question: str,
        answer: str,
        citations: list[Citation],
        documents: list[RetrievedDocument],
    ) -> JudgeEvaluation:
        payload = {
            "task": "full_multi_agent_multi_rag_evaluation",
            "question": question,
            "answer": answer,
            "citations": [citation.model_dump(mode="json") for citation in citations],
            "documents": [_document_payload(document, rank) for rank, document in enumerate(documents[:5], start=1)],
            "metric_definitions": _metric_definitions(),
        }
        return await self._evaluate(payload)

    async def _evaluate(self, payload: dict[str, Any]) -> JudgeEvaluation:
        raw = await self.llm.generate_json(SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
        judged = JudgeEvaluation.model_validate(_coerce_payload(raw))
        return _recompute_rank_metrics(judged)


def _metric_definitions() -> dict[str, str]:
    return {
        "recall_at_5": "0 to 1 judgment of whether the top-5 evidence contains enough relevant evidence to answer the question.",
        "mrr": "1 / rank of the first relevant document, or 0 if no relevant document exists.",
        "ndcg_at_5": "NDCG computed from relevance grades 0-3 for ranks 1 through 5.",
        "citation_accuracy": "0 to 1 fraction of returned/generated citations that are accurate and supported by the provided evidence.",
        "retrieved_chunks_accuracy": "0 to 1 fraction of returned chunks that are relevant to the question.",
    }


def _document_payload(document: RetrievedDocument, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "id": document.id,
        "source": document.source,
        "score": document.score,
        "citation_label": document.citation.label if document.citation else "",
        "citation": document.citation.model_dump(mode="json") if document.citation else None,
        "metadata": _compact_metadata(document.metadata),
        "text": _truncate(document.text, _MAX_DOC_TEXT_CHARS),
    }


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in metadata.items():
        if value in (None, "", [], {}):
            continue
        if isinstance(value, str):
            compact[key] = _truncate(value, 300)
        elif isinstance(value, (int, float, bool)):
            compact[key] = value
        elif isinstance(value, (list, tuple, set)):
            compact[key] = list(value)[:8]
        if len(compact) >= 16:
            break
    return compact


def _coerce_payload(payload: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(payload or {})
    docs = coerced.get("judged_documents")
    if not isinstance(docs, list):
        docs = []
    coerced["judged_documents"] = [_coerce_doc(item, index + 1) for index, item in enumerate(docs[:5])]

    for key in ("recall_at_5", "mrr", "ndcg_at_5", "citation_accuracy", "retrieved_chunks_accuracy"):
        coerced[key] = _bounded_float(coerced.get(key), 0.0, 1.0)
    coerced["reason"] = str(coerced.get("reason") or "").strip()
    return coerced


def _coerce_doc(value: Any, fallback_rank: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    grade = int(_bounded_float(value.get("relevance_grade"), 0.0, 3.0))
    return {
        "rank": int(value.get("rank") or fallback_rank),
        "relevant": bool(value.get("relevant", grade >= 2)),
        "relevance_grade": grade,
        "reason": str(value.get("reason") or "").strip(),
    }


def _recompute_rank_metrics(judged: JudgeEvaluation) -> JudgeEvaluation:
    docs = sorted(judged.judged_documents, key=lambda item: item.rank)[:5]
    if not docs:
        return judged

    first_relevant_rank = next((doc.rank for doc in docs if doc.relevant or doc.relevance_grade >= 2), None)
    mrr = 1 / first_relevant_rank if first_relevant_rank else 0.0
    ndcg = _ndcg([doc.relevance_grade for doc in docs])
    chunk_accuracy = sum(1 for doc in docs if doc.relevant or doc.relevance_grade >= 2) / len(docs)
    return judged.model_copy(
        update={
            "mrr": round(mrr, 4),
            "ndcg_at_5": round(ndcg, 4),
            "retrieved_chunks_accuracy": round(chunk_accuracy, 4),
        }
    )


def _ndcg(grades: list[int]) -> float:
    if not grades:
        return 0.0
    dcg = sum(((2**grade) - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))
    ideal_grades = sorted(grades, reverse=True)
    ideal = sum(((2**grade) - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades, start=1))
    return dcg / ideal if ideal else 0.0


def _bounded_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = minimum
    return max(minimum, min(maximum, number))


def _truncate(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip() + "..."
