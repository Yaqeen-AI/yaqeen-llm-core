from __future__ import annotations

from pydantic import BaseModel, Field

from orchestrator.models import (
    AggregatedEvidence,
    AskResponse,
    GeneratedAnswer,
    QueryRewrite,
    QueryUnderstanding,
    RagSelection,
    RetrievalPlan,
    RetrievedDocument,
)


class WorkflowState(BaseModel):
    original_query: str
    normalized_query: str = ""
    cache_hit: bool = False
    cached_response: AskResponse | None = None
    understanding: QueryUnderstanding | None = None
    rewrite: QueryRewrite | None = None
    selection: RagSelection | None = None
    retrieval_plan: RetrievalPlan | None = None
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    evidence: AggregatedEvidence | None = None
    generated: GeneratedAnswer | None = None
    response: AskResponse | None = None

