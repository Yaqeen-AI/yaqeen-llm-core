from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, Field


class RagSource(StrEnum):
    QURAN = "quran"
    HADITH = "hadith"
    FIQH = "fiqh"


class Citation(BaseModel):
    source: RagSource
    label: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedDocument(BaseModel):
    id: str
    source: RagSource
    text: str
    score: float = 0.0
    normalized_score: float = 0.0
    citation: Citation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryUnderstanding(BaseModel):
    language: str = "unknown"
    intent: str = "unknown"
    domain: str = "unknown"
    specificity: Literal["specific", "general"] = "general"
    retrieval_depth: Literal["pinpoint", "focused", "expanded", "survey"] = "focused"
    query_scope: Literal[
        "single_reference",
        "range_reference",
        "named_section",
        "narrative",
        "broad_theme",
        "comparative",
        "ruling",
        "metadata_lookup",
        "unknown",
    ] = "unknown"
    tafsir_depth: Literal["auto", "concise", "detailed", "both"] = "auto"
    requested_references: dict[str, Any] = Field(default_factory=dict)
    named_entities: dict[str, list[str]] = Field(default_factory=dict)
    key_concepts: list[str] = Field(default_factory=list)
    evidence_need: Literal["direct_text", "tafsir_context", "hadith_grade", "fiqh_positions", "mixed", "unknown"] = "unknown"
    answer_style: Literal["direct", "structured", "comparative", "step_by_step", "summary", "unknown"] = "unknown"
    retrieval_notes: list[str] = Field(default_factory=list)
    wants_explanation: bool = False
    wants_summary: bool = False
    ambiguity_detected: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class QueryRewrite(BaseModel):
    normalized_query: str
    rewritten_query: str
    expanded_query: str
    source_queries: dict[str, str] = Field(default_factory=dict)
    search_terms: list[str] = Field(default_factory=list)
    must_match_terms: list[str] = Field(default_factory=list)
    quran_reference_terms: list[str] = Field(default_factory=list)
    hadith_reference_terms: list[str] = Field(default_factory=list)
    fiqh_reference_terms: list[str] = Field(default_factory=list)
    negative_terms: list[str] = Field(default_factory=list)


class RagSelection(BaseModel):
    selected_sources: list[RagSource] = Field(default_factory=list)
    route_type: Literal["single_source", "multi_source", "out_of_scope"] = "out_of_scope"
    confidence: Literal["high", "medium", "low"] = "low"
    reason: str = ""


class SourceRetrievalConfig(BaseModel):
    top_k: int = Field(default=8, ge=1, le=50)
    similarity_top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_n: int = Field(default=8, ge=1, le=50)
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    filters: dict[str, Any] = Field(default_factory=dict)
    skip_rerank: bool = False


class RetrievalPlan(BaseModel):
    model_config = {"populate_by_name": True}

    configs: dict[RagSource, SourceRetrievalConfig] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "configs", "retrieval_config", "retrieval_configs", "retrieval_configuration",
        ),
    )
    final_top_k: int = Field(default=8, ge=1, le=30)


class AggregatedEvidence(BaseModel):
    documents: list[RetrievedDocument] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    sources: list[RagSource] | None = Field(default=None, min_length=1, max_length=3)

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    sources: list[RagSource] = Field(default_factory=list)
    cache_hit: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
