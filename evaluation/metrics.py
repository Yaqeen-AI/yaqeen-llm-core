from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from orchestrator.models import Citation, RagSource, RetrievedDocument


SOURCE_LABELS = {
    "quran": RagSource.QURAN,
    "قرآن": RagSource.QURAN,
    "القرآن": RagSource.QURAN,
    "hadith": RagSource.HADITH,
    "حديث": RagSource.HADITH,
    "fiqh": RagSource.FIQH,
    "فقه": RagSource.FIQH,
}

RELEVANCE_COLUMNS = (
    "Relevant Document IDs",
    "Relevant Documents",
    "Relevant IDs",
    "Expected Document IDs",
    "Expected Documents",
    "Expected IDs",
    "Ground Truth Document IDs",
    "Ground Truth IDs",
    "Gold Document IDs",
    "Gold IDs",
)

CITATION_COLUMNS = (
    "Expected Citations",
    "Expected Citation",
    "Relevant Citations",
    "Ground Truth Citations",
    "Gold Citations",
    "Citation",
)


def parse_expected_sources(value: Any) -> list[RagSource]:
    text = str(value or "").strip()
    if not text:
        return []

    sources: list[RagSource] = []
    for part in re.split(r"\s*(?:\+|,|،|/|\||;)\s*", text):
        normalized = _normalize_label(part)
        source = SOURCE_LABELS.get(normalized)
        if source and source not in sources:
            sources.append(source)
    return sources


def extract_truth_items(row: dict[str, Any], columns: Iterable[str]) -> list[str]:
    items: list[str] = []
    for column in columns:
        value = row.get(column)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        items.extend(_split_truth_value(value))
    return _dedupe(items)


def ranking_metrics(
    documents: list[RetrievedDocument],
    relevant_items: list[str],
    *,
    k: int = 5,
) -> dict[str, float | None]:
    if not relevant_items:
        return {"recall_at_k": None, "mrr": None, "ndcg_at_k": None}

    limited = documents[:k]
    relevant_norms = [_normalize_match_text(item) for item in relevant_items if str(item).strip()]
    if not relevant_norms:
        return {"recall_at_k": None, "mrr": None, "ndcg_at_k": None}

    hits_by_truth: set[int] = set()
    relevance_by_rank: list[int] = []
    first_hit_rank: int | None = None

    for rank, document in enumerate(limited, start=1):
        matched_indexes = _matched_truth_indexes(document, relevant_norms)
        is_relevant = bool(matched_indexes)
        relevance_by_rank.append(1 if is_relevant else 0)
        if is_relevant and first_hit_rank is None:
            first_hit_rank = rank
        hits_by_truth.update(matched_indexes)

    recall = len(hits_by_truth) / len(relevant_norms)
    mrr = 1 / first_hit_rank if first_hit_rank else 0.0
    ndcg = _ndcg(relevance_by_rank, ideal_relevant=min(len(relevant_norms), k))
    return {"recall_at_k": recall, "mrr": mrr, "ndcg_at_k": ndcg}


def citation_accuracy(citations: list[Citation | str], expected_items: list[str]) -> float | None:
    if not expected_items:
        return None

    expected = [_normalize_match_text(item) for item in expected_items if str(item).strip()]
    observed = [_normalize_match_text(_citation_label(item)) for item in citations]
    if not expected:
        return None

    hits = 0
    for item in expected:
        if any(_text_match(item, candidate) for candidate in observed):
            hits += 1
    return hits / len(expected)


def retrieved_chunks_accuracy(documents: list[RetrievedDocument], relevant_items: list[str]) -> float | None:
    if not relevant_items or not documents:
        return None

    relevant_norms = [_normalize_match_text(item) for item in relevant_items if str(item).strip()]
    if not relevant_norms:
        return None

    relevant_documents = sum(1 for document in documents if _matched_truth_indexes(document, relevant_norms))
    return relevant_documents / len(documents)


def percentile(values: list[float], percent: float) -> float | None:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    rank = max(1, math.ceil((percent / 100) * len(clean)))
    return clean[rank - 1]


def mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def documents_to_ids(documents: list[RetrievedDocument]) -> list[str]:
    return [document.id for document in documents]


def documents_to_citations(documents: list[RetrievedDocument]) -> list[str]:
    return [document.citation.label for document in documents if document.citation and document.citation.label]


def _split_truth_value(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[\n;|]+", text) if part.strip()]


def _matched_truth_indexes(document: RetrievedDocument, relevant_norms: list[str]) -> set[int]:
    candidates = [_normalize_match_text(value) for value in _document_match_values(document)]
    matched: set[int] = set()
    for index, truth in enumerate(relevant_norms):
        if any(_text_match(truth, candidate) for candidate in candidates):
            matched.add(index)
    return matched


def _document_match_values(document: RetrievedDocument) -> list[str]:
    values = [
        document.id,
        document.text[:800],
        document.citation.label if document.citation else "",
    ]
    for value in document.metadata.values():
        if isinstance(value, (str, int, float)):
            values.append(str(value))
        elif isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value)
    return [value for value in values if value]


def _ndcg(relevance_by_rank: list[int], *, ideal_relevant: int) -> float:
    if ideal_relevant <= 0:
        return 0.0
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance_by_rank, start=1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return dcg / ideal if ideal else 0.0


def _citation_label(citation: Citation | str) -> str:
    if isinstance(citation, Citation):
        return citation.label
    return str(citation)


def _text_match(truth: str, candidate: str) -> bool:
    if not truth or not candidate:
        return False
    if truth == candidate:
        return True
    if len(truth) >= 4 and truth in candidate:
        return True
    return len(candidate) >= 4 and candidate in truth


def _normalize_label(value: str) -> str:
    return str(value or "").strip().casefold()


def _normalize_match_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[\[\](){}]", " ", text)
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    text = re.sub(r"[^\w\u0600-\u06FF:/.-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean or clean in seen:
            continue
        result.append(clean)
        seen.add(clean)
    return result
