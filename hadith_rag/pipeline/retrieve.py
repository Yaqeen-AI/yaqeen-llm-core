# ============================================================
# YaqeenAI — ChromaDB Retrieval [LOCAL]
# ============================================================
# Queries the pre-built ChromaDB collection for similar hadiths.
# This runs LOCALLY — ChromaDB is already populated (built on Colab).
#
# Performs cosine similarity search with optional metadata filtering
# by grade and/or source book (masdar).

import logging
from typing import Optional
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings as ChromaSettings

from pipeline.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedHadith:
    """A single hadith retrieved from ChromaDB."""

    id: str
    text_ar: str  # The embedded text (normalized, no tashkeel)
    distance: float  # Cosine distance (lower = more similar)
    grade: str = ""
    grade_ar: str = ""
    ruling: str = ""
    rawi: str = ""
    muhaddith: str = ""
    masdar: str = ""
    safha_raqam: str = ""
    category: str = ""
    subcategory_name: str = ""
    hadith_tag: str = ""
    has_explanation: bool = False
    canonical_group_id: str = ""

    @property
    def similarity_score(self) -> float:
        """Convert cosine distance to similarity score (0-1)."""
        return 1.0 - self.distance


@dataclass
class RetrievalResult:
    """Result of a ChromaDB retrieval query."""

    query: str
    hadiths: list[RetrievedHadith] = field(default_factory=list)
    total_candidates: int = 0


class HadithRetriever:
    """
    Queries the pre-built ChromaDB hadith collection.

    The collection was built on Colab with:
    - jina-embeddings-v3, task='retrieval.passage', 1024 dims
    - HNSW with cosine space, ef=200, M=32

    This retriever runs LOCALLY on CPU — just reading the persistent DB.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        logger.info(f"Connecting to ChromaDB at: {self.persist_dir}")

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_collection(
            name=self.collection_name,
        )

        count = self.collection.count()
        logger.info(
            f"Connected to collection '{self.collection_name}' "
            f"with {count:,} documents"
        )

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        grade_filter: Optional[str | list[str]] = None,
        masdar_filter: Optional[str | list[str]] = None,
    ) -> RetrievalResult:
        """
        Retrieve similar hadiths from ChromaDB.

        Args:
            query_embedding: 1024-dim query vector from Jina API.
            top_k: Number of results to return (default: RETRIEVAL_TOP_K=20).
            grade_filter: Filter by grade(s). Single string or list.
                         Values: 'sahih', 'hasan', 'daif', 'mawdu', 'unknown'
            masdar_filter: Filter by source book name (Arabic).

        Returns:
            RetrievalResult with list of RetrievedHadith objects.
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K

        # Build metadata filter
        where_filter = self._build_where_filter(grade_filter, masdar_filter)

        logger.info(
            f"Querying ChromaDB: top_k={top_k}, "
            f"grade_filter={grade_filter}, masdar_filter={masdar_filter}"
        )

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Parse results
        hadiths = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                hadith = RetrievedHadith(
                    id=doc_id,
                    text_ar=results["documents"][0][i] if results["documents"] else "",
                    distance=results["distances"][0][i] if results["distances"] else 1.0,
                    grade=metadata.get("grade", ""),
                    grade_ar=metadata.get("grade_ar", ""),
                    ruling=metadata.get("ruling", ""),
                    rawi=metadata.get("rawi", ""),
                    muhaddith=metadata.get("mohadeth", ""),        # stored as 'mohadeth' in ChromaDB
                    masdar=metadata.get("book", ""),               # stored as 'book' in ChromaDB
                    safha_raqam=str(metadata.get("numberOrPage", "")),  # stored as 'numberOrPage' in ChromaDB
                    category=metadata.get("category", ""),
                    subcategory_name=metadata.get("subcategory_name", ""),
                    hadith_tag=metadata.get("hadith_tag", ""),
                    has_explanation=str(metadata.get("hasExplanation", "False")).lower() == "true",  # stored as 'hasExplanation'
                    canonical_group_id=metadata.get("canonical_group_id", ""),
                )
                hadiths.append(hadith)

        logger.info(f"Retrieved {len(hadiths)} hadiths from ChromaDB")

        return RetrievalResult(
            query="",  # Will be set by the orchestrator
            hadiths=hadiths,
            total_candidates=len(hadiths),
        )

    def _build_where_filter(
        self,
        grade_filter: Optional[str | list[str]] = None,
        masdar_filter: Optional[str | list[str]] = None,
    ) -> Optional[dict]:
        """Build ChromaDB where filter from parameters."""
        conditions = []

        if grade_filter:
            if isinstance(grade_filter, str):
                conditions.append({"grade": {"$eq": grade_filter}})
            elif isinstance(grade_filter, list) and len(grade_filter) == 1:
                conditions.append({"grade": {"$eq": grade_filter[0]}})
            elif isinstance(grade_filter, list):
                conditions.append({"grade": {"$in": grade_filter}})

        if masdar_filter:
            # ChromaDB stores the book name in the 'book' field (not 'masdar')
            if isinstance(masdar_filter, str):
                conditions.append({"book": {"$eq": masdar_filter}})
            elif isinstance(masdar_filter, list) and len(masdar_filter) == 1:
                conditions.append({"book": {"$eq": masdar_filter[0]}})
            elif isinstance(masdar_filter, list):
                conditions.append({"book": {"$in": masdar_filter}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}


# Module-level convenience
_retriever: Optional[HadithRetriever] = None


def get_retriever() -> HadithRetriever:
    """Get or create the singleton retriever."""
    global _retriever
    if _retriever is None:
        _retriever = HadithRetriever()
    return _retriever


def retrieve(
    query_embedding: list[float],
    top_k: Optional[int] = None,
    grade_filter: Optional[str | list[str]] = None,
    masdar_filter: Optional[str] = None,
) -> RetrievalResult:
    """Convenience function for retrieval."""
    return get_retriever().retrieve(
        query_embedding=query_embedding,
        top_k=top_k,
        grade_filter=grade_filter,
        masdar_filter=masdar_filter,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    retriever = HadithRetriever()
    print(f"Collection count: {retriever.collection.count():,}")
