# ============================================================
# YaqeenAI — Vector DB Retrieval [LOCAL]
# ============================================================
# Queries Qdrant for similar hadiths, with ChromaDB kept as a legacy fallback.
#
# Performs cosine similarity search with optional metadata filtering
# by grade and/or source book (masdar).

import logging
from typing import Any, Iterable, Optional
from dataclasses import dataclass, field

from pipeline.config import resolve_grade_bucket, settings

# Import both backends (to support dynamic switching)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
except ImportError:
    QdrantClient = None
    Filter = FieldCondition = MatchAny = MatchValue = None

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None
    ChromaSettings = None

logger = logging.getLogger(__name__)


@dataclass
class RetrievedHadith:
    """A single hadith retrieved from vector DB (Qdrant or ChromaDB)."""

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
    explanation: str = ""
    canonical_group_id: str = ""

    @property
    def similarity_score(self) -> float:
        """Convert cosine distance to similarity score (0-1)."""
        return 1.0 - self.distance


@dataclass
class RetrievalResult:
    """Result of a vector DB retrieval query."""

    query: str
    hadiths: list[RetrievedHadith] = field(default_factory=list)
    total_candidates: int = 0


class HadithRetriever:
    """
    Queries the vector database for similar hadiths.

    Qdrant is the primary backend. ChromaDB is retained only so older local
    artifacts can still be inspected or re-exported during migration.

    The collection was built with:
    - jina-embeddings-v3, task='retrieval.passage', 1024 dims
    - Cosine similarity space
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.vector_db_type = settings.VECTOR_DB_TYPE.strip().lower()
        if self.vector_db_type not in {"qdrant", "chroma"}:
            raise ValueError(
                f"Unsupported VECTOR_DB_TYPE={settings.VECTOR_DB_TYPE!r}; "
                "expected 'qdrant' or 'chroma'."
            )
        self.collection_name = collection_name or (
            settings.QDRANT_COLLECTION_NAME
            if self.vector_db_type == "qdrant"
            else settings.CHROMA_COLLECTION_NAME
        )

        if self.vector_db_type == "qdrant":
            self._init_qdrant()
        else:
            self._init_chroma(persist_dir)

    def _init_qdrant(self):
        """Initialize Qdrant client."""
        if QdrantClient is None:
            raise RuntimeError(
                "qdrant-client is not installed. Run: pip install qdrant-client"
            )

        logger.info(f"Connecting to Qdrant at: {settings.QDRANT_URL}")
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=settings.QDRANT_TIMEOUT_SECONDS,
            check_compatibility=False,
        )

        try:
            collection_info = self.client.get_collection(self.collection_name)
            count = collection_info.points_count
            logger.info(
                f"Connected to Qdrant collection '{self.collection_name}' "
                f"with {count:,} documents"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def _init_chroma(self, persist_dir: Optional[str] = None):
        """Initialize ChromaDB client (legacy)."""
        if chromadb is None or ChromaSettings is None:
            raise RuntimeError("chromadb is not installed.")

        persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        logger.info(f"Connecting to ChromaDB at: {persist_dir}")

        self.client = chromadb.PersistentClient(
            path=persist_dir,
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

    @staticmethod
    def _payload_text(payload: dict[str, Any]) -> str:
        """Return the best matn text field available in migrated Qdrant payloads."""
        return str(
            payload.get("text_ar")
            or payload.get("text_ar_raw")
            or payload.get("text")
            or payload.get("document")
            or ""
        )

    @staticmethod
    def _payload_bool(payload: dict[str, Any], key: str) -> bool:
        value = payload.get(key, False)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes"}

    @staticmethod
    def _hadith_from_payload(
        payload: dict[str, Any],
        fallback_id: object,
        distance: float = 0.5,
    ) -> RetrievedHadith:
        raw_grade = payload.get("grade", "")
        raw_grade_ar = payload.get("grade_ar", "")
        raw_ruling = payload.get("ruling", "")

        return RetrievedHadith(
            id=str(payload.get("hadith_id") or fallback_id),
            text_ar=HadithRetriever._payload_text(payload),
            distance=distance,
            grade=resolve_grade_bucket(raw_grade, raw_grade_ar, raw_ruling),
            grade_ar=str(raw_grade_ar or ""),
            ruling=str(raw_ruling or ""),
            rawi=str(payload.get("rawi", "") or ""),
            muhaddith=str(payload.get("mohadeth", "") or payload.get("muhaddith", "") or ""),
            masdar=str(payload.get("book", "") or payload.get("masdar", "") or ""),
            safha_raqam=str(payload.get("numberOrPage", "") or payload.get("safha_raqam", "") or ""),
            category=str(payload.get("category", "") or ""),
            subcategory_name=str(payload.get("subcategory_name", "") or ""),
            hadith_tag=str(payload.get("hadith_tag", "") or ""),
            has_explanation=HadithRetriever._payload_bool(payload, "hasExplanation"),
            explanation=str(payload.get("explanation", "") or ""),
            canonical_group_id=str(payload.get("canonical_group_id", "") or ""),
        )

    def count(self) -> int:
        """Return the number of dense vector documents in the active backend."""
        if self.vector_db_type == "qdrant":
            return int(self.client.get_collection(self.collection_name).points_count or 0)
        return int(self.collection.count())

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: Optional[int] = None,
        grade_filter: Optional[str | list[str]] = None,
        masdar_filter: Optional[str | list[str]] = None,
    ) -> RetrievalResult:
        """
        Retrieve similar hadiths from the vector database.

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

        vector_db_type = getattr(self, "vector_db_type", "chroma")

        logger.info(
            f"Querying {vector_db_type}: top_k={top_k}, "
            f"grade_filter={grade_filter}, masdar_filter={masdar_filter}"
        )

        if vector_db_type == "qdrant":
            return self._retrieve_qdrant(
                query_embedding, top_k, grade_filter, masdar_filter
            )
        else:
            return self._retrieve_chroma(
                query_embedding, top_k, grade_filter, masdar_filter
            )

    def _retrieve_qdrant(
        self,
        query_embedding: list[float],
        top_k: int,
        grade_filter: Optional[str | list[str]] = None,
        masdar_filter: Optional[str | list[str]] = None,
    ) -> RetrievalResult:
        """Retrieve from Qdrant."""
        # Build filter
        query_filter = self._build_qdrant_filter(grade_filter, masdar_filter)

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

            hadiths = []
            for scored_point in response.points:
                payload = scored_point.payload or {}
                hadiths.append(
                    self._hadith_from_payload(
                        payload,
                        fallback_id=scored_point.id,
                        distance=1.0 - float(scored_point.score),
                    )
                )

            logger.info(f"Retrieved {len(hadiths)} hadiths from Qdrant")
            return RetrievalResult(
                query="",
                hadiths=hadiths,
                total_candidates=len(hadiths),
            )
        except Exception as e:
            logger.error(f"Qdrant retrieval error: {e}")
            return RetrievalResult(query="", hadiths=[], total_candidates=0)

    def _retrieve_chroma(
        self,
        query_embedding: list[float],
        top_k: int,
        grade_filter: Optional[str | list[str]] = None,
        masdar_filter: Optional[str | list[str]] = None,
    ) -> RetrievalResult:
        """Retrieve from ChromaDB (legacy)."""
        where_filter = self._build_chroma_where_filter(grade_filter, masdar_filter)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hadiths = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                raw_grade = metadata.get("grade", "")
                raw_grade_ar = metadata.get("grade_ar", "")
                raw_ruling = metadata.get("ruling", "")

                hadith = RetrievedHadith(
                    id=doc_id,
                    text_ar=results["documents"][0][i] if results["documents"] else "",
                    distance=results["distances"][0][i] if results["distances"] else 1.0,
                    grade=resolve_grade_bucket(raw_grade, raw_grade_ar, raw_ruling),
                    grade_ar=raw_grade_ar,
                    ruling=raw_ruling,
                    rawi=metadata.get("rawi", ""),
                    muhaddith=metadata.get("mohadeth", ""),
                    masdar=metadata.get("book", ""),
                    safha_raqam=str(metadata.get("numberOrPage", "")),
                    category=metadata.get("category", ""),
                    subcategory_name=metadata.get("subcategory_name", ""),
                    hadith_tag=metadata.get("hadith_tag", ""),
                    has_explanation=str(metadata.get("hasExplanation", "False")).lower()
                    == "true",
                    explanation=str(metadata.get("explanation", "") or ""),
                    canonical_group_id=metadata.get("canonical_group_id", ""),
                )
                hadiths.append(hadith)

        logger.info(f"Retrieved {len(hadiths)} hadiths from ChromaDB")
        return RetrievalResult(
            query="",
            hadiths=hadiths,
            total_candidates=len(hadiths),
        )

    @staticmethod
    def _qdrant_match_condition(key: str, value: str | list[str]) -> FieldCondition:
        if isinstance(value, list):
            values = [item for item in value if item]
            if len(values) == 1:
                return FieldCondition(key=key, match=MatchValue(value=values[0]))
            return FieldCondition(key=key, match=MatchAny(any=values))
        return FieldCondition(key=key, match=MatchValue(value=value))

    def _build_qdrant_filter(
        self,
        grade_filter: Optional[str | list[str]] = None,
        masdar_filter: Optional[str | list[str]] = None,
    ) -> Optional[Filter]:
        """Build Qdrant filter from parameters."""
        conditions = []

        if grade_filter:
            if not isinstance(grade_filter, list) or grade_filter:
                conditions.append(self._qdrant_match_condition("grade", grade_filter))

        if masdar_filter:
            if not isinstance(masdar_filter, list) or masdar_filter:
                conditions.append(self._qdrant_match_condition("book", masdar_filter))

        if not conditions:
            return None
        elif len(conditions) == 1:
            return Filter(must=[conditions[0]])
        else:
            return Filter(must=conditions)

    def _build_chroma_where_filter(
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

    def get_by_ids(self, ids: Iterable[str]) -> dict[str, RetrievedHadith]:
        """Fetch hadith metadata/text by canonical hadith ID from the active backend."""
        wanted_ids = [str(doc_id) for doc_id in ids if doc_id]
        if not wanted_ids:
            return {}

        if self.vector_db_type == "qdrant":
            return self._get_by_ids_qdrant(wanted_ids)
        return self._get_by_ids_chroma(wanted_ids)

    def _get_by_ids_qdrant(self, ids: list[str]) -> dict[str, RetrievedHadith]:
        found: dict[str, RetrievedHadith] = {}
        batch_size = 256

        for offset in range(0, len(ids), batch_size):
            batch_ids = ids[offset : offset + batch_size]
            missing_ids = list(batch_ids)

            try:
                points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=batch_ids,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    hadith = self._hadith_from_payload(
                        point.payload or {},
                        fallback_id=point.id,
                    )
                    found[hadith.id] = hadith

                missing_ids = [doc_id for doc_id in batch_ids if doc_id not in found]
                if not missing_ids:
                    continue
            except Exception as exc:
                logger.debug(
                    "Direct Qdrant ID fetch failed; falling back to payload scroll: %s",
                    exc,
                )

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="hadith_id",
                        match=MatchAny(any=missing_ids),
                    )
                ]
            )
            next_page = None
            while True:
                points, next_page = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=min(len(missing_ids), 256),
                    offset=next_page,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in points:
                    hadith = self._hadith_from_payload(
                        point.payload or {},
                        fallback_id=point.id,
                    )
                    found[hadith.id] = hadith

                if next_page is None:
                    break

        return found

    def _get_by_ids_chroma(self, ids: list[str]) -> dict[str, RetrievedHadith]:
        fetched = self.collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )

        found: dict[str, RetrievedHadith] = {}
        if not fetched.get("ids"):
            return found

        for i, doc_id in enumerate(fetched["ids"]):
            metadata = fetched["metadatas"][i] if fetched.get("metadatas") else {}
            raw_grade = metadata.get("grade", "")
            raw_grade_ar = metadata.get("grade_ar", "")
            raw_ruling = metadata.get("ruling", "")
            found[str(doc_id)] = RetrievedHadith(
                id=str(doc_id),
                text_ar=fetched["documents"][i] if fetched.get("documents") else "",
                distance=0.5,
                grade=resolve_grade_bucket(raw_grade, raw_grade_ar, raw_ruling),
                grade_ar=raw_grade_ar,
                ruling=raw_ruling,
                rawi=metadata.get("rawi", ""),
                muhaddith=metadata.get("mohadeth", ""),
                masdar=metadata.get("book", ""),
                safha_raqam=str(metadata.get("numberOrPage", "")),
                category=metadata.get("category", ""),
                subcategory_name=metadata.get("subcategory_name", ""),
                hadith_tag=metadata.get("hadith_tag", ""),
                has_explanation=str(metadata.get("hasExplanation", "False")).lower()
                == "true",
                explanation=str(metadata.get("explanation", "") or ""),
                canonical_group_id=metadata.get("canonical_group_id", ""),
            )
        return found

    def iter_corpus(self, batch_size: int = 5000) -> Iterable[tuple[list[str], list[str]]]:
        """Yield canonical IDs and texts from the active vector backend."""
        if self.vector_db_type == "qdrant":
            next_page = None
            while True:
                points, next_page = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=next_page,
                    with_payload=True,
                    with_vectors=False,
                )
                if not points:
                    break

                ids: list[str] = []
                texts: list[str] = []
                for point in points:
                    payload = point.payload or {}
                    ids.append(str(payload.get("hadith_id") or point.id))
                    texts.append(self._payload_text(payload))
                yield ids, texts

                if next_page is None:
                    break
            return

        total = self.collection.count()
        for offset in range(0, total, batch_size):
            batch = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents"],
            )
            yield (
                [str(doc_id) for doc_id in (batch.get("ids") or [])],
                [str(text or "") for text in (batch.get("documents") or [])],
            )


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
    print(f"Collection count: {retriever.count():,}")
