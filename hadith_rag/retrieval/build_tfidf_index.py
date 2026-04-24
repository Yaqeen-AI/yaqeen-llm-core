import argparse
import logging
from pathlib import Path

from pipeline.config import settings
from pipeline.retrieve import HadithRetriever
from retrieval.tfidf_service import TFIDFService


logger = logging.getLogger(__name__)


def _load_corpus_from_chroma(batch_size: int) -> tuple[list[str], list[str]]:
    """
    Read the exact document IDs and texts stored in ChromaDB.

    Using Chroma as the source guarantees sparse-only hits can be mapped back
    to the dense collection by ID during hybrid retrieval.
    """
    retriever = HadithRetriever()
    collection = retriever.collection
    total = collection.count()

    logger.info("Loading %s documents from ChromaDB in batches of %s", f"{total:,}", batch_size)

    doc_ids: list[str] = []
    texts: list[str] = []

    for offset in range(0, total, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents"],
        )
        batch_ids = batch.get("ids") or []
        batch_docs = batch.get("documents") or []

        doc_ids.extend(str(doc_id) for doc_id in batch_ids)
        texts.extend(str(text or "") for text in batch_docs)

        logger.info(
            "Loaded batch %s-%s / %s",
            f"{offset + 1:,}",
            f"{min(offset + batch_size, total):,}",
            f"{total:,}",
        )

    return doc_ids, texts


def build_tfidf_index(
    output_path: Path | None = None,
    batch_size: int = 5000,
) -> Path:
    """Build and persist the local TF-IDF sparse index from ChromaDB documents."""
    output_path = output_path or Path(settings.TFIDF_INDEX_PATH)

    doc_ids, texts = _load_corpus_from_chroma(batch_size=batch_size)
    if not doc_ids:
        raise RuntimeError("No documents were loaded from ChromaDB; cannot build TF-IDF index.")

    service = TFIDFService()
    service.build_index(doc_ids=doc_ids, texts=texts)
    service.save(output_path)

    logger.info("TF-IDF build complete: %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the local TF-IDF sparse index from ChromaDB documents.")
    parser.add_argument("--batch-size", type=int, default=5000, help="Number of Chroma documents to read per batch.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.TFIDF_INDEX_PATH),
        help="Output path for the TF-IDF pickle file.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    build_tfidf_index(output_path=args.output, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
