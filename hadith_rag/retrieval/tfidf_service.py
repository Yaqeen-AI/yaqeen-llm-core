# ============================================================
# YaqeenAI — TF-IDF Sparse Retrieval Service
# ============================================================
# Replaces BM25 with a TF-IDF index that uses character n-grams
# (subword-level) in addition to word-level features.
#
# Why TF-IDF + char n-grams over BM25 for Arabic hadiths:
#   1. Arabic morphology: "الصلاة" / "صلاتك" / "يصلي" all share
#      the root صلو — char 3-5-grams catch this automatically.
#   2. Narrator names with morphological variants are handled
#      (e.g. "أبي هريرة" vs "أبو هريرة").
#   3. No OOV problem: unseen word forms still match via subword overlap.
#   4. scipy sparse matrix: 155K × ~300K matrix fits in <1 GB RAM.
#   5. Cosine similarity via sklearn is vectorized (fast NumPy).
#   6. Single .pkl file, no extra dependencies beyond sklearn + scipy.
#
# Index parameters (tuned for Arabic hadith corpus):
#   analyzer  : 'char_wb'  — character n-grams within word boundaries
#   ngram_range: (3, 5)    — 3, 4, 5-character n-grams
#   max_features: 300_000  — vocabulary cap to control RAM
#   min_df     : 3         — ignore very rare n-grams (noise)
#   max_df     : 0.85      — ignore near-universal n-grams (stopword-like)
#   sublinear_tf: True     — log(tf) smoothing, same as BM25 saturation
#   norm       : 'l2'      — cosine similarity via dot product

import logging
import pickle
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

logger = logging.getLogger(__name__)

# Arabic diacritic / tatweel stripping (same as rest of pipeline)
_TASHKEEL = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc"
    "\u06df-\u06e4\u06e7-\u06e8\u06ea-\u06ed\ufe70-\ufe7f]+"
)
_TATWEEL = re.compile("\u0640+")
_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Light normalisation before TF-IDF vectorisation."""
    text = _TASHKEEL.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


class TFIDFService:
    """
    TF-IDF sparse retrieval index for 155K Arabic hadith texts.

    Uses character n-gram features (3-5 chars) for morphology-aware
    matching, combined with cosine similarity scoring.

    Typical performance on 155K docs:
      - Build time : ~90 s on CPU
      - Index size : ~400 MB on disk
      - Query time : ~50-100 ms per query (scipy sparse dot product)
    """

    def __init__(self) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None  # scipy sparse csr_matrix (N × vocab)
        self._doc_ids: list[str] = []
        self._is_built = False

    # ------------------------------------------------------------------ #
    @property
    def is_built(self) -> bool:
        return self._is_built

    # ------------------------------------------------------------------ #
    def build_index(self, doc_ids: list[str], texts: list[str]) -> None:
        """
        Fit TF-IDF vectoriser and build the document matrix.

        Args:
            doc_ids : Parallel list of document IDs.
            texts   : Parallel list of Arabic text strings.

        Raises:
            ValueError: If the corpus is empty or doc_ids/texts lengths differ.
        """
        if not doc_ids and not texts:
            raise ValueError(
                "build_index() received an empty corpus — "
                "both doc_ids and texts are empty."
            )
        if len(doc_ids) != len(texts):
            raise ValueError(
                f"build_index() length mismatch: "
                f"doc_ids has {len(doc_ids)} items but texts has {len(texts)} items. "
                "Both lists must be the same length."
            )
        if len(texts) == 0:
            raise ValueError(
                "build_index() received an empty corpus — cannot build index."
            )

        logger.info(f"Building TF-IDF index from {len(texts):,} documents…")
        t0 = time.time()

        normalized = [_normalize(t) for t in texts]

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",  # character n-grams within word boundaries
            ngram_range=(3, 5),  # 3-, 4-, 5-char n-grams
            max_features=300_000,  # vocabulary cap — keeps RAM < 1 GB
            min_df=3,  # drop very rare n-grams
            max_df=0.85,  # drop near-universal n-grams
            sublinear_tf=True,  # log(1 + tf) — reduces dominance of high-freq terms
            norm="l2",  # L2 norm enables cosine via dot product
        )

        self._matrix = self._vectorizer.fit_transform(normalized)
        self._doc_ids = list(doc_ids)
        self._is_built = True

        elapsed = time.time() - t0
        vocab_size = len(self._vectorizer.vocabulary_)
        matrix_mb = self._matrix.data.nbytes / (1024**2)

        logger.info(
            f"TF-IDF index built in {elapsed:.1f}s | "
            f"vocab={vocab_size:,} | "
            f"matrix={self._matrix.shape} | "
            f"~{matrix_mb:.0f} MB (sparse data)"
        )

    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        """
        Search the index.

        Args:
            query  : Arabic query string (raw or pre-normalised).
            top_k  : Number of top results to return.

        Returns:
            List of (doc_id, cosine_score) sorted descending.

        Raises:
            RuntimeError: If the index has not been built or loaded yet.
            ValueError: If top_k is negative.
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_index() or load() first.")

        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")

        if top_k == 0:
            return []

        q_vec = self._vectorizer.transform([_normalize(query)])

        # linear_kernel on L2-normed vectors = cosine similarity
        scores = linear_kernel(q_vec, self._matrix).flatten()

        # Grab top_k without full sort (argpartition is O(n))
        if top_k >= len(scores):
            top_idx = np.argsort(scores)[::-1]
        else:
            # partial sort: O(n + k log k) instead of O(n log n)
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = [
            (self._doc_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0.0
        ]
        return results

    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        """Persist the index to disk as a single pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "vectorizer": self._vectorizer,
                    "matrix": self._matrix,
                    "doc_ids": self._doc_ids,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        size_mb = path.stat().st_size / (1024**2)
        logger.info(f"TF-IDF index saved → {path} ({size_mb:.0f} MB)")

    # ------------------------------------------------------------------ #
    def load(self, path: Path) -> bool:
        """
        Load a previously saved index from disk.

        Returns:
            True on success, False if file doesn't exist.
        """
        if not path.exists():
            logger.warning(f"TF-IDF index not found: {path}")
            return False

        logger.info(f"Loading TF-IDF index from {path}…")
        t0 = time.time()
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
        except Exception as e:
            logger.error(
                f"TF-IDF index at {path} is corrupt or incompatible: {e}. "
                "Deleting it — will run in dense-only mode. "
                "Run: python -m retrieval.build_tfidf_index to rebuild."
            )
            try:
                path.unlink()
            except OSError:
                pass
            return False

        self._vectorizer = data["vectorizer"]
        self._matrix = data["matrix"]
        self._doc_ids = data["doc_ids"]
        self._is_built = True

        elapsed = time.time() - t0
        logger.info(
            f"TF-IDF index loaded in {elapsed:.1f}s — {len(self._doc_ids):,} docs"
        )
        return True


# ------------------------------------------------------------------ #
# Module-level singleton
# ------------------------------------------------------------------ #
_tfidf_service: Optional[TFIDFService] = None


def get_tfidf_service() -> TFIDFService:
    """Get (or create) the module-level TF-IDF service singleton."""
    global _tfidf_service
    if _tfidf_service is None:
        _tfidf_service = TFIDFService()
    return _tfidf_service


# ------------------------------------------------------------------ #
# Quick smoke-test
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    service = TFIDFService()
    test_docs = [
        "فضل الصلاة على النبي صلى الله عليه وسلم",
        "أحكام الصيام في شهر رمضان المبارك",
        "النية في العبادة وأهميتها في الإسلام",
        "فضل قيام الليل والتهجد",
        "أحاديث في فضل الذكر والدعاء",
    ]
    test_ids = [f"doc_{i}" for i in range(len(test_docs))]

    service.build_index(test_ids, test_docs)

    for q in ["الصلاة", "رمضان الصوم", "النية العبادة"]:
        results = service.search(q, top_k=3)
        print(f"\nQuery: {q}")
        for doc_id, score in results:
            print(f"  {doc_id}: {score:.4f}")

    print("\n✅ TF-IDF service working")
