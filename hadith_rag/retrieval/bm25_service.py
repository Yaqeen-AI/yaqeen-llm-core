# ============================================================
# YaqeenAI — BM25 Sparse Retrieval Service
# ============================================================
# Arabic-aware BM25 index for lexical retrieval.
# Persisted to disk via pickle for fast startup.
#
# BM25 is critical for Arabic Islamic text because:
# 1. Exact hadith terminology matching (specific Arabic words)
# 2. Named entity matching (narrator names, book titles)
# 3. Short queries that need exact keyword overlap

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Arabic stopwords — common particles that add noise to BM25
_ARABIC_STOPWORDS = {
    "من", "في", "على", "إلى", "عن", "أن", "ما", "لا", "هو", "هي",
    "هم", "هن", "أنا", "نحن", "أنت", "هذا", "هذه", "ذلك", "تلك",
    "الذي", "التي", "الذين", "اللاتي", "كان", "كانت", "كانوا",
    "يكون", "قد", "لم", "لن", "إن", "إذا", "كل", "بعض", "غير",
    "عند", "بين", "بعد", "قبل", "حتى", "مع", "ثم", "أو", "لكن",
    "بل", "إذ", "حين", "منذ", "لما", "كما", "إنما", "فإن", "وإن",
    "أما", "لو", "ولو", "فقد", "وقد", "ولا", "فلا", "ألا",
}

# Normalize for BM25 tokenization
_TASHKEEL = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC"
    "\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED\uFE70-\uFE7F]+"
)
_TATWEEL = re.compile("\u0640+")
_NON_ARABIC = re.compile(r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s]")
_WHITESPACE = re.compile(r"\s+")


def _tokenize_arabic(text: str) -> list[str]:
    """
    Arabic-aware tokenizer for BM25.
    
    Steps:
    1. Strip tashkeel (diacritics)
    2. Remove tatweel
    3. Remove non-Arabic characters (punctuation, numbers, etc.)
    4. Split on whitespace
    5. Filter stopwords
    6. Filter very short tokens (<2 chars)
    """
    text = _TASHKEEL.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _NON_ARABIC.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip().lower()
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in _ARABIC_STOPWORDS and len(t) >= 2]
    return tokens


class BM25Service:
    """
    BM25 sparse retrieval index for hadith texts.
    
    Features:
    - Arabic-aware tokenization with stopword removal
    - Persisted to disk via pickle for fast startup
    - k1=1.5, no custom b (BM25Okapi default b=0.75)
    """
    
    def __init__(self):
        self._index: Optional[BM25Okapi] = None
        self._doc_ids: list[str] = []
        self._is_built = False
    
    @property
    def is_built(self) -> bool:
        return self._is_built
    
    def build_index(self, doc_ids: list[str], texts: list[str]) -> None:
        """
        Build BM25 index from document texts.
        
        Args:
            doc_ids: List of document IDs (parallel to texts)
            texts: List of normalized Arabic texts
        """
        logger.info(f"Building BM25 index from {len(texts):,} documents...")
        
        self._doc_ids = doc_ids
        tokenized_corpus = [_tokenize_arabic(text) for text in texts]
        
        self._index = BM25Okapi(tokenized_corpus, k1=1.5, b=0.5)
        self._is_built = True
        
        logger.info(f"BM25 index built: {len(texts):,} documents indexed")
    
    def search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        """
        Search the BM25 index.
        
        Args:
            query: Arabic search query
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, score) tuples sorted by score descending
        """
        if not self._is_built:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        
        query_tokens = _tokenize_arabic(query)
        
        if not query_tokens:
            logger.warning("Query produced no tokens after Arabic tokenization")
            return []
        
        scores = self._index.get_scores(query_tokens)

        if top_k <= 0:
            return []

        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        results = [
            (self._doc_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0  # Only return non-zero scores
        ]
        
        return results
    
    def save(self, path: Path) -> None:
        """Save BM25 index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "index": self._index,
                "doc_ids": self._doc_ids,
            }, f)
        logger.info(f"BM25 index saved to {path}")
    
    def load(self, path: Path) -> bool:
        """
        Load BM25 index from disk.
        
        Returns:
            True if loaded successfully, False if file doesn't exist.
        """
        if not path.exists():
            logger.warning(f"BM25 cache not found: {path}")
            return False
        
        logger.info(f"Loading BM25 index from {path}...")
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self._index = data["index"]
        self._doc_ids = data["doc_ids"]
        self._is_built = True
        
        logger.info(f"BM25 index loaded: {len(self._doc_ids):,} documents")
        return True


# Module-level singleton
_bm25_service: Optional[BM25Service] = None


def get_bm25_service() -> BM25Service:
    """Get or create the singleton BM25 service."""
    global _bm25_service
    if _bm25_service is None:
        _bm25_service = BM25Service()
    return _bm25_service


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Quick test
    service = BM25Service()
    test_docs = ["فضل الصلاة على النبي", "أحكام الصيام في رمضان", "النية في العبادة"]
    test_ids = ["doc_1", "doc_2", "doc_3"]
    
    service.build_index(test_ids, test_docs)
    results = service.search("الصلاة", top_k=3)
    
    print("BM25 test results:")
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")
    print("✅ BM25 service working")
