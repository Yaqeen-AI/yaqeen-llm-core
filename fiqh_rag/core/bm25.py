from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from hashlib import blake2b
from math import log, sqrt
from typing import Any

from qdrant_client.models import SparseVector

try:  # optional GPU backend for dense BM25
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is already a dependency, but keep fallback safe
    np = None


@dataclass
class BM25Okapi:
    corpus: list[list[str]]
    k1: float = 1.5
    b: float = 0.75
    dense_dim: int = 2048
    use_gpu: bool = False
    corpus_size: int = field(init=False)
    avgdl: float = field(init=False)
    doc_freqs: list[Counter[str]] = field(init=False, repr=False)
    doc_len: list[int] = field(init=False, repr=False)
    idf: dict[str, float] = field(init=False)
    term_to_idx: dict[str, int] = field(init=False)
    postings: dict[str, list[tuple[int, int]]] = field(init=False, repr=False)
    term_to_bucket: dict[str, tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.dense_dim = max(1, int(getattr(self, "dense_dim", 2048)))
        self.use_gpu = bool(getattr(self, "use_gpu", False))
        self.corpus_size = len(self.corpus)
        self.doc_freqs = []
        self.doc_len = []
        df = Counter()
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_idx, doc in enumerate(self.corpus):
            freqs = Counter(doc)
            self.doc_freqs.append(freqs)
            doc_len = len(doc)
            self.doc_len.append(doc_len)
            for term, tf in freqs.items():
                df[term] += 1
                postings[term].append((doc_idx, tf))

        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 0.0
        self.idf = {
            term: log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self.term_to_idx = {term: idx for idx, term in enumerate(df.keys())}
        self.postings = dict(postings)
        self.term_to_bucket = {term: self._bucket_info(term) for term in df.keys()}

    def _bucket_info(self, term: str) -> tuple[int, int]:
        digest = blake2b(term.encode("utf-8"), digest_size=8).digest()
        dense_dim = max(1, int(getattr(self, "dense_dim", 2048)))
        bucket = int.from_bytes(digest[:4], "little") % dense_dim
        sign = 1 if (digest[4] & 1) else -1
        return bucket, sign

    def _dense_vector(self, weights: Counter[str], doc_len: int, *, is_query: bool) -> list[float]:
        dense_dim = max(1, int(getattr(self, "dense_dim", 2048)))
        buckets = getattr(self, "term_to_bucket", {})
        dense_device = None
        if bool(getattr(self, "use_gpu", False)) and torch is not None and torch.cuda.is_available():
            dense_device = "cuda"

        if torch is not None and dense_device is not None:
            vec = torch.zeros(dense_dim, device=dense_device, dtype=torch.float32)
            for term, tf in weights.items():
                info = buckets.get(term)
                if info is None:
                    info = self._bucket_info(term)
                bucket, sign = info
                idf = self.idf.get(term, 0.0)
                if idf <= 0:
                    continue
                if is_query:
                    weight = idf * tf
                else:
                    K = self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) if self.avgdl else self.k1
                    weight = idf * (tf * (self.k1 + 1)) / (tf + K)
                vec[bucket] += float(sign * weight)
            norm = torch.linalg.vector_norm(vec)
            if float(norm) > 0:
                vec = vec / norm
            return vec.detach().cpu().tolist()

        if np is None:  # pragma: no cover - numpy is a hard dependency in practice
            raise RuntimeError("numpy is required for dense BM25 encoding")

        vec = np.zeros(dense_dim, dtype=np.float32)
        for term, tf in weights.items():
            info = buckets.get(term)
            if info is None:
                info = self._bucket_info(term)
            bucket, sign = info
            idf = self.idf.get(term, 0.0)
            if idf <= 0:
                continue
            if is_query:
                weight = idf * tf
            else:
                K = self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) if self.avgdl else self.k1
                weight = idf * (tf * (self.k1 + 1)) / (tf + K)
            vec[bucket] += float(sign * weight)
        norm = float(sqrt(float(np.dot(vec, vec))))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        if not query_tokens or not self.corpus_size:
            return scores

        qtf = Counter(query_tokens)
        for term, qcount in qtf.items():
            if term not in self.postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, tf in self.postings[term]:
                K = self.k1 * (1 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl) if self.avgdl else self.k1
                score = idf * (tf * (self.k1 + 1)) / (tf + K)
                scores[doc_idx] += score * qcount
        return scores

    def sparse_vector_for_doc(self, doc_tokens: list[str]) -> SparseVector:
        freqs = Counter(doc_tokens)
        indices: list[int] = []
        values: list[float] = []
        for term, tf in freqs.items():
            if term not in self.term_to_idx:
                continue
            idf = self.idf.get(term, 0.0)
            if idf <= 0:
                continue
            K = self.k1 * (1 - self.b + self.b * len(doc_tokens) / self.avgdl) if self.avgdl else self.k1
            weight = idf * (tf * (self.k1 + 1)) / (tf + K)
            if weight > 0:
                indices.append(self.term_to_idx[term])
                values.append(weight)
        return SparseVector(indices=indices, values=values)

    def dense_vector_for_doc(self, doc_tokens: list[str]) -> list[float]:
        return self._dense_vector(Counter(doc_tokens), len(doc_tokens), is_query=False)

    def sparse_vector_for_query(self, query_tokens: list[str]) -> SparseVector:
        freqs = Counter(query_tokens)
        indices: list[int] = []
        values: list[float] = []
        for term, qtf in freqs.items():
            if term not in self.term_to_idx:
                continue
            idf = self.idf.get(term, 0.0)
            if idf <= 0:
                continue
            indices.append(self.term_to_idx[term])
            values.append(idf * qtf)
        return SparseVector(indices=indices, values=values)

    def dense_vector_for_query(self, query_tokens: list[str]) -> list[float]:
        return self._dense_vector(Counter(query_tokens), len(query_tokens), is_query=True)

