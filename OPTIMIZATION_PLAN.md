# Quranic RAG Colab — Optimization Implementation Plan

> **Goal**: Fix performance issues on T4 GPU, solve semantic dilution in large themes, keep English theme names as metadata-only, add weighted RRF.

## ✅ Done: Database Cleanup

`clean_ayah_themes.py` created `ayah-themes-clean.db`:
- 2,098 → 1,049 rows (removed exact duplicates)
- Use `ayah-themes-clean.db` in the notebook going forward

---

## Changes Overview (by Notebook Cell)

| Section | Cell ID | What Changes |
|---------|---------|-------------|
| §3 Config | `config` | New `ChunkingConfig` tiers, 512d Matryoshka, weighted RRF params |
| §7 Chunker | `chunker` | Adaptive chunking, English removed from embedding text |
| §8 Embedder | `embedder` | SDPA attention, 512d truncation, dynamic batch sizing |
| §9 Qdrant | `qdrant` | Arabic stop words in BM25, weighted RRF |
| §10 Pipeline | `pipeline` | Use clean DB path |

---

## §3 — Configuration Changes

### Replace `ChunkingConfig` with adaptive tiers:

```python
@dataclass(frozen=True)
class ChunkingConfig:
    """Adaptive chunking configuration — tiered by theme size."""
    # Tier boundaries (ayah count)
    ATOMIC_MAX: int = 3           # 1-3 ayahs → single chunk, no parent
    LIGHT_SPLIT_MAX: int = 8      # 4-8 ayahs → window=3, overlap=1
    MEDIUM_SPLIT_MAX: int = 15    # 9-15 ayahs → window=4, overlap=1
    # Above 15 → window=5, overlap=2

    # Parent chunk behavior
    PARENT_MAX_AYAHS: int = 8     # Only create parents for themes ≤ 8 ayahs
    SUMMARY_PARENT_ABOVE: int = 8 # Above this: verse-only summary parent (no tafsir)

    MAX_CHUNK_TOKENS: int = 6000
```

### Replace `EmbeddingConfig`:

```python
@dataclass(frozen=True)
class EmbeddingConfig:
    """Jina Embeddings v3 — optimized for T4 GPU."""
    MODEL_NAME: str = "jinaai/jina-embeddings-v3"
    DIMENSIONS: int = 512          # Matryoshka: 512 instead of 1024
    MAX_CONTEXT_WINDOW: int = 512
    RETRIEVAL_QUERY_PREFIX: str = ""    # Remove English prefixes
    RETRIEVAL_PASSAGE_PREFIX: str = ""  # Remove English prefixes
    USE_FP16: bool = True
    EMBEDDING_BATCH_SIZE: int = 8
```

### Add RRF weights to `QdrantConfig`:

```python
@dataclass(frozen=True)
class QdrantConfig:
    """Qdrant vector database configuration."""
    HOST: str = "localhost"
    PORT: int = 6333
    GRPC_PORT: int = 6334
    COLLECTION_NAME: str = "quranic_themes"
    DENSE_VECTOR_NAME: str = "dense"
    SPARSE_VECTOR_NAME: str = "sparse"
    DISTANCE_METRIC: str = "Cosine"
    DEFAULT_TOP_K: int = 10
    RRF_K: int = 60
    RRF_DENSE_WEIGHT: float = 0.6   # NEW: semantic gets more weight
    RRF_SPARSE_WEIGHT: float = 0.4  # NEW: keyword match
    BM25_B: float = 0.75
    BM25_K1: float = 1.2
```

---

## §7 — Adaptive Thematic Chunker

### Key principle: English theme names are METADATA ONLY

The `_assemble_chunk_text` method must NOT put theme_name into `text_for_embedding`.

### Replace the entire chunker cell with:

```python
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class ChunkingStats:
    total_themes: int = 0
    themes_atomic: int = 0       # 1-3 ayahs, single chunk
    themes_light: int = 0        # 4-8 ayahs
    themes_medium: int = 0       # 9-15 ayahs
    themes_heavy: int = 0        # 16+ ayahs
    total_child_chunks: int = 0
    total_parent_chunks: int = 0
    total_summary_chunks: int = 0


class AdaptiveThematicChunker:
    """
    Adaptive chunker that adjusts window size based on theme size.
    
    Solves semantic dilution by:
    1. Not creating parent chunks for large themes (they'd be truncated)
    2. Using verse-only summary chunks instead of full parents for medium themes
    3. Scaling window/overlap to theme size
    4. Keeping English theme names OUT of embedding text (metadata only)
    """

    def __init__(
        self,
        chunking_config: Optional[ChunkingConfig] = None,
        normalizer: Optional[TextNormalizer] = None,
    ):
        self.config = chunking_config or config.chunking
        self.normalizer = normalizer or DEFAULT_NORMALIZER
        self.stats = ChunkingStats()

    def _get_tier_params(self, total_ayahs: int) -> dict:
        """Return window_size, overlap, and parent strategy for a given theme size."""
        if total_ayahs <= self.config.ATOMIC_MAX:
            return {"window": total_ayahs, "overlap": 0, "parent": "none", "tier": "atomic"}
        elif total_ayahs <= self.config.LIGHT_SPLIT_MAX:
            return {"window": 3, "overlap": 1, "parent": "full", "tier": "light"}
        elif total_ayahs <= self.config.MEDIUM_SPLIT_MAX:
            return {"window": 4, "overlap": 1, "parent": "summary", "tier": "medium"}
        else:
            return {"window": 5, "overlap": 2, "parent": "summary", "tier": "heavy"}

    def _calculate_num_children(self, total_ayahs: int, window: int, overlap: int) -> int:
        stride = window - overlap
        if total_ayahs <= window:
            return 1
        return max(1, math.ceil((total_ayahs - window) / stride) + 1)

    def _get_window_boundaries(self, total_ayahs: int, chunk_index: int, window: int, overlap: int) -> tuple:
        stride = window - overlap
        start = chunk_index * stride
        end = min(start + window - 1, total_ayahs - 1)
        start = min(start, total_ayahs - 1)
        return (start, end)

    def _assemble_embedding_text(self, ayahs: List, include_tafsir: bool = True) -> str:
        """
        Build text_for_embedding: PURE ARABIC ONLY.
        NO English theme names, NO English prefixes.
        """
        parts = []
        for ayah in ayahs:
            # Normalized Arabic verse text
            parts.append(ayah.text_normalized)
            if include_tafsir:
                tafsir_combined = ayah.get_combined_tafsir()
                if tafsir_combined:
                    norm_tafsir, _ = self.normalizer.normalize(tafsir_combined)
                    parts.append(norm_tafsir)
        return " ".join(parts)

    def _assemble_display_text(self, ayahs: List, theme_name: str, include_tafsir: bool = True) -> tuple:
        """
        Build text_original and text_with_tafsir for DISPLAY purposes.
        Theme name appears here (wrapped in Arabic label) for LLM context.
        """
        structured_parts = [f"[الموضوع: {theme_name}]"]
        original_parts = []

        for ayah in ayahs:
            ayah_header = f"\u064f{ayah.ayah_number}\u064e"
            structured_parts.append(f"{ayah_header} {ayah.text_uthmani}")
            original_parts.append(f"{ayah_header} {ayah.text_uthmani}")

            if include_tafsir:
                tafsir_combined = ayah.get_combined_tafsir()
                if tafsir_combined:
                    structured_parts.append(tafsir_combined)

        text_with_tafsir = "\n".join(structured_parts)
        text_original = " ".join(original_parts)
        return (text_original, text_with_tafsir)

    def _create_summary_chunk(self, theme: Theme) -> Chunk:
        """
        Lightweight summary chunk: verses only (NO tafsir) to stay within 512 tokens.
        Used instead of a full parent for medium/large themes.
        """
        text_for_embedding = self._assemble_embedding_text(theme.ayahs, include_tafsir=False)
        text_original, text_with_tafsir = self._assemble_display_text(
            theme.ayahs, theme.theme_name, include_tafsir=True
        )
        return Chunk(
            chunk_id=uuid4(),
            parent_theme_id=theme.theme_id,
            theme_name=theme.theme_name,
            surah_number=theme.surah_number,
            ayah_from=theme.ayah_from,
            ayah_to=theme.ayah_to,
            keywords=theme.keywords,
            text_for_embedding=text_for_embedding,
            text_original=text_original,
            text_with_tafsir=text_with_tafsir,
            is_parent=True,
            child_index=0,
            total_children=1,
        )

    def _create_full_parent_chunk(self, theme: Theme) -> Chunk:
        """Full parent with tafsir — only used for small themes (≤8 ayahs)."""
        text_for_embedding = self._assemble_embedding_text(theme.ayahs, include_tafsir=True)
        text_original, text_with_tafsir = self._assemble_display_text(
            theme.ayahs, theme.theme_name, include_tafsir=True
        )
        return Chunk(
            chunk_id=uuid4(),
            parent_theme_id=theme.theme_id,
            theme_name=theme.theme_name,
            surah_number=theme.surah_number,
            ayah_from=theme.ayah_from,
            ayah_to=theme.ayah_to,
            keywords=theme.keywords,
            text_for_embedding=text_for_embedding,
            text_original=text_original,
            text_with_tafsir=text_with_tafsir,
            is_parent=True,
            child_index=0,
            total_children=1,
        )

    def _create_child_chunks(self, theme: Theme, window: int, overlap: int) -> List[Chunk]:
        total_ayahs = len(theme.ayahs)
        num_children = self._calculate_num_children(total_ayahs, window, overlap)
        chunks = []

        for idx in range(num_children):
            start_idx, end_idx = self._get_window_boundaries(total_ayahs, idx, window, overlap)
            window_ayahs = theme.ayahs[start_idx : end_idx + 1]
            ayah_from = theme.ayah_from + start_idx
            ayah_to = theme.ayah_from + end_idx

            text_for_embedding = self._assemble_embedding_text(window_ayahs, include_tafsir=True)
            text_original, text_with_tafsir = self._assemble_display_text(
                window_ayahs, theme.theme_name, include_tafsir=True
            )

            chunk = Chunk(
                chunk_id=uuid4(),
                parent_theme_id=theme.theme_id,
                theme_name=theme.theme_name,
                surah_number=theme.surah_number,
                ayah_from=ayah_from,
                ayah_to=ayah_to,
                keywords=theme.keywords,
                text_for_embedding=text_for_embedding,
                text_original=text_original,
                text_with_tafsir=text_with_tafsir,
                is_parent=False,
                child_index=idx,
                total_children=num_children,
            )
            chunks.append(chunk)
        return chunks

    def chunk_theme(self, theme: Theme) -> List[Chunk]:
        if not theme.ayahs:
            return []

        total_ayahs = len(theme.ayahs)
        params = self._get_tier_params(total_ayahs)
        chunks = []

        # === ATOMIC: single chunk, no parent needed ===
        if params["tier"] == "atomic":
            text_for_embedding = self._assemble_embedding_text(theme.ayahs, include_tafsir=True)
            text_original, text_with_tafsir = self._assemble_display_text(
                theme.ayahs, theme.theme_name, include_tafsir=True
            )
            chunk = Chunk(
                chunk_id=uuid4(),
                parent_theme_id=theme.theme_id,
                theme_name=theme.theme_name,
                surah_number=theme.surah_number,
                ayah_from=theme.ayah_from,
                ayah_to=theme.ayah_to,
                keywords=theme.keywords,
                text_for_embedding=text_for_embedding,
                text_original=text_original,
                text_with_tafsir=text_with_tafsir,
                is_parent=False,
                child_index=0,
                total_children=1,
            )
            chunks.append(chunk)
            self.stats.themes_atomic += 1

        else:
            # === Create children ===
            children = self._create_child_chunks(theme, params["window"], params["overlap"])
            chunks.extend(children)
            self.stats.total_child_chunks += len(children)

            # === Parent strategy ===
            if params["parent"] == "full":
                parent = self._create_full_parent_chunk(theme)
                parent.total_children = len(children)
                chunks.insert(0, parent)
                self.stats.total_parent_chunks += 1
            elif params["parent"] == "summary":
                summary = self._create_summary_chunk(theme)
                summary.total_children = len(children)
                chunks.insert(0, summary)
                self.stats.total_summary_chunks += 1

            # Track tier
            if params["tier"] == "light":
                self.stats.themes_light += 1
            elif params["tier"] == "medium":
                self.stats.themes_medium += 1
            else:
                self.stats.themes_heavy += 1

        self.stats.total_themes += 1
        return chunks

    def get_stats(self) -> ChunkingStats:
        return self.stats

    def reset_stats(self):
        self.stats = ChunkingStats()


print("Adaptive Chunker defined!")
```

---

## §8 — Jina Embedder Optimization

### Three key changes:
1. **SDPA attention** instead of `eager`
2. **Matryoshka 512d** truncation in `_mean_pooling`
3. **Dynamic batch sizing** based on text length

### Replace `JinaEmbedder` class:

```python
class JinaEmbedder:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        use_fp16: bool = True,
    ):
        self.config = embedding_config or config.embedding
        self.model_name = model_name or self.config.MODEL_NAME
        self.use_fp16 = use_fp16
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_dim = self.config.DIMENSIONS  # 512 Matryoshka

        print(f"Loading Jina Embeddings v3 on {self.device} (target_dim={self.target_dim})...")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # SDPA attention: 2-3x faster than eager, less memory
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"  # Changed from "eager"
        ).to(self.device)

        self.model.eval()
        
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"Model loaded! GPU memory used: {mem_gb:.2f} GB")
        print("Jina Embeddings v3 loaded successfully!")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # Matryoshka truncation: take first target_dim dimensions
        return pooled[:, :self.target_dim]

    def _get_dynamic_batch_size(self, texts: List[str], max_batch: int = 8) -> int:
        """Reduce batch size for longer texts to prevent OOM."""
        if not texts:
            return max_batch
        avg_len = sum(len(t) for t in texts) / len(texts)
        if avg_len > 2000:
            return max(1, max_batch // 4)
        elif avg_len > 1000:
            return max(2, max_batch // 2)
        return max_batch

    @torch.no_grad()
    def embed_documents(self, texts: List[str], batch_size: int = 8, show_progress: bool = True) -> List[List[float]]:
        MAX_SAFE_LEN = self.config.MAX_CONTEXT_WINDOW  # 512

        # Dynamic batch sizing
        actual_batch = self._get_dynamic_batch_size(texts, batch_size)
        if actual_batch != batch_size and show_progress:
            print(f"Dynamic batch: {batch_size} → {actual_batch} (avg text len: {sum(len(t) for t in texts)//len(texts)})")

        all_embeddings = []

        for i in range(0, len(texts), actual_batch):
            batch = texts[i : i + actual_batch]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_SAFE_LEN,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**encoded)
            embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.float().cpu().tolist())

            del encoded, outputs, embeddings
            if i % (actual_batch * 5) == 0:
                torch.cuda.empty_cache()

            if show_progress:
                done = min(i + actual_batch, len(texts))
                print(f"\rEmbedded {done}/{len(texts)} chunks", end="", flush=True)

        if show_progress:
            print()
        return all_embeddings

    @torch.no_grad()
    def embed_query(self, query: str) -> List[float]:
        encoded = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.config.MAX_CONTEXT_WINDOW,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**encoded)
        embedding = self._mean_pooling(outputs, encoded["attention_mask"])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        result = embedding.float().cpu().tolist()[0]
        del encoded, outputs, embedding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    @property
    def embedding_dim(self) -> int:
        return self.target_dim  # Returns 512, not 1024
```

---

## §9 — BM25 + Weighted RRF

### Add Arabic stop words to `ArabicBM25Vectorizer`:

```python
# Add this constant before the class:
ARABIC_STOP_WORDS = {
    'في', 'من', 'على', 'الى', 'عن', 'هو', 'هي', 'هم', 'ان', 'كان',
    'ما', 'لا', 'الا', 'هذا', 'هذه', 'ذلك', 'تلك', 'الذي', 'التي',
    'قد', 'قال', 'بل', 'ثم', 'او', 'لم', 'لن', 'حتى', 'اذا',
    'اذ', 'عند', 'بعد', 'قبل', 'كل', 'بين', 'فيها', 'فيه', 'منها',
    'منه', 'عليه', 'عليها', 'اليه', 'اليها', 'به', 'بها', 'له', 'لها',
}
```

### Update `_tokenize` method:

```python
def _tokenize(self, text: str) -> List[str]:
    tokens = self.normalizer.tokenize_arabic(text)
    return [t for t in tokens if t not in ARABIC_STOP_WORDS]
```

### Update `_reciprocal_rank_fusion` for weighted RRF:

```python
def _reciprocal_rank_fusion(
    self,
    dense_results: List[Tuple[int, float]],
    sparse_results: List[Tuple[int, float]],
    k: int = 60,
) -> List[Tuple[int, float, float, float]]:
    dense_weight = self.config.RRF_DENSE_WEIGHT   # 0.6
    sparse_weight = self.config.RRF_SPARSE_WEIGHT  # 0.4

    dense_ranks = {id_: rank for rank, (id_, _) in enumerate(dense_results, 1)}
    dense_scores = {id_: score for id_, score in dense_results}
    sparse_ranks = {id_: rank for rank, (id_, _) in enumerate(sparse_results, 1)}
    sparse_scores = {id_: score for id_, score in sparse_results}

    all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
    max_rank = len(all_ids) + 1

    rrf_results = []
    for id_ in all_ids:
        d_rank = dense_ranks.get(id_, max_rank)
        s_rank = sparse_ranks.get(id_, max_rank)
        rrf_score = dense_weight * (1 / (k + d_rank)) + sparse_weight * (1 / (k + s_rank))
        rrf_results.append((id_, rrf_score, dense_scores.get(id_, 0.0), sparse_scores.get(id_, 0.0)))

    rrf_results.sort(key=lambda x: x[1], reverse=True)
    return rrf_results
```

---

## §10 — Pipeline Updates

### In `ThemeLoader.load_all_themes()`:

Change SQL to use DISTINCT (as safety net even with clean DB):
```python
cursor.execute("""
    SELECT DISTINCT theme, surah_number, ayah_from, ayah_to, keywords, total_ayahs
    FROM themes ORDER BY surah_number, ayah_from
""")
```

### In `QuranicRAGPipeline.__init__()`:

1. Change DB path to use clean DB:
```python
self.theme_loader = ThemeLoader(Path("/content/ayah-themes-clean.db"))
```

2. Use the new adaptive chunker:
```python
self.chunker = AdaptiveThematicChunker()
```

---

## Summary of All Changes

| What | Before | After | Impact |
|------|--------|-------|--------|
| DB rows | 2,098 (duped) | 1,049 (clean) | **50% less API calls + embeddings** |
| Theme name in embeddings | Mixed English+Arabic | Arabic only (metadata) | **Cleaner semantic space** |
| Chunking | Fixed window=3, all parents | Adaptive 3-5 window, smart parents | **No semantic dilution** |
| Attention | `eager` | `sdpa` | **2-3x faster inference** |
| Embedding dims | 1024 | 512 (Matryoshka) | **50% less memory + storage** |
| Batch sizing | Fixed 8 | Dynamic 2-8 | **Prevents OOM on long texts** |
| English prefix | `"Represent this..."` | None | **No English in Arabic pipeline** |
| BM25 | No stop words | Arabic stop words | **Better keyword matching** |
| RRF | Equal weights | 0.6 dense / 0.4 sparse | **Tunable fusion** |
| Parents (20+ ayah themes) | Full parent (truncated) | Verse-only summary | **Useful anchors instead of garbage** |
