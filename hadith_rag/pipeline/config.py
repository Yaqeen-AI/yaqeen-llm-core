# ============================================================
# YaqeenAI — Hadith RAG Configuration
# ============================================================
# Central configuration for the Hadith RAG pipeline.
# All values can be overridden via environment variables or .env file.

import os
import re
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env from hadith_rag root
_HADITH_RAG_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_HADITH_RAG_ROOT / ".env")


class Settings:
    """Configuration for the Hadith RAG system."""

    # --- Paths ---
    HADITH_RAG_ROOT: Path = _HADITH_RAG_ROOT
    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR",
        str(_HADITH_RAG_ROOT / "chroma_db" / "hadith_chroma_db"),
    )
    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "hadiths"
    )
    DATA_DIR: Path = _HADITH_RAG_ROOT / "data"
    DATASET_STATS_PATH: Path = _HADITH_RAG_ROOT / "data" / "dataset_stats.json"

    # --- Jina API (query-time embedding) ---
    JINA_API_KEY: str = os.getenv("JINA_API_KEY", "")
    JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"
    JINA_EMBEDDING_MODEL: str = os.getenv(
        "JINA_EMBEDDING_MODEL", "jina-embeddings-v3"
    )
    JINA_EMBEDDING_DIM: int = int(os.getenv("JINA_EMBEDDING_DIM", "1024"))

    # --- Reranker ---
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
    )

    # --- Retrieval ---
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))

    # --- TF-IDF Sparse Retrieval ---
    TFIDF_INDEX_PATH: str = os.getenv(
        "TFIDF_INDEX_PATH",
        str(_HADITH_RAG_ROOT / "data" / "tfidf_index.pkl"),
    )
    TFIDF_MAX_FEATURES: int = int(os.getenv("TFIDF_MAX_FEATURES", "300000"))

    # --- Hybrid Retrieval ---
    DENSE_TOP_K: int = int(os.getenv("DENSE_TOP_K", "30"))
    SPARSE_TOP_K: int = int(os.getenv("SPARSE_TOP_K", "30"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))

    # --- Caching ---
    EMBEDDING_CACHE_SIZE: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "1000"))

    # --- Gemini --- FREE
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemma-3-27b-it")

    # --- Claude (Anthropic) --- PAID
    # CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")
    # CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-3-5-haiku-20241022")

    # --- Groq --- FREE (commented, kept for quick rollback)
    # GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    # GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # --- ChromaDB HNSW Settings (used during collection creation on Colab) ---
    HNSW_SPACE: str = "cosine"
    HNSW_CONSTRUCTION_EF: int = 200
    HNSW_M: int = 32
    HNSW_SEARCH_EF: int = 150

    # --- Grade Mappings ---
    GRADE_MAP = {
        "sahih": "صحيح",
        "hasan": "حسن",
        "daif": "ضعيف",
        "mawdu": "موضوع",
        "unknown": "غير متحقق",
    }


_ARABIC_GRADE_NORMALIZER = re.compile(r"[أإآٱ]")
_GRADE_WHITESPACE = re.compile(r"\s+")

_GRADE_ALIASES = {
    "sahih": "sahih",
    "authentic": "sahih",
    "saheeh": "sahih",
    "hasan": "hasan",
    "good": "hasan",
    "daif": "daif",
    "daeef": "daif",
    "weak": "daif",
    "mawdu": "mawdu",
    "mawdoo": "mawdu",
    "fabricated": "mawdu",
    "unknown": "unknown",
    "unverified": "unknown",
}

_MAWDU_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"موضوع",
        r"باطل",
        r"لا اصل له",
        r"لا اصل",
        r"مكذوب",
        r"كذب",
        r"مختلق",
        r"مصنوع",
    )
]

_DAIF_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"لا شيء في الحديث",
        r"ضعيف",
        r"ضعيف جدا",
        r"ضعيف جد[ااً]",
        r"لا يصح",
        r"لا يثبت",
        r"غير صحيح",
        r"ليس بصحيح",
        r"منكر",
        r"شاذ",
        r"مرسل",
        r"منقطع",
        r"معضل",
        r"متروك",
        r"واه",
        r"واهي",
        r"غير محفوظ",
        r"ليس بمحفوظ",
        r"مجهول",
        r"مدلس",
        r"معلول",
    )
]

_SAHIH_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"حسن صحيح",
        r"صحيح",
        r"ثابت",
        r"اسناده صحيح",
        r"إسناده صحيح",
        r"رجاله ثقات",
        r"على شرط الشيخين",
        r"على شرط البخاري",
        r"على شرط مسلم",
    )
]

_HASAN_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"حسن",
        r"اسناده حسن",
        r"إسناده حسن",
        r"جيد",
        r"لا باس به",
        r"لا بأس به",
        r"صالح",
        r"قوي",
    )
]


def _normalize_grade_text(text: str) -> str:
    text = _ARABIC_GRADE_NORMALIZER.sub("ا", str(text or "").strip().lower())
    return _GRADE_WHITESPACE.sub(" ", text).strip()


def _classify_grade_text(text: str) -> str:
    """Map free-form Arabic grading text to a canonical bucket."""
    normalized = _normalize_grade_text(text)
    if not normalized:
        return "unknown"

    if normalized in _GRADE_ALIASES and _GRADE_ALIASES[normalized] != "unknown":
        return _GRADE_ALIASES[normalized]

    for pattern in _MAWDU_PATTERNS:
        if pattern.search(normalized):
            return "mawdu"

    for pattern in _DAIF_PATTERNS:
        if pattern.search(normalized):
            return "daif"

    for pattern in _SAHIH_PATTERNS:
        if pattern.search(normalized):
            return "sahih"

    for pattern in _HASAN_PATTERNS:
        if pattern.search(normalized):
            return "hasan"

    return "unknown"


def resolve_display_grade_bucket(
    grade: str = "",
    grade_ar: str = "",
) -> str:
    """Resolve the short displayed grade without consulting the detailed hukm."""
    grade_norm = _normalize_grade_text(grade)
    if grade_norm in _GRADE_ALIASES and _GRADE_ALIASES[grade_norm] != "unknown":
        return _GRADE_ALIASES[grade_norm]

    grade_ar_bucket = _classify_grade_text(grade_ar)
    if grade_ar_bucket != "unknown":
        return grade_ar_bucket

    return _GRADE_ALIASES.get(grade_norm, "unknown")


def resolve_detailed_grade_bucket(ruling: str = "") -> str:
    """Resolve the detailed scholarly grading statement (hukm tafsili)."""
    return _classify_grade_text(ruling)


@dataclass(frozen=True)
class GradeAudit:
    display_bucket: str
    detailed_bucket: str
    effective_bucket: str
    is_conflicted: bool
    is_usable_for_evidence: bool
    exclusion_reason: str


def audit_grade(
    grade: str = "",
    grade_ar: str = "",
    ruling: str = "",
) -> GradeAudit:
    """
    Audit a narration's grading data.

    The short displayed grade and the detailed hukm are evaluated separately.
    If they conflict, the narration is treated as unreliable for evidence.
    """
    display_bucket = resolve_display_grade_bucket(grade, grade_ar)
    detailed_bucket = resolve_detailed_grade_bucket(ruling)

    is_conflicted = (
        display_bucket != "unknown"
        and detailed_bucket != "unknown"
        and display_bucket != detailed_bucket
    )

    if is_conflicted:
        if detailed_bucket in {"daif", "mawdu"}:
            effective_bucket = detailed_bucket
        else:
            effective_bucket = "unknown"

        display_label = Settings.GRADE_MAP.get(display_bucket, display_bucket)
        detailed_label = Settings.GRADE_MAP.get(detailed_bucket, detailed_bucket)
        exclusion_reason = (
            f"تعارض بين الدرجة المختصرة ({display_label}) "
            f"والحكم التفصيلي ({ruling or detailed_label})"
        )
        return GradeAudit(
            display_bucket=display_bucket,
            detailed_bucket=detailed_bucket,
            effective_bucket=effective_bucket,
            is_conflicted=True,
            is_usable_for_evidence=False,
            exclusion_reason=exclusion_reason,
        )

    if detailed_bucket != "unknown":
        effective_bucket = detailed_bucket
    else:
        effective_bucket = display_bucket

    if effective_bucket in {"daif", "mawdu"}:
        exclusion_reason = f"الحكم التفصيلي أو التصنيف يدل على {Settings.GRADE_MAP[effective_bucket]}"
    elif effective_bucket in {"sahih", "hasan"}:
        exclusion_reason = ""
    else:
        exclusion_reason = "الدرجة غير متحققة أو لا توجد صيغة قبول واضحة في بيانات الحكم"

    return GradeAudit(
        display_bucket=display_bucket,
        detailed_bucket=detailed_bucket,
        effective_bucket=effective_bucket,
        is_conflicted=False,
        is_usable_for_evidence=effective_bucket in {"sahih", "hasan"},
        exclusion_reason=exclusion_reason,
    )


def resolve_grade_bucket(
    grade: str = "",
    grade_ar: str = "",
    ruling: str = "",
) -> str:
    """
    Resolve a narration into a canonical grade bucket.

    The indexed `grade` field is preferred when already canonical. Otherwise,
    infer the bucket from Arabic labels and detailed rulings such as `مرسل`,
    `منكر`, or `لا أصل له`.
    """
    return audit_grade(grade, grade_ar, ruling).effective_bucket


def is_authentic_grade(
    grade: str = "",
    grade_ar: str = "",
    ruling: str = "",
) -> bool:
    """Return True only for sahih or hasan narrations."""
    audit = audit_grade(grade, grade_ar, ruling)
    return audit.is_usable_for_evidence and audit.effective_bucket in {"sahih", "hasan"}


def resolve_grade_label(
    grade: str = "",
    grade_ar: str = "",
    ruling: str = "",
) -> str:
    """
    Resolve the best Arabic label to display for a hadith ruling.

    Prefer the canonical mapped grade when it is known. If the indexed grade is
    unknown but the dataset still carries an Arabic label or detailed ruling
    such as "مرسل", surface that instead of the generic "غير محدد".
    """
    canonical_grade = resolve_grade_bucket(grade, grade_ar, ruling)
    return Settings.GRADE_MAP.get(canonical_grade, Settings.GRADE_MAP["unknown"])


settings = Settings()
