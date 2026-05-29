# ============================================================
# YaqeenAI — Query Preprocessor (Production Enhanced)
# ============================================================
# Normalizes, classifies, and enriches user queries before
# sending them to the retrieval pipeline.
#
# Handles:
# 1. Language detection (Arabic vs non-Arabic)
# 2. Arabic normalization (tashkeel, tatweel, whitespace)
# 3. Short query expansion (adds relevant context terms)
# 4. Query type classification (hadith lookup, ruling, topic, narrator, metadata, greeting, out-of-scope)
# 5. Metadata-aware query routing
# 6. Transliterated Arabic handling
# 7. Islamic greeting detection (no retrieval needed)
# 8. Out-of-scope query detection
# 9. Alef normalization for improved recall

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.query_expander import ExpandedQuery

logger = logging.getLogger(__name__)

# ============================================================
# Query Classification
# ============================================================

class QueryType(Enum):
    """Categories of hadith-related queries."""
    HADITH_LOOKUP = "hadith_lookup"    # Looking for a specific hadith by text
    EXPLAIN_HADITH = "explain_hadith"  # Requesting explanation/clarification of a specific hadith
    RULING = "ruling"                   # Asking about authenticity/ruling
    TOPIC = "topic"                     # Topical search (e.g., "prayer", "fasting")
    NARRATOR = "narrator"               # Searching by narrator name
    METADATA = "metadata"               # Asking about metadata (grade, book, number, chain)
    DATASET_STATS = "dataset_stats"     # Questions about dataset counts/statistics
    GREETING = "greeting"               # Islamic greeting, no retrieval needed
    OUT_OF_SCOPE = "out_of_scope"       # Not related to Islam/hadith
    GENERAL = "general"                 # General question about hadiths


@dataclass
class ProcessedQuery:
    """Result of query preprocessing."""
    original: str
    normalized: str           # Cleaned/normalized query text
    query_type: QueryType
    is_arabic: bool
    is_short: bool            # Less than 3 meaningful words
    expanded: str             # Expanded query (if short query expansion applied)
    dense_query: str          # Query to send to dense retrieval (Jina)
    sparse_query: str         # Query to send to sparse retrieval (TF-IDF)
    metadata_fields: list[str] = field(default_factory=list)   # Which metadata fields the user is asking about
    skip_retrieval: bool = False       # True for greetings/out-of-scope
    direct_response: str = ""          # Pre-built response for greetings/out-of-scope
    # ── Query Expansion fields ─────────────────────────────
    multi_queries: list[str] = field(default_factory=list)   # All query variants for multi-query retrieval
    expansion_tokens: list[str] = field(default_factory=list) # Additional tokens added by expander
    # ── Book/source filter extracted from query text ────────
    extracted_masdar: str | list[str] = ""  # Book name(s) detected in query; list when multiple source aliases exist
    excluded_masdar: list[str] = field(default_factory=list)  # Books to exclude (e.g. "ولم تذكر في صحيح مسلم")


# ============================================================
# Regex Patterns
# ============================================================

_TASHKEEL = re.compile(
    "[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC"
    "\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED\uFE70-\uFE7F]+"
)
_TATWEEL = re.compile("\u0640+")
_WHITESPACE = re.compile(r"\s+")
_ARABIC_CHARS = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")

# Alef normalization: أ إ آ ا → ا (for query matching only, not storage)
_ALEF_VARIANTS = re.compile(r"[أإآٱ]")

# ============================================================
# Greeting Detection
# ============================================================

_GREETING_PATTERNS_AR = [
    re.compile(r"^(السلام\s*عليكم|سلام\s*عليكم|السلام\s*عليكم\s*ورحمة\s*الله)"),
    re.compile(r"^(مرحبا|مرحبًا|أهلا|أهلاً|هلا|اهلا)$"),
    re.compile(r"^(جزاك\s*الله\s*خير|بارك\s*الله\s*فيك|شكرا|شكراً)$"),
    re.compile(r"^(صباح\s*الخير|مساء\s*الخير)$"),
]

_GREETING_PATTERNS_EN = [
    re.compile(r"^(assalamu?\s*alaikum|salam\s*alaikum|salaam\s*alaikum)", re.IGNORECASE),
    re.compile(r"^(hello|hi|hey|good\s*morning|good\s*evening|thank\s*you|thanks)$", re.IGNORECASE),
    re.compile(r"^(barakallah|jazakallah)", re.IGNORECASE),
]

_GREETING_RESPONSE = "وعليكم السلام ورحمة الله وبركاته! أهلاً بك. كيف يمكنني مساعدتك في البحث عن الأحاديث النبوية الشريفة؟"

# ============================================================
# Out-of-Scope Detection
# ============================================================

_OUT_OF_SCOPE_PATTERNS = [
    # Programming / Technology
    re.compile(r"\b(python|javascript|java\b|coding|programming|code|debug|error|bug|api|sql|html|css|react|node|docker)\b", re.IGNORECASE),
    # Entertainment
    re.compile(r"\b(movie|film|game|sport|football|soccer|basketball|music|song|netflix|youtube)\b", re.IGNORECASE),
    # General non-Islamic
    re.compile(r"\b(weather|stock market|recipe|cooking|restaurant|hotel|flight|laptop)\b", re.IGNORECASE),
    # Math / Science (non-Islamic context)
    re.compile(r"^(what\s+is\s+\d|calculate|solve\s|math\b|equation)", re.IGNORECASE),
]

_OUT_OF_SCOPE_RESPONSE = "عذراً، هذا النظام متخصص في الأحاديث النبوية الشريفة فقط. يمكنني مساعدتك في البحث عن الأحاديث، والتحقق من صحتها، ومعرفة رواتها ومصادرها. كيف يمكنني مساعدتك؟"

# ============================================================
# Dataset Statistics Detection
# ============================================================

_STATS_PATTERNS = [
    # Arabic — "how many hadiths" patterns
    re.compile(r"(كم\s*(عدد)?\s*(حديث|الأحاديث|احاديث))", re.IGNORECASE),
    re.compile(r"(عدد\s*(الأحاديث|الاحاديث|احاديث))", re.IGNORECASE),
    # Arabic — "how many sahih/weak/..." patterns
    re.compile(r"(كم\s*(عدد)?\s*(حديث|الأحاديث)?\s*(صحيح|حسن|ضعيف|موضوع))", re.IGNORECASE),
    re.compile(r"(كم\s*(حديث|الأحاديث)\s*(الصحيحة|الحسنة|الضعيفة|الموضوعة))", re.IGNORECASE),
    re.compile(r"(عدد\s*(الأحاديث|الاحاديث)\s*(الصحيحة|الحسنة|الضعيفة|الموضوعة))", re.IGNORECASE),
    # Arabic — "how many narrators/sources"
    re.compile(r"(كم\s*(عدد)?\s*(الرواة|راوي|رواة|المصادر|مصدر|الكتب|كتاب|المحدث|محدث))", re.IGNORECASE),
    re.compile(r"(عدد\s*(الرواة|المصادر|الكتب|المحدثين))", re.IGNORECASE),
    # Arabic — "statistics / numbers / dataset"
    re.compile(r"(إحصائيات|احصائيات|إحصاء|أرقام\s*(البيانات|القاعدة|الأحاديث))", re.IGNORECASE),
    # English patterns
    re.compile(r"(how\s+many\s+(hadith|hadiths|narrat|source|book))", re.IGNORECASE),
    re.compile(r"(total\s+(number|count)\s+of\s+(hadith|hadiths))", re.IGNORECASE),
    re.compile(r"(dataset\s+stats|dataset\s+statistics|corpus\s+size)", re.IGNORECASE),
    re.compile(r"(number\s+of\s+(sahih|hasan|daif|weak|authentic|fabricated))", re.IGNORECASE),
    # Arabic — top narrators / sources
    re.compile(r"(أكثر\s*(الرواة|المصادر|المحدثين)\s*(رواية)?)", re.IGNORECASE),
    re.compile(r"(من\s+أكثر\s*(الرواة|الصحابة)\s*(رواية)?)", re.IGNORECASE),
]


def _load_dataset_stats() -> dict | None:
    """Load pre-computed dataset statistics from JSON file."""
    import json
    from pipeline.config import settings
    stats_path = settings.DATA_DIR / "dataset_stats.json"
    if not stats_path.exists():
        logger.warning(f"dataset_stats.json not found at {stats_path}")
        return None
    try:
        with open(stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset_stats.json: {e}")
        return None


def _build_stats_response(query: str) -> str:
    """
    Build a formatted Arabic response for dataset statistics questions.
    Loads stats from dataset_stats.json and formats based on what was asked.
    """
    stats = _load_dataset_stats()
    if not stats:
        return "عذراً، لم أتمكن من تحميل إحصائيات قاعدة البيانات. تأكد من وجود ملف dataset_stats.json في مجلد data."

    total = stats.get("total_hadiths", 0)
    by_grade = stats.get("by_grade", {})
    by_grade_ar = stats.get("by_grade_ar", {})
    top_narrators = stats.get("top_narrators", {})
    top_sources = stats.get("top_sources", {})
    unique_narrators = stats.get("unique_narrators", 0)
    unique_sources = stats.get("unique_sources", 0)
    unique_muhadditheen = stats.get("unique_muhadditheen", 0)

    query_lower = query.lower()
    normalized_q = normalize_query(query)

    # Check if asking about a specific grade
    grade_keywords = {
        "صحيح": ("sahih", "صحيح"),
        "حسن": ("hasan", "حسن"),
        "ضعيف": ("daif", "ضعيف"),
        "موضوع": ("mawdu", "موضوع"),
        "sahih": ("sahih", "صحيح"),
        "authentic": ("sahih", "صحيح"),
        "hasan": ("hasan", "حسن"),
        "good": ("hasan", "حسن"),
        "daif": ("daif", "ضعيف"),
        "weak": ("daif", "ضعيف"),
        "mawdu": ("mawdu", "موضوع"),
        "fabricated": ("mawdu", "موضوع"),
    }

    for keyword, (grade_en, grade_ar_label) in grade_keywords.items():
        if keyword in normalized_q or keyword in query_lower:
            count = by_grade.get(grade_en, 0)
            return (
                f"📊 عدد الأحاديث ذات درجة «{grade_ar_label}» في قاعدة بياناتنا: **{count:,}** حديث\n"
                f"من إجمالي **{total:,}** حديث."
            )

    # Check if asking about narrators
    narrator_keywords = ["رواة", "الرواة", "راوي", "narrat", "الصحابة"]
    if any(k in normalized_q or k in query_lower for k in narrator_keywords):
        response = f"📊 إحصائيات الرواة في قاعدة بياناتنا:\n\n"
        response += f"• **عدد الرواة المختلفين:** {unique_narrators:,} راوٍ\n\n"
        response += f"**أكثر الرواة رواية:**\n"
        for i, (name, count) in enumerate(list(top_narrators.items())[:10], 1):
            response += f"  {i}. {name} — {count:,} حديث\n"
        return response

    # Check if asking about sources / books
    source_keywords = ["مصادر", "المصادر", "كتب", "الكتب", "مصدر", "source", "book"]
    if any(k in normalized_q or k in query_lower for k in source_keywords):
        response = f"📊 إحصائيات المصادر في قاعدة بياناتنا:\n\n"
        response += f"• **عدد المصادر المختلفة:** {unique_sources:,} مصدر\n\n"
        response += f"**أكثر المصادر:**\n"
        for i, (name, count) in enumerate(list(top_sources.items())[:10], 1):
            response += f"  {i}. {name} — {count:,} حديث\n"
        return response

    # Check if asking about muhadditheen
    muhaddith_keywords = ["محدث", "المحدث", "المحدثين", "muhaddith"]
    if any(k in normalized_q or k in query_lower for k in muhaddith_keywords):
        top_m = stats.get("top_muhadditheen", {})
        response = f"📊 إحصائيات المحدثين في قاعدة بياناتنا:\n\n"
        response += f"• **عدد المحدثين المختلفين:** {unique_muhadditheen:,} محدِّث\n\n"
        response += f"**أكثر المحدثين:**\n"
        for i, (name, count) in enumerate(list(top_m.items())[:10], 1):
            response += f"  {i}. {name} — {count:,} حديث\n"
        return response

    # Default: full summary
    response = f"📊 إحصائيات قاعدة بيانات الأحاديث النبوية:\n\n"
    response += f"• **إجمالي الأحاديث:** {total:,} حديث\n\n"
    response += f"**التوزيع حسب الدرجة:**\n"
    for grade_ar_label, count in by_grade_ar.items():
        response += f"  • {grade_ar_label}: {count:,} حديث\n"
    response += f"\n• **عدد الرواة المختلفين:** {unique_narrators:,}\n"
    response += f"• **عدد المصادر المختلفة:** {unique_sources:,}\n"
    response += f"• **عدد المحدثين المختلفين:** {unique_muhadditheen:,}\n"
    has_exp = stats.get("has_explanation_count", 0)
    if has_exp:
        response += f"• **أحاديث لها شرح:** {has_exp:,}\n"

    return response


def is_dataset_stats_query(text: str) -> bool:
    """Check if the query is asking about dataset statistics."""
    for pattern in _STATS_PATTERNS:
        if pattern.search(text):
            return True
    return False

# ============================================================
# Book / Source Name Detection
# ============================================================

# Maps query mentions -> canonical masdar values stored in vector payload field 'book'.
# Keys are patterns that may appear in queries; values are exact payload masdar strings.
# NOTE: all patterns use alef-normalized forms (أ إ آ → ا) for robust matching.
_BOOK_NAME_MAP: list[tuple[re.Pattern, str | list[str]]] = [
    # Sahih Bukhari
    (re.compile(r"(صحيح\s*البخاري|البخاري|بخاري|bukhari)", re.IGNORECASE), "صحيح البخاري"),
    # Sahih Muslim
    (re.compile(r"(صحيح\s*مسلم|مسلم|muslim)", re.IGNORECASE), "صحيح مسلم"),
    # Sunan Abi Dawud has two source aliases: 'سنن أبي داود' and 'صحيح أبي داود'.
    (re.compile(r"(سنن\s*ابي?\s*داود|ابو?\s*داود|ابي\s*داود|abu\s*daw[ou]d)", re.IGNORECASE), ["سنن أبي داود", "صحيح أبي داود"]),
    # Sunan al-Tirmidhi
    (re.compile(r"(سنن\s*الترمذي|الترمذي|ترمذي|tirmidhi)", re.IGNORECASE), "سنن الترمذي"),
    # Sunan Ibn Majah
    (re.compile(r"(سنن\s*ابن\s*ماجه|ابن\s*ماجه|ibn\s*majah)", re.IGNORECASE), "سنن ابن ماجه"),
    # Sunan al-Nasai
    (re.compile(r"(سنن\s*النسائي|النسائي|نسائي|nasai|nasa'i)", re.IGNORECASE), "صحيح النسائي"),
    # Musnad Ahmad — alef-normalized (احمد instead of أحمد)
    (re.compile(r"(مسند\s*احمد|مسند\s*الامام\s*احمد|احمد\s*بن\s*حنبل|ahmad)", re.IGNORECASE), "تخريج المسند لشعيب"),
    # Sahih Ibn Hibban — alef-normalized
    (re.compile(r"(صحيح\s*ابن\s*حبان|ابن\s*حبان|ibn\s*hibban)", re.IGNORECASE), "صحيح ابن حبان"),
    # Muwatta Malik
    (re.compile(r"(موطا\s*مالك|الموطا|muwatta)", re.IGNORECASE), "موطأ مالك"),
    # Sahih Ibn Khuzaymah — alef-normalized
    (re.compile(r"(صحيح\s*ابن\s*خزيمة|ابن\s*خزيمة|ibn\s*khuzaymah)", re.IGNORECASE), "صحيح ابن خزيمة"),
    # Silsila Sahiha
    (re.compile(r"(السلسلة\s*الصحيحة|السلسله\s*الصحيحه|silsila\s*sahiha)", re.IGNORECASE), "السلسلة الصحيحة"),
    # Sahih al-Jami
    (re.compile(r"(صحيح\s*الجامع|الجامع\s*الصغير)", re.IGNORECASE), "صحيح الجامع"),
    # Targheeb
    (re.compile(r"(الترغيب\s*والترهيب|الترغيب)", re.IGNORECASE), "الترغيب والترهيب"),
]

# Patterns that indicate INCLUSION of a book (positive filter)
_BOOK_INCLUDE_PATTERNS = [
    re.compile(r"(?:ذكر[ةت]?\s+في|(?:ورد[ةت]?|جاء[تة]?)\s+في|(?:في|من)\s+(?:كتاب|صحيح|سنن|مسند)\s+)([\u0600-\u06FF\s]+?)(?=\s*(?:و|$|،|,))", re.IGNORECASE),
    re.compile(r"(?:في)\s+(صحيح\s*[\u0600-\u06FF]+|سنن\s*[\u0600-\u06FF]+|مسند\s*[\u0600-\u06FF]+)", re.IGNORECASE),
]

# Patterns that indicate EXCLUSION of a book (negative filter — post-retrieval only)
_BOOK_EXCLUDE_PATTERNS = [
    re.compile(r"(?:ولم\s+(?:تذكر?|يذكر?|ترد?|يرد?)\s+في|ولا\s+(?:في|توجد?\s+في))\s+([\u0600-\u06FF\s]+?)(?=\s*(?:و|$|،|,))", re.IGNORECASE),
    re.compile(r"(?:لم\s+(?:تذكر?|يذكر?|ترد?|يرد?)\s+في)\s+([\u0600-\u06FF\s]+?)(?=\s*(?:و|$|،|,))", re.IGNORECASE),
]


def detect_book_filter(text: str) -> tuple[str | list[str], list[str]]:
    """
    Detect book name inclusions and exclusions from query text.

    Returns:
        (extracted_masdar, excluded_masdars)
        - extracted_masdar: book to INCLUDE; str for single book or list[str] for multiple
          source aliases of the same book (e.g. Abi Dawud has two entries)
        - excluded_masdars: books to EXCLUDE (applied post-retrieval by the LLM prompt)
    """
    extracted_masdar: str | list[str] = ""
    excluded_masdars: list[str] = []

    # Always use alef-normalized text for matching (أ إ آ ٱ → ا)
    norm = normalize_alef(normalize_query(text))

    # Check for exclusion patterns first to avoid capturing them as inclusions
    excluded_raw_texts: list[str] = []
    for pat in _BOOK_EXCLUDE_PATTERNS:
        for m in pat.finditer(norm):
            excluded_raw_texts.append(m.group(1).strip())

    # Map excluded raw texts to canonical book names
    for raw in excluded_raw_texts:
        raw_norm = normalize_alef(raw)
        for book_pat, canonical in _BOOK_NAME_MAP:
            if book_pat.search(raw_norm):
                if isinstance(canonical, list):
                    excluded_masdars.extend(canonical)
                else:
                    excluded_masdars.append(canonical)
                break

    # Check for inclusion: scan the normalized text for known book names.
    # Only count as an inclusion if it's not solely appearing in an exclusion phrase.
    for book_pat, canonical in _BOOK_NAME_MAP:
        if book_pat.search(norm):
            if isinstance(canonical, list):
                is_only_excluded = all(c in excluded_masdars for c in canonical)
                if not is_only_excluded and not extracted_masdar:
                    extracted_masdar = canonical  # keep as list for multi-value $in filter
            else:
                is_only_excluded = canonical in excluded_masdars
                if not is_only_excluded and not extracted_masdar:
                    extracted_masdar = canonical

    return extracted_masdar, list(dict.fromkeys(excluded_masdars))  # dedup preserving order


# ============================================================
# Metadata Question Detection
# ============================================================

_METADATA_PATTERNS = {
    "rawi": [
        re.compile(r"(من\s+رو[اى]ه?|الراوي|رواة|من\s+روى|who\s+narrat)", re.IGNORECASE),
        re.compile(r"(سند\s|إسناد|سلسلة\s+الرواة|chain\s+of\s+narrat)", re.IGNORECASE),
    ],
    "grade": [
        re.compile(r"(ما\s+درجة|ما\s+صحة|هل\s+صح|هل\s+صحيح|هل\s+هو\s+صحيح)", re.IGNORECASE),
        re.compile(r"(is\s+it\s+authentic|is\s+it\s+sahih|what.*grade|what.*authenticity)", re.IGNORECASE),
        re.compile(r"(صحة\s+حديث|درجة\s+حديث|تصحيح|تضعيف|حكم\s+على)", re.IGNORECASE),
        re.compile(r"(صحيح\s+أم\s+ضعيف|صحيح\s+ولا\s+ضعيف)", re.IGNORECASE),
    ],
    "masdar": [
        re.compile(r"(في\s+أي\s+كتاب|ما\s+مصدر|أين\s+ورد|which\s+book|in\s+which\s+book|source\s+of)", re.IGNORECASE),
        re.compile(r"(من\s+أي\s+كتاب|مرجع|ما\s+الكتاب)", re.IGNORECASE),
    ],
    "safha_raqam": [
        re.compile(r"(ما\s+رقم|رقم\s+الحديث|رقم\s+الصفحة|what\s+number|hadith\s+number|page\s+number)", re.IGNORECASE),
    ],
    "muhaddith": [
        re.compile(r"(من\s+حكم|من\s+صحح|من\s+ضعف|المحدث|who\s+graded|who\s+authenticated)", re.IGNORECASE),
        re.compile(r"(من\s+خرجه|من\s+أخرجه|تخريج)", re.IGNORECASE),
    ],
    "category": [
        re.compile(r"(ما\s+باب|في\s+أي\s+باب|التصنيف|ما\s+تصنيف|which\s+category|what\s+topic)", re.IGNORECASE),
    ],
}

# ============================================================
# Transliteration Map (English → Arabic)
# ============================================================

_TRANSLITERATION_MAP = {
    # Common Islamic terms
    "salah": "صلاة",
    "salat": "صلاة",
    "prayer": "صلاة",
    "siyam": "صيام",
    "sawm": "صيام",
    "fasting": "صيام",
    "zakat": "زكاة",
    "zakah": "زكاة",
    "hajj": "حج",
    "umrah": "عمرة",
    "wudu": "وضوء",
    "wudhu": "وضوء",
    "ablution": "وضوء",
    "quran": "قرآن",
    "hadith": "حديث",
    "hadeeth": "حديث",
    "sunnah": "سنة",
    "dua": "دعاء",
    "duaa": "دعاء",
    "dhikr": "ذكر",
    "zikr": "ذكر",
    "tawbah": "توبة",
    "taubah": "توبة",
    "repentance": "توبة",
    "jannah": "جنة",
    "paradise": "جنة",
    "jahannam": "جهنم",
    "hellfire": "نار",
    "sabr": "صبر",
    "patience": "صبر",
    "taqwa": "تقوى",
    "iman": "إيمان",
    "faith": "إيمان",
    "ihsan": "إحسان",
    "jihad": "جهاد",
    "niyyah": "نية",
    "niyah": "نية",
    "intention": "نية",
    "sadaqah": "صدقة",
    "charity": "صدقة",
    "nikah": "نكاح",
    "marriage": "نكاح زواج",
    "talaq": "طلاق",
    "divorce": "طلاق",
    "haram": "حرام",
    "halal": "حلال",
    "inshallah": "إن شاء الله",
    "insha allah": "إن شاء الله",
    "alhamdulillah": "الحمد لله",
    "subhanallah": "سبحان الله",
    "astaghfirullah": "أستغفر الله",
    "bismillah": "بسم الله",
    "allahu akbar": "الله أكبر",
    "ramadan": "رمضان",
    "eid": "عيد",
    "jumuah": "جمعة",
    "friday": "جمعة",
    "masjid": "مسجد",
    "mosque": "مسجد",
    "imam": "إمام",
    "sharia": "شريعة",
    "fiqh": "فقه",
    "fatwa": "فتوى",
    "haya": "حياء",
    "modesty": "حياء",
    "birr": "بر",
    "riba": "ربا",
    "interest": "ربا",
    "istikhara": "استخارة",
    "tahajjud": "تهجد",
    "qiyam": "قيام",
    "taraweeh": "تراويح",
    "itikaf": "اعتكاف",
    # Narrator names
    "abu hurairah": "أبو هريرة",
    "abu hurayrah": "أبو هريرة",
    "aisha": "عائشة",
    "aishah": "عائشة",
    "ibn umar": "ابن عمر",
    "ibn abbas": "ابن عباس",
    "anas ibn malik": "أنس بن مالك",
    "abu bakr": "أبو بكر",
    "umar": "عمر",
    "uthman": "عثمان",
    # Book names
    "bukhari": "البخاري صحيح البخاري",
    "muslim": "صحيح مسلم",
    "tirmidhi": "الترمذي",
    "abu dawud": "أبو داود",
    "ibn majah": "ابن ماجه",
    "nasai": "النسائي",
    "ahmad": "أحمد مسند أحمد",
}

# Query type detection patterns
_RULING_PATTERNS = [
    re.compile(r"(ما\s+صحة|هل\s+صح|درجة|صحيح|ضعيف|موضوع|حكم)"),
    re.compile(r"(ما\s+حكم|هل\s+يصح|تخريج|إسناد|سند)"),
]

_NARRATOR_PATTERNS = [
    re.compile(r"(أحاديث\s+رواها|روى\s+عن|الراوي|رواه|حدثنا|أخبرنا)"),
    re.compile(r"(أحاديث\s+عن\s+\w+\s+بن|أحاديث\s+أبو?\s+\w+)"),
]

_HADITH_LOOKUP_PATTERNS = [
    re.compile(r"(حديث\s+[\"«]|ما\s+نص|قال\s+رسول|قال\s+النبي|صلى\s+الله\s+عليه)"),
    re.compile(r"(من\s+غشنا|لا\s+ضرر|إنما\s+الأعمال|الدين\s+النصيحة)"),  # Famous hadith starts
]

_TOPICAL_HADITH_REQUEST_PATTERNS = [
    re.compile(r"(?:حديث|احاديث|أحاديث|الأحاديث)\s+(?:عن|في|حول|بخصوص)\s+\S+"),
    re.compile(r"(?:اذكر|أذكر|اعطني|أعطني|هات|اريد|أريد)\s+(?:لي\s+)?(?:حديث|احاديث|أحاديث)\s+(?:عن|في|حول|بخصوص)\s+\S+"),
]

_SPARSE_QUERY_STOPWORDS = {
    "اذكر", "أذكر", "اعطني", "أعطني", "هات", "اريد", "أريد", "حديث",
    "احاديث", "أحاديث", "الحديث", "عن", "في", "حول", "بخصوص", "ما",
    "من", "الى", "إلى", "على", "هذا", "هذه", "الذي", "التي", "يجب",
    "واجب", "ينبغي", "يلزم", "لنا", "لي", "به", "فيه",
}

# Topic expansion keywords
_TOPIC_EXPANSIONS = {
    "صلاة": "صلاة فرض نافلة ركعة سجود قيام فضل الصلاة",
    "صيام": "صيام صوم رمضان إفطار سحور فضل الصيام",
    "زكاة": "زكاة صدقة نصاب مال إنفاق فريضة",
    "حج": "حج عمرة طواف سعي عرفة منى مناسك",
    "وضوء": "وضوء طهارة غسل تيمم ماء نواقض",
    "نية": "نية إخلاص عمل قصد إنما الأعمال بالنيات",
    "صدق": "صدق كذب أمانة صادق الصدق",
    "صبر": "صبر بلاء ابتلاء احتساب الصابرين",
    "توبة": "توبة استغفار ذنب معصية التائب",
    "جنة": "جنة نار آخرة حساب ثواب يوم القيامة",
    "دعاء": "دعاء ذكر استغفار تسبيح أذكار",
    "بر": "بر والدين أم أب إحسان حق الوالدين",
    "علم": "علم طلب فقه تعلم عالم فضل العلم",
    "جهاد": "جهاد سبيل الله غزوة قتال فضل الجهاد",
    "أخلاق": "أخلاق أدب حسن الخلق معاملة",
    "رحمة": "رحمة رحم رفق لين شفقة",
    "تواضع": "تواضع كبر تكبر خشوع",
    "أمانة": "أمانة خيانة وفاء عهد",
    "ظلم": "ظلم عدل قسط ظالم مظلوم",
    "تعليم": "تعليم تعلم علموا",
    "تعليمه": "تعليم تعلم علموا",
    "أولاد": "أولاد أبناء صبيان",
    "اولاد": "اولاد ابناء صبيان",
}


# ============================================================
# Core Functions
# ============================================================

def detect_arabic(text: str) -> bool:
    """Check if text is predominantly Arabic."""
    arabic_chars = len(_ARABIC_CHARS.findall(text))
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return False
    return arabic_chars / total_alpha > 0.5


def normalize_query(text: str) -> str:
    """Normalize an Arabic query: strip tashkeel, tatweel, whitespace."""
    text = _TASHKEEL.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def normalize_alef(text: str) -> str:
    """Normalize alef variants for query matching (أ إ آ ٱ → ا)."""
    return _ALEF_VARIANTS.sub("ا", text)


def translate_transliterations(text: str) -> str:
    """
    Convert transliterated Islamic terms to Arabic.
    Handles mixed Arabic-English queries.
    """
    # Check for multi-word transliterations first (longer matches first)
    sorted_terms = sorted(_TRANSLITERATION_MAP.keys(), key=len, reverse=True)
    
    result = text
    for term in sorted_terms:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub(_TRANSLITERATION_MAP[term], result)
    
    return result.strip()


def is_greeting(text: str) -> bool:
    """Check if the query is a greeting or pleasantry."""
    text_clean = normalize_query(text.strip())
    
    for pattern in _GREETING_PATTERNS_AR:
        if pattern.search(text_clean):
            return True
    
    for pattern in _GREETING_PATTERNS_EN:
        if pattern.search(text_clean):
            return True
    
    return False


def is_out_of_scope(text: str) -> bool:
    """Check if the query is unrelated to Islam/hadith."""
    for pattern in _OUT_OF_SCOPE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def detect_metadata_fields(text: str) -> list[str]:
    """
    Detect which metadata fields the user is asking about.
    
    Returns list of field names: ['rawi', 'grade', 'masdar', etc.]
    """
    fields = []
    for field_name, patterns in _METADATA_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                fields.append(field_name)
                break
    return fields


def classify_query(text: str, metadata_fields: list[str]) -> QueryType:
    """Classify the query into a type based on pattern matching."""
    # Greeting check first
    if is_greeting(text):
        return QueryType.GREETING
    
    # Out-of-scope check
    if is_out_of_scope(text):
        return QueryType.OUT_OF_SCOPE
    
    # Dataset statistics check (before metadata — "كم عدد" is stats, not metadata)
    if is_dataset_stats_query(text):
        return QueryType.DATASET_STATS
    
    # Metadata question (highest priority after greeting/OOS/stats)
    if metadata_fields:
        return QueryType.METADATA

    # Topical requests like "اذكر حديث عن الصبر" are not requests to explain
    # a specific hadith text. Route them as topic searches so generation does
    # not fall into the "exact hadith not found" branch.
    for pattern in _TOPICAL_HADITH_REQUEST_PATTERNS:
        if pattern.search(text):
            return QueryType.TOPIC

    # Explain/clarify a specific hadith (before HADITH_LOOKUP)
    if is_explain_query(text):
        return QueryType.EXPLAIN_HADITH
    
    for pattern in _HADITH_LOOKUP_PATTERNS:
        if pattern.search(text):
            return QueryType.HADITH_LOOKUP
    
    for pattern in _RULING_PATTERNS:
        if pattern.search(text):
            return QueryType.RULING
    
    for pattern in _NARRATOR_PATTERNS:
        if pattern.search(text):
            return QueryType.NARRATOR
    
    # If it contains topical keywords, it's a topic query
    for keyword in _TOPIC_EXPANSIONS:
        if keyword in text:
            return QueryType.TOPIC
    
    return QueryType.GENERAL


def _strip_clitic_prefix(token: str) -> str:
    """Remove common Arabic one-word prefixes used before nouns and verbs."""
    for prefix in ("وال", "بال", "كال", "فال", "ولل", "فلل", "لل", "ال"):
        if len(token) - len(prefix) >= 3 and token.startswith(prefix):
            return token[len(prefix):]
    for prefix in ("و", "ف", "ب", "ك", "ل"):
        if len(token) - len(prefix) >= 3 and token.startswith(prefix):
            return token[len(prefix):]
    return token


def _strip_pronoun_suffix(token: str) -> str:
    """Return a light stem by removing common Arabic attached pronouns."""
    for suffix in ("كما", "هما", "كم", "كن", "هم", "هن", "نا", "ها", "ه", "ك", "ي"):
        if len(token) - len(suffix) >= 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _generic_sparse_variants(token: str) -> list[str]:
    """
    Generate morphology-oriented variants without adding hadith-specific phrases.

    This improves sparse retrieval when the user's wording uses abstract forms
    (تعليم/يجب) while hadith matn often uses imperative or attached-pronoun
    forms (علموا/مروا/أولادكم).
    """
    token = normalize_alef(token)
    base = _strip_pronoun_suffix(_strip_clitic_prefix(token))
    variants = [base] if base and base != token else []

    if any(root in base for root in ("علم", "تعلم", "تعليم")):
        variants.extend(["علم", "تعلم", "تعليم", "علموا"])

    if base in {"يجب", "واجب", "ينبغي", "يلزم"}:
        variants.extend(["امر", "امروا", "مروا", "اوصى", "وصية"])

    if base in {"ولد", "اولاد", "ابن", "ابناء", "بنين", "صبي", "صبيان", "طفل", "اطفال"}:
        variants.extend([
            "ولد", "اولاد", "اولادكم", "اولادهم",
            "ابن", "ابناء", "ابناءكم", "صبي", "صبيان",
        ])

    result = []
    for variant in variants:
        if variant and variant not in result:
            result.append(variant)
    return result


def build_sparse_query_text(text: str) -> str:
    """Build a cleaner sparse query with generic Arabic morphology variants."""
    normalized = normalize_alef(normalize_query(text))
    raw_tokens = re.findall(r"[\u0600-\u06FF]+", normalized)

    terms: list[str] = []
    for token in raw_tokens:
        if len(token) < 2:
            continue
        if token not in _SPARSE_QUERY_STOPWORDS and token not in terms:
            terms.append(token)
        for variant in _generic_sparse_variants(token):
            if variant not in terms and variant not in _SPARSE_QUERY_STOPWORDS:
                terms.append(variant)

    return " ".join(terms).strip() or normalized


def expand_short_query(text: str) -> str:
    """
    Expand short queries with related terms.
    
    Short queries (<3 meaningful words) often miss relevant results
    because they lack the lexical overlap needed for sparse retrieval
    and may be too ambiguous for dense retrieval.
    """
    words = text.split()
    if len(words) >= 3:
        return text  # Not short, no expansion needed
    
    expansions = []
    for word in words:
        if word in _TOPIC_EXPANSIONS:
            expansions.append(_TOPIC_EXPANSIONS[word])
    
    if expansions:
        return text + " " + " ".join(expansions)
    
    return text


def _extract_hadith_text_from_explain_query(text: str) -> str:
    """
    For explanation/clarification questions, extract the hadith text for retrieval.

    Strips meta-instruction prefixes so retrieval focuses on the hadith content
    rather than the instruction verb.

    E.g., "اشرح حديث اختلاف امتي رحمة"  → "اختلاف امتي رحمة"
    E.g., "فسر لي حديث إنما الأعمال بالنيات" → "إنما الأعمال بالنيات"
    E.g., "ما معنى حديث الدين النصيحة"    → "الدين النصيحة"
    """
    explain_prefixes = [
        r"اشرح\s+(لي\s+)?(حديث|هذا\s+الحديث)?\s*",
        r"أشرح\s+(لي\s+)?(حديث|هذا\s+الحديث)?\s*",
        r"شرح\s+(حديث)?\s*",
        r"فسر\s+(لي\s+)?(حديث)?\s*",
        r"فسّر\s+(لي\s+)?(حديث)?\s*",
        r"ما\s+معنى\s+(حديث)?\s*",
        r"ما\s+مفهوم\s+(حديث)?\s*",
        r"اذكر\s+(لي\s+)?(حديث)?\s*",
        r"أذكر\s+(لي\s+)?(حديث)?\s*",
        r"وضح\s+(لي\s+)?(حديث)?\s*",
        r"أوضح\s+(لي\s+)?(حديث)?\s*",
        r"تفسير\s+(حديث)?\s*",
        r"شرح\s+معنى\s+(حديث)?\s*",
    ]

    cleaned = text
    for prefix_pattern in explain_prefixes:
        new = re.sub(f"^{prefix_pattern}", "", cleaned, flags=re.IGNORECASE).strip()
        if new and new != cleaned and len(new) > 3:
            cleaned = new
            break  # Only strip the first matching prefix

    # Also strip a leading "حديث" if it remains as a bare word at start
    cleaned = re.sub(r"^حديث\s+", "", cleaned).strip()

    if cleaned and cleaned != text and len(cleaned) > 3:
        return cleaned
    return text


def is_explain_query(text: str) -> bool:
    """Check if this is a request to explain/clarify a specific hadith."""
    explain_patterns = [
        re.compile(r"^(اشرح|أشرح|شرح)\s+(لي\s+)?(حديث|هذا\s+الحديث)?", re.IGNORECASE),
        re.compile(r"^(فسر|فسّر)\s+(لي\s+)?(حديث)?", re.IGNORECASE),
        re.compile(r"^(ما\s+معنى|ما\s+مفهوم)\s+(حديث)?", re.IGNORECASE),
        re.compile(r"^(اذكر|أذكر|وضح|أوضح)\s+(لي\s+)?(حديث)?", re.IGNORECASE),
        re.compile(r"^تفسير\s+(حديث)?", re.IGNORECASE),
    ]
    for p in explain_patterns:
        if p.search(text):
            return True
    return False


def _extract_hadith_text_from_metadata_query(text: str) -> str:
    """
    For metadata questions, extract the hadith text portion for retrieval.
    
    E.g., "من رواه حديث من غشنا فليس منا" → "من غشنا فليس منا"
    E.g., "ما درجة حديث إنما الأعمال بالنيات" → "إنما الأعمال بالنيات"
    """
    # Remove metadata question prefixes to get the hadith text
    prefixes_to_strip = [
        r"من\s+رو[اى]ه?\s*(حديث)?\s*",
        r"ما\s+درجة\s*(حديث)?\s*",
        r"ما\s+صحة\s*(حديث)?\s*",
        r"هل\s+صح\s*(حديث)?\s*",
        r"في\s+أي\s+كتاب\s*(ورد)?\s*(حديث)?\s*",
        r"ما\s+مصدر\s*(حديث)?\s*",
        r"ما\s+رقم\s*(حديث)?\s*",
        r"رقم\s+(حديث)\s*",
        r"من\s+حكم\s+على\s*(حديث)?\s*",
        r"من\s+صحح\s*(حديث)?\s*",
        r"من\s+ضعف\s*(حديث)?\s*",
        r"من\s+خرج\s*(حديث)?\s*",
        r"من\s+أخرج\s*(حديث)?\s*",
        r"تخريج\s*(حديث)?\s*",
        r"ما\s+باب\s*(حديث)?\s*",
        r"ما\s+تصنيف\s*(حديث)?\s*",
        r"سند\s*(حديث)?\s*",
        r"إسناد\s*(حديث)?\s*",
    ]
    
    cleaned = text
    for prefix_pattern in prefixes_to_strip:
        cleaned = re.sub(f"^{prefix_pattern}", "", cleaned).strip()
    
    # If we stripped something meaningful, return the cleaned version
    if cleaned and cleaned != text and len(cleaned) > 3:
        return cleaned
    
    return text


def preprocess_query(query: str) -> ProcessedQuery:
    """
    Full query preprocessing pipeline.
    
    Steps:
    1. Detect language
    2. Handle transliterations (English → Arabic)
    3. Check for greetings
    4. Check for out-of-scope
    5. Normalize (strip tashkeel, tatweel, whitespace)
    6. Detect metadata fields being asked about
    7. Classify query type
    8. Expand short queries
    9. For metadata queries, extract hadith text for retrieval
    10. Prepare separate dense and sparse query texts
    
    Args:
        query: Raw user query string
        
    Returns:
        ProcessedQuery with all preprocessing results
    """
    query = query.strip()
    if not query:
        return ProcessedQuery(
            original=query,
            normalized="",
            query_type=QueryType.GENERAL,
            is_arabic=False,
            is_short=True,
            expanded="",
            dense_query="",
            sparse_query="",
        )
    
    # Step 1: Detect language
    is_arabic = detect_arabic(query)
    
    # Step 2: Handle transliterations for non-Arabic / mixed queries
    working_query = query
    if not is_arabic:
        translated = translate_transliterations(query)
        if translated != query:
            working_query = translated
            is_arabic = detect_arabic(working_query)
    
    # Step 3: Greeting detection
    if is_greeting(query):
        return ProcessedQuery(
            original=query,
            normalized=query,
            query_type=QueryType.GREETING,
            is_arabic=is_arabic,
            is_short=True,
            expanded="",
            dense_query="",
            sparse_query="",
            metadata_fields=[],
            skip_retrieval=True,
            direct_response=_GREETING_RESPONSE,
        )
    
    # Step 4: Out-of-scope detection
    if not is_arabic and is_out_of_scope(query):
        return ProcessedQuery(
            original=query,
            normalized=query,
            query_type=QueryType.OUT_OF_SCOPE,
            is_arabic=False,
            is_short=True,
            expanded="",
            dense_query="",
            sparse_query="",
            metadata_fields=[],
            skip_retrieval=True,
            direct_response=_OUT_OF_SCOPE_RESPONSE,
        )
    
    # Step 4b: Dataset statistics detection (early exit — no retrieval needed)
    if is_dataset_stats_query(query) or is_dataset_stats_query(working_query):
        stats_response = _build_stats_response(query)
        return ProcessedQuery(
            original=query,
            normalized=normalize_query(working_query) if is_arabic else working_query,
            query_type=QueryType.DATASET_STATS,
            is_arabic=is_arabic,
            is_short=False,
            expanded="",
            dense_query="",
            sparse_query="",
            metadata_fields=[],
            skip_retrieval=True,
            direct_response=stats_response,
        )
    
    # Step 5: Normalize
    normalized = normalize_query(working_query) if is_arabic else working_query
    
    # Step 6: Detect metadata fields
    metadata_fields = detect_metadata_fields(query) + detect_metadata_fields(normalized)
    metadata_fields = list(set(metadata_fields))  # deduplicate

    # Step 6b: Detect book/source filters from query text
    extracted_masdar, excluded_masdar = detect_book_filter(query)
    if not extracted_masdar:
        extracted_masdar, excluded_masdar_norm = detect_book_filter(normalized)
        if excluded_masdar_norm:
            excluded_masdar = list(dict.fromkeys(excluded_masdar + excluded_masdar_norm))

    # Step 7: Classify query type
    query_type = classify_query(normalized, metadata_fields)
    
    # Step 8: Short query handling (legacy flag kept for compatibility)
    words = normalized.split()
    is_short = len(words) < 3

    # Step 9: For metadata queries, extract hadith text for better retrieval
    if query_type == QueryType.METADATA:
        retrieval_text = _extract_hadith_text_from_metadata_query(normalized)
    elif query_type == QueryType.EXPLAIN_HADITH:
        # Strip "اشرح حديث X" → use just "X" as retrieval query
        retrieval_text = _extract_hadith_text_from_explain_query(normalized)
    else:
        retrieval_text = normalized

    # Step 10: Multi-strategy query expansion ─────────────────────────────
    # Lazy import to avoid circular dependencies
    from retrieval.query_expander import expand_query

    expansion = expand_query(
        normalized_text=retrieval_text,
        is_arabic=is_arabic,
        query_type=query_type.value,
    )

    generic_sparse_query = build_sparse_query_text(retrieval_text)
    expanded_sparse_query = build_sparse_query_text(expansion.sparse_query)

    # Determine final dense/sparse queries based on query type
    if query_type == QueryType.METADATA:
        # Metadata: use extracted hadith text directly (no expansion noise)
        dense_query = retrieval_text
        sparse_query = generic_sparse_query
        multi_queries = [retrieval_text]
    elif query_type == QueryType.EXPLAIN_HADITH:
        # Explain: use stripped hadith text for retrieval (not the instruction verb)
        dense_query = retrieval_text
        sparse_query = f"{generic_sparse_query} {expanded_sparse_query}".strip()
        multi_queries = [retrieval_text] + expansion.reformulations[:2]
    elif query_type == QueryType.NARRATOR:
        # Narrator: keep full original for dense; use expanded for sparse
        dense_query = normalized
        sparse_query = f"{generic_sparse_query} {expanded_sparse_query}".strip()
        multi_queries = [normalized] + expansion.reformulations[:2]
    else:
        # General / Topic / Ruling / HADITH_LOOKUP: use full expansion
        dense_query = expansion.dense_query
        sparse_query = f"{generic_sparse_query} {expanded_sparse_query}".strip()
        multi_queries = expansion.multi_queries + [generic_sparse_query]

    # Apply alef normalization for sparse query (better recall)
    sparse_query_alef = normalize_alef(sparse_query)

    # Legacy 'expanded' field: the sparse query for backward compat
    expanded = sparse_query_alef

    logger.info(
        f"Query preprocessed: type={query_type.value}, "
        f"arabic={is_arabic}, short={is_short}, "
        f"metadata_fields={metadata_fields}, "
        f"extracted_masdar={repr(extracted_masdar)}, "
        f"excluded_masdar={excluded_masdar}, "
        f"expansion_tokens={len(expansion.expanded_terms)}, "
        f"reformulations={len(expansion.reformulations)}, "
        f"multi_queries={len(multi_queries)}"
    )

    return ProcessedQuery(
        original=query,
        normalized=normalized,
        query_type=query_type,
        is_arabic=is_arabic,
        is_short=is_short,
        expanded=expanded,
        dense_query=dense_query,
        sparse_query=sparse_query_alef,
        metadata_fields=metadata_fields,
        skip_retrieval=False,
        direct_response="",
        multi_queries=multi_queries,
        expansion_tokens=expansion.expanded_terms,
        extracted_masdar=extracted_masdar,
        excluded_masdar=excluded_masdar,
    )
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_queries = [
        "ما صحة حديث من غشنا فليس منا",
        "أحاديث عن الصلاة",
        "صيام",
        "الدين النصيحة",
        "أحاديث رواها أبو هريرة",
        "What is the hadith about intention?",
        "من رواه حديث إنما الأعمال بالنيات",
        "ما درجة حديث الدين النصيحة",
        "في أي كتاب حديث من غشنا",
        "السلام عليكم",
        "how to code in python",
        "what is the hadith about wudu",
        "salah",
        "hadith about sabr",
        # Dataset statistics queries
        "كم عدد الأحاديث",
        "كم حديث صحيح",
        "عدد الأحاديث الضعيفة",
        "كم عدد الرواة",
        "how many hadiths",
        "من أكثر الرواة رواية",
        "إحصائيات قاعدة البيانات",
    ]
    
    for q in test_queries:
        result = preprocess_query(q)
        print(f"\n  Original:    {result.original}")
        print(f"  Type:        {result.query_type.value}")
        print(f"  Arabic:      {result.is_arabic}")
        print(f"  Short:       {result.is_short}")
        print(f"  Metadata:    {result.metadata_fields}")
        print(f"  Skip:        {result.skip_retrieval}")
        print(f"  Dense Q:     {result.dense_query}")
        print(f"  Sparse Q:    {result.sparse_query}")
        if result.direct_response:
            print(f"  Response:    {result.direct_response[:60]}...")
