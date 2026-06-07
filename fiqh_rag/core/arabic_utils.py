"""
Arabic text normalisation and analysis utilities.

Normalisation pipeline (in order):
  1. NFKC  — expands compatibility forms (ﷺ → صلى الله عليه وسلم, ligatures, etc.)
  2. Alef variants       (أ إ آ ٱ → ا)
  3. Ta marbuta          (ة → ه)  — optional, off by default
  4. Ya variants         (ى → ي)
  5. Waw/Ya hamza        (ؤ → و,  ئ → ي)
  6. Tatweel             (ـ removed)
  7. Tashkeel/diacritics (حركات removed)
  8. Eastern numerals    (٠–٩ → 0–9)
  9. Whitespace          (collapsed to single space)

Mazhab detection:
  detect_mazhabs(text) → list of mazhab keys found in text

Citation formatting:
  format_citation(result) → human-readable academic reference string
"""

import re
import unicodedata

# ── Compiled regex patterns ──────────────────────────────────────────────────

_TASHKEEL   = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)
_TATWEEL    = re.compile(r"\u0640")
_ALEF       = re.compile(r"[أإآٱ\u0671-\u0673\u0675]")
_YA         = re.compile(r"ى")
_WAW_HAMZA  = re.compile(r"ؤ")
_YA_HAMZA   = re.compile(r"ئ")
_TA_MARBUTA = re.compile(r"ة")
_WHITESPACE = re.compile(r"\s+")

_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


# ── Core normalisation ───────────────────────────────────────────────────────

def normalize(
    text: str,
    *,
    remove_tashkeel: bool = True,
    normalize_ta_marbuta: bool = False,
    normalize_numerals: bool = True,
) -> str:
    """
    Full Arabic normalisation pipeline.

    Args:
        remove_tashkeel:      Strip diacritics (default True — improves TF-IDF recall).
        normalize_ta_marbuta: Map ة → ه (default False — changes morphology).
        normalize_numerals:   Map ٠–٩ → 0–9 (default True).
    """
    if not text:
        return text

    # 1. NFKC: expands ﷺ, Arabic presentation-form ligatures, etc.
    text = unicodedata.normalize("NFKC", text)

    # 2–5. Character-level substitutions
    text = _ALEF.sub("ا", text)
    if normalize_ta_marbuta:
        text = _TA_MARBUTA.sub("ه", text)
    text = _YA.sub("ي", text)
    text = _WAW_HAMZA.sub("و", text)
    text = _YA_HAMZA.sub("ي", text)

    # 6. Remove tatweel
    text = _TATWEEL.sub("", text)

    # 7. Remove tashkeel/diacritics
    if remove_tashkeel:
        text = _TASHKEEL.sub("", text)

    # 8. Eastern → Western numerals
    if normalize_numerals:
        text = text.translate(_EASTERN_DIGITS)

    # 9. Collapse whitespace
    text = _WHITESPACE.sub(" ", text).strip()

    return text


# Aliases expected by ingest.py and retriever.py
normalize_query  = normalize
normalize_corpus = normalize


# ── Mazhab detection ─────────────────────────────────────────────────────────

_MAZHAB_PATTERNS: dict[str, list[re.Pattern]] = {
    "حنفي": [re.compile(p) for p in [
        r"الحنفي[ةه]?",
        r"الاحناف",
        r"ابو حنيفة",
        r"ابن عابدين",
        r"السرخسي",
        r"الكاساني",
    ]],
    "مالكي": [re.compile(p) for p in [
        r"المالكي[ةه]?",
        r"الامام مالك",
        r"القرافي",
        r"الحطاب",
        r"ابن عبد البر",
        r"ابن رشد",
    ]],
    "شافعي": [re.compile(p) for p in [
        r"الشافعي[ةه]?",
        r"الامام الشافعي",
        r"النووي",
        r"الرافعي",
        r"الماوردي",
        r"الغزالي",
    ]],
    "حنبلي": [re.compile(p) for p in [
        r"الحنابل[ةه]?",
        r"الحنبلي[ةه]?",
        r"ابن قدامة",
        r"ابن حنبل",
        r"ابن تيمية",
        r"ابن القيم",
    ]],
    "جمهور": [re.compile(p) for p in [
        r"الجمهور",
        r"جمهور الفقهاء",
        r"المذاهب الاربعة",
        r"اتفق الفقهاء",
        r"اجمع الفقهاء",
    ]],
}


def detect_mazhabs(text: str) -> list[str]:
    """Return sorted list of mazhab keys mentioned in text."""
    text = normalize(text)  # strip diacritics + unify alef variants before matching
    return sorted(
        name
        for name, patterns in _MAZHAB_PATTERNS.items()
        if any(p.search(text) for p in patterns)
    )


# ── Fiqh topic detection ─────────────────────────────────────────────────────

# Minimum pattern hits before a topic filter is trusted.
# 1 works well for targeted short Fiqh questions (e.g. "ما حكم الزكاة؟").
# Raise to 2 if you see false-positive filters on long general queries.
_MIN_TOPIC_SCORE = 1

_FIQH_TOPIC_PATTERNS: dict[str, list[re.Pattern]] = {
    # Root r"وضو" catches وضوء/الوضوء/وضوءه/توضأ-derived forms better than exact r"وضوء"
    "طهارة": [re.compile(p) for p in [
        r"وضو",                              # root — catches all wudu forms
        r"طهار[ةه]", r"نجاس[ةه]",
        r"غسل", r"تيمم",
        r"حيض", r"حايض", r"نفاس",
        r"جناب[ةه]", r"استنجاء", r"استجمار",
        r"خفين", r"المسح على الخف",
        r"سواك",
        r"مياه", r"ماء مستعمل",
    ]],
    "صلاة": [re.compile(p) for p in [
        r"صلا[ةه]", r"صلوات",
        r"اذان", r"اقام[ةه]",
        r"ركع[ةه]", r"سجود", r"ركوع",
        r"قبل[ةه]", r"مسجد",
        r"جماع[ةه]",
        r"الامام[ةه]",
        r"قصر الصلا",
        r"سجود السهو", r"سهو",
        r"الجمع[ةه]", r"تشهد", r"وتر",
    ]],
    "زكاة": [re.compile(p) for p in [
        r"زكا[ةه]", r"نصاب", r"الفطر", r"صدق[ةه]", r"العشر",
        r"عروض التجار[ةه]?",
        r"ركاز",
        r"غنم", r"ابل", r"بقر",
    ]],
    "صيام": [re.compile(p) for p in [
        r"صوم", r"صيام", r"رمضان", r"افطار", r"سحور", r"اعتكاف",
    ]],
    "حج": [re.compile(p) for p in [
        r"حج", r"عمر[ةه]", r"احرام", r"طواف", r"سعي",
        r"مك[ةه]", r"مني", r"عرف[ةه]", r"مزدلف[ةه]",
    ]],
    # معاملات split into two sub-categories (~12% each vs. 25% combined)
    # Old Qdrant data tagged "معاملات" falls back gracefully via the retriever fallback.
    "بيوع": [re.compile(p) for p in [
        r"بيع", r"شراء", r"تجار[ةه]", r"ربا",
        r"بيع السلم", r"عقد السلم",
        r"الصرف",
        r"خيار",
        r"الغرر", r"المجهول",
    ]],
    "الشركات والديون": [re.compile(p) for p in [
        r"اجار[ةه]", r"مضارب[ةه]", r"شرك[ةه]", r"وكال[ةه]",
        r"قرض", r"الدين", r"رهن", r"كفال[ةه]", r"ضمان",
    ]],
    "نكاح": [re.compile(p) for p in [
        r"نكاح", r"زواج", r"مهر", r"الولي", r"الزوجين", r"عقد الزواج",
    ]],
    "طلاق": [re.compile(p) for p in [
        r"طلاق", r"خلع", r"رجع[ةه]", r"العد[ةه]", r"ايلاء", r"ظهار", r"فسخ",
    ]],
    "ميراث": [re.compile(p) for p in [
        r"ميراث", r"الارث", r"وصي[ةه]", r"ترك[ةه]", r"الوارث", r"الفرائض",
    ]],
    "جنايات": [re.compile(p) for p in [
        r"قتل", r"قصاص", r"الدي[ةه]", r"حدود", r"سرق[ةه]", r"قذف", r"زنا",
        r"تعزير",
        r"حراب[ةه]",
        r"رد[ةه]",
    ]],
}


def detect_fiqh_topic(text: str) -> list[str] | None:
    """
    Return the Fiqh topic(s) found in text, or None.

    - None      → no hits or below _MIN_TOPIC_SCORE → full-corpus search
    - ["topic"] → single clear winner → narrow to that topic slice
    - ["t1","t2"] → tie → union search across both slices (still a corpus reduction)
    """
    normalized = normalize(text)
    scores: dict[str, int] = {
        topic: sum(1 for p in patterns if p.search(normalized))
        for topic, patterns in _FIQH_TOPIC_PATTERNS.items()
    }
    top_score = max(scores.values())
    if top_score < _MIN_TOPIC_SCORE:
        return None
    return [t for t, s in scores.items() if s == top_score]


# ── Citation formatting ───────────────────────────────────────────────────────

def format_citation(volume_id: str, book_page: str, chunk_page: str) -> str:
    """
    Return a compact academic reference string.
    Example: "م.ف.ك — ج٢، ص١٢٣"
    """
    vol_num = "".join(filter(str.isdigit, volume_id))
    page_num = "".join(c for c in book_page if c.isdigit())
    return f"م.ف.ك — ج{vol_num}، ص{page_num}" if vol_num and page_num else f"{volume_id} | {book_page}"
