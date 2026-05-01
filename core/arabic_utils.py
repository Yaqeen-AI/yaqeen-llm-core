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
        r"الأحناف",
        r"أبو حنيفة",
        r"ابن عابدين",
        r"السرخسي",
        r"الكاساني",
    ]],
    "مالكي": [re.compile(p) for p in [
        r"المالكي[ةه]?",
        r"الإمام مالك",
        r"القرافي",
        r"الحطاب",
        r"ابن عبد البر",
        r"ابن رشد",
    ]],
    "شافعي": [re.compile(p) for p in [
        r"الشافعي[ةه]?",
        r"الإمام الشافعي",
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
        r"المذاهب الأربعة",
        r"اتفق الفقهاء",
        r"أجمع الفقهاء",
    ]],
}


def detect_mazhabs(text: str) -> list[str]:
    """Return sorted list of mazhab keys mentioned in text."""
    return sorted(
        name
        for name, patterns in _MAZHAB_PATTERNS.items()
        if any(p.search(text) for p in patterns)
    )


# ── Citation formatting ───────────────────────────────────────────────────────

def format_citation(volume_id: str, book_page: str, chunk_page: str) -> str:
    """
    Return a compact academic reference string.
    Example: "م.ف.ك — ج٢، ص١٢٣"
    """
    vol_num = "".join(filter(str.isdigit, volume_id))
    page_num = "".join(c for c in book_page if c.isdigit())
    return f"م.ف.ك — ج{vol_num}، ص{page_num}" if vol_num and page_num else f"{volume_id} | {book_page}"
