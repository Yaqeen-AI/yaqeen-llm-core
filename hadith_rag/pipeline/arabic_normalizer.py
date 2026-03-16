# ============================================================
# YaqeenAI — Arabic Text Normalizer for Hadith RAG
# ============================================================
# Strips tashkeel (diacritics), normalizes whitespace, and cleans
# Arabic text for embedding consistency.
#
# DECISION: Strip tashkeel because:
# 1. Inconsistent diacritization across hadith sources
# 2. Jina v3 trained on unvocalized Arabic web text
# 3. Users never type diacritics in queries
#
# DECISION: Do NOT normalize Alef variations (أ إ آ → ا) because
# hamza on Alef distinguishes different Arabic words in hadith text.

import re
import unicodedata
from typing import Any


def _ensure_str(value: Any, fn_name: str) -> str:
    """
    Validate that *value* is a string.

    Raises:
        TypeError: If *value* is not a str instance.
        ValueError: If *value* is None.
    """
    if value is None:
        raise ValueError(
            f"{fn_name}() received None — expected a str. "
            "Pass an empty string '' for empty-text cases."
        )
    if not isinstance(value, str):
        raise TypeError(
            f"{fn_name}() expected str, got {type(value).__name__!r}. Value: {value!r}"
        )
    return value


# Unicode ranges for Arabic diacritics (tashkeel)
_TASHKEEL_PATTERN = re.compile(
    "["
    "\u0610-\u061a"  # Various Arabic marks
    "\u064b-\u065f"  # Fathatan through Hamza Below
    "\u0670"  # Superscript Alef
    "\u06d6-\u06dc"  # Quranic annotation marks
    "\u06df-\u06e4"  # More Quranic marks
    "\u06e7-\u06e8"  # More marks
    "\u06ea-\u06ed"  # More marks
    "\ufe70-\ufe7f"  # Arabic presentation forms (diacritics)
    "]+"
)

# Tatweel / Kashida (elongation character)
_TATWEEL_PATTERN = re.compile("\u0640+")

# Multiple whitespace → single space
_WHITESPACE_PATTERN = re.compile(r"\s+")

# Matn prefix "- " that appears at the start of every hadith text
_MATN_PREFIX_PATTERN = re.compile(r"^-\s*")


def strip_tashkeel(text: str) -> str:
    """Remove all Arabic diacritical marks (tashkeel) from text."""
    text = _ensure_str(text, "strip_tashkeel")
    return _TASHKEEL_PATTERN.sub("", text)


def remove_tatweel(text: str) -> str:
    """Remove Arabic tatweel/kashida elongation characters."""
    text = _ensure_str(text, "remove_tatweel")
    return _TATWEEL_PATTERN.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space and strip."""
    text = _ensure_str(text, "normalize_whitespace")
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def strip_matn_prefix(text: str) -> str:
    """Remove the '- ' prefix that appears at the start of matn texts."""
    text = _ensure_str(text, "strip_matn_prefix")
    return _MATN_PREFIX_PATTERN.sub("", text)


def normalize_arabic_for_embedding(text: str) -> str:
    """
    Full normalization pipeline for Arabic text before embedding.

    Steps:
    1. Strip the '- ' matn prefix
    2. Remove tashkeel (diacritics)
    3. Remove tatweel (kashida)
    4. Normalize whitespace

    Does NOT normalize Alef variations — preserving أ إ آ ا distinctions.

    Args:
        text: Raw Arabic hadith text (matn field)

    Returns:
        Cleaned text suitable for embedding

    Raises:
        TypeError: If *text* is not a str.
        ValueError: If *text* is None.
    """
    text = _ensure_str(text, "normalize_arabic_for_embedding")
    text = strip_matn_prefix(text)
    text = strip_tashkeel(text)
    text = remove_tatweel(text)
    text = normalize_whitespace(text)
    return text


def clean_matn_preserve_diacritics(text: str) -> str:
    """
    Light cleaning that preserves diacritics (for display purposes).

    Steps:
    1. Strip the '- ' matn prefix
    2. Remove tatweel
    3. Normalize whitespace

    Args:
        text: Raw Arabic hadith text (matn field)

    Returns:
        Cleaned text with diacritics preserved (for human reading)

    Raises:
        TypeError: If *text* is not a str.
        ValueError: If *text* is None.
    """
    text = _ensure_str(text, "clean_matn_preserve_diacritics")
    text = strip_matn_prefix(text)
    text = remove_tatweel(text)
    text = normalize_whitespace(text)
    return text


if __name__ == "__main__":
    # Quick test
    sample = "- إيَّاكم والظَّنَّ فإنَّ الظَّنَّ أكذَبُ الحديثِ ولا تجسَّسوا ولا تحسَّسوا"
    print("Original:", sample)
    print("For embedding:", normalize_arabic_for_embedding(sample))
    print("For display:", clean_matn_preserve_diacritics(sample))
