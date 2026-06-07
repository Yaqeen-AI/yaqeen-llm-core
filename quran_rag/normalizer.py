"""
quran_rag/normalizer.py

Arabic text normalization for the Quranic RAG pipeline.
Extracted from the Kaggle notebook (KawnPreprocessor).

NOTE: core/arabic_utils.py likely has a simpler normalizer.
      Consider consolidating once both codebases are merged —
      KawnPreprocessor is more comprehensive (handles HTML, URLs, emojis, etc.).

Public API:
    normalize_arabic(text: str) -> str   — main entry point for pipeline code
"""
from __future__ import annotations

import html
import re
from typing import Optional

import emoji
import regex

# ── Compiled regex constants ────────────────────────────────────────────────

_URL_REGEXES = [
    r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
    r"http[s]?://[a-zA-Z0-9_\-./~\?=%&]+",
    r"www[a-zA-Z0-9_\-?=%&/.~]+",
    r"[a-zA-Z]+\.com",
    r"(?=http)[^\s]+",
    r"(?=www)[^\s]+",
    r"://",
]
_USER_MENTION_REGEX = r"@[\w\d]+"
_EMAIL_REGEXES = [r"[\w-]+@([\w-]+\.)+[\w-]+", r"\S+@\S+"]
_MULTIPLE_CHAR_PATTERN = re.compile(r"(\D)\1{2,}", re.DOTALL)

_HYPHEN_PATTERN = r"\u002D\u2010-\u2015_"
_ACCEPTED_NON_LETTER_OR_NUMBER = (
    r"\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_`{\|}~—٪'،؟`୍\u201C؛\u201Dۚ»؛\s«–…'/"
    + _HYPHEN_PATTERN + "·•×÷&!"
)
_ACCEPTED_NON_LETTER_OR_NUMBER_NO_SPACE = (
    r"\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_`{\|}~—٪'،؟`୍\u201C؛\u201Dۚ»؛«–…'/"
    + _HYPHEN_PATTERN + "·•×÷&!"
)
_ACCEPTED_NON_LETTER_OR_NUMBER_NO_DOT_PLUS_MINUS_SPACE = (
    r"\[\]!\"#\$%\'\(\)\*,\:;\<=·>?@\[\\\]\^_`{\|}~—٪'،؟`୍\u201C؛\u201Dۚ»؛«–…'/"
    + "·•×÷&!"
)
_ACCEPTED_NON_LETTER_OR_NUMBER_NO_APOSTROPHE_HYPHEN_DOT_UNDERSCORE_SPACE = (
    r"\[\]!\"#\$%\\(\)\*,\:;\<=·>?@\[\\\]\^`{\|}~٪'،؟`୍\u201C؛\u201Dۚ»؛«…'/"
    + "·•×÷&!"
)

_ARABIC_LETTER_PATTERN = r"\u0621-\u063A\u0641-\u064A"
_HARAKA_PATTERN = r"\u064b-\u0652"
_TATWEEL_PATTERN = r"ـ"
_EXTENDED_ARABIC_LETTER_PATTERN = r"\u0600-\u06FF"
_LATTEN_LETTER_PATTERN = r"A-Za-zÀ-ÖØ-öø-ÿĀ-ſƀ-ƿǝ-ǿ0-9"
_ALL_LETTERS_AND_NUMBERS = r"\w"
_ALL_NUMBER_PATTERN = r"0-9\u0660-\u0669"
_NUMBER_WITH_FLOATING_POINT = r"[+-]?((\d+(\.\d*)?)|(\.\d+))"
_NUMBER_WITH_FLOATING_POINT_STARTS_NON_DIGIT = r"([+-](\d+(\.\d*)?))"

ARABIC_NUMBER_PATTERN = r"0-9" # 0 to 9
HINDI_NUMBER_PATTERN = r"\u0660-\u0669" #٠ to ٩
ALL_NUMBER_PATTERN = ARABIC_NUMBER_PATTERN + HINDI_NUMBER_PATTERN

_EMOJI_PATTERN = "".join(sorted(emoji.EMOJI_DATA, key=len, reverse=True))

_HINDI_NUMS = "٠١٢٣٤٥٦٧٨٩"
_ARABIC_NUMS = "0123456789"
_HINDI_TO_ARABIC_MAP = str.maketrans(_HINDI_NUMS, _ARABIC_NUMS)

_ARABIC_MAP = str.maketrans({
    "ﺏ": "ب", "ﺗ": "ت", "ﺷ": "ش", "ﻦ": "ن", "ﻷ": "لا", "ﺴ": "س", "ﻧ": "ن",
    "ٲ": "ا", "ﻻ": "لا", "ﻜ": "ك", "ﺇ": "ا", "ﺑ": "ب", "ﺎ": "ا", "ک": "ك",
    "ﺯ": "ز", "ﺌ": "ئ", "ﻰ": "ى", "ﻣ": "م", "ﺕ": "ت", "ﺆ": "ؤ", "ﺼ": "ص",
    "ﻥ": "ن", "ﺲ": "س", "ﻠ": "ل", "ﺻ": "ص", "ﺟ": "ج", "ﺰ": "ز", "ﺃ": "ا",
    "ﺁ": "ا", "ﺘ": "ت", "ﺈ": "ا", "ﻳ": "ي", "ی": "ى", "ﺓ": "ة", "ﺜ": "ث",
    "ﺒ": "ب", "ﻟ": "ل", "ﺺ": "ص", "ﺧ": "خ", "ﻒ": "ف", "ﻚ": "ك", "ﺿ": "ض",
    "ﺣ": "ح", "ﻔ": "ف", "ﻴ": "ي", "ﺳ": "س", "ﻭ": "و", "ﻤ": "م", "ﻘ": "ق",
    "ﻋ": "ع", "ﻲ": "ي", "ﻵ": "لا", "ﻞ": "ل", "ﻏ": "غ", "ﻱ": "ي", "ﻙ": "ك",
    "ﻊ": "ع", "ﻪ": "ه", "ﺛ": "ث", "ﻨ": "ن", "ﺨ": "خ", "ۃ": "ة", "ﻡ": "م",
    "ﻫ": "ه", "ﻩ": "ه", "ﺍ": "ا", "ﻗ": "ق", "ﺬ": "ذ", "ﻝ": "ل", "ﻛ": "ك",
    "ﺖ": "ت", "ﻈ": "ظ", "ﺫ": "ذ", "ﺤ": "ح", "ﻼ": "لا", "ٱ": "ا", "ﺪ": "د",
    "ﺮ": "ر", "ﺹ": "ص", "ﺱ": "س", "ﻌ": "ع", "ﺠ": "ج", "ﺄ": "ا", "ﺐ": "ب",
    "ﺩ": "د", "ﺔ": "ة", "ﺭ": "ر", "ﻢ": "م", "ﻬ": "ه", "ﻮ": "و", "ﻓ": "ف",
    "ﷺ": "صلى الله عليه وسلم", "ﷻ": "جل جلاله",
    "ﷲ": "الله", "ﷴ": "محمد", "ﷵ": "صلى الله عليه وسلم",
    "﷽": "بسم الله الرحمن الرحيم",
})

_SCRIPT_TAGS = r"<script.*>[\s\S]*?<\/script>"
_STYLE_TAGS = r"<style.*>[\s\S]*?<\/style>"


class KawnPreprocessor:
    """
    Comprehensive Arabic text normalizer.
    All parameters default to sensible settings for Quranic corpus processing
    (tashkeel preserved, tatweel stripped, non-standard chars standardised).
    """

    def __init__(
        self,
        remove_html_markup: bool = True,
        replace_urls_emails_mentions: bool = True,
        strip_tashkeel: bool = False,           # keep diacritics — important for Quran
        strip_tatweel: bool = True,
        strip_extended_arabic_and_quranic_symbols: bool = True,
        keep_latten: bool = True,
        keep_all_non_arabic_letters: bool = True,
        keep_emojis: bool = True,
        map_hindi_numbers_to_arabic: bool = True,
        standarize_non_traditional_ar_chars: bool = True,
        seperate_nums_and_words: bool = True,
        seperate_nums_from_pucks: bool = False,
        seperate_words_from_pucks: bool = False,
        remove_non_digit_repetition: bool = True,
    ):
        to_keep = _ACCEPTED_NON_LETTER_OR_NUMBER

        self.remove_html_markup = remove_html_markup
        self.replace_urls_emails_mentions = replace_urls_emails_mentions
        self.strip_tatweel = strip_tatweel
        self.map_hindi_numbers_to_arabic = map_hindi_numbers_to_arabic
        self.seperate_nums_and_words = seperate_nums_and_words
        self.seperate_nums_from_pucks = seperate_nums_from_pucks
        self.seperate_words_from_pucks = seperate_words_from_pucks
        self.remove_non_digit_repetition = remove_non_digit_repetition
        self.standarize_non_traditional_ar_chars = standarize_non_traditional_ar_chars

        if not strip_tashkeel:
            to_keep += _HARAKA_PATTERN

        if not strip_extended_arabic_and_quranic_symbols:
            to_keep += r"\u0600-\u064A\u0653-\u06FF" if strip_tashkeel else _EXTENDED_ARABIC_LETTER_PATTERN
        else:
            to_keep += _ARABIC_LETTER_PATTERN

        if keep_all_non_arabic_letters:
            to_keep += _ALL_LETTERS_AND_NUMBERS
        elif keep_latten:
            to_keep += _LATTEN_LETTER_PATTERN

        if keep_emojis:
            to_keep += _EMOJI_PATTERN

        to_keep += r"0-9" if map_hindi_numbers_to_arabic else _ALL_NUMBER_PATTERN

        self._rejected_chars_re = re.compile(r"[^" + to_keep + r"]")

    # ── Public ───────────────────────────────────────────────────────────────

    def preprocess(self, text: str) -> str:
        text = str(text)
        text = html.unescape(text)

        if self.replace_urls_emails_mentions:
            for pat in _URL_REGEXES:
                text = re.sub(pat, "", text)
            for pat in _EMAIL_REGEXES:
                text = re.sub(pat, "", text)
            text = re.sub(_USER_MENTION_REGEX, "", text)

        if self.remove_html_markup:
            text = re.sub(_SCRIPT_TAGS, "", text)
            text = re.sub(_STYLE_TAGS, "", text)
            text = re.sub("<br />", " ", text)
            text = re.sub("</?[^>]+>", " ", text)

        if self.strip_tatweel:
            text = re.sub(_TATWEEL_PATTERN, "", text)

        if self.standarize_non_traditional_ar_chars:
            text = text.translate(_ARABIC_MAP)
        elif self.map_hindi_numbers_to_arabic:
            text = text.translate(_HINDI_TO_ARABIC_MAP)

        if self.remove_non_digit_repetition:
            text = _MULTIPLE_CHAR_PATTERN.sub(r"\1\1", text)

        if self.seperate_nums_and_words:
            text = regex.sub(r"([\p{L}\p{M}ًٌٍَُِْ]+)([\p{N}]+)", r"\1 \2", text)
            text = regex.sub(r"([\p{N}]+)([\p{L}\p{M}ًٌٍَُِْ]+)", r"\1 \2", text)

        text = re.sub(self._rejected_chars_re, "", text)
        text = text.replace("\uFE0F", "")
        text = "\n".join(
            "\t".join(" ".join(sub.split()).strip() for sub in sent.split("\t"))
            for sent in text.split("\n")
            if len(sent) >= 3
        )
        return text


# ── Module-level singleton — avoids re-compiling regexes per call ────────────

_preprocessor = KawnPreprocessor()


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for embedding / indexing.
    Strips tatweel, standardises non-traditional chars, removes URLs/HTML,
    preserves tashkeel (diacritics).
    """
    if not text:
        return ""
    cleaned = _preprocessor.preprocess(text)
    return re.sub(r"\s+", " ", cleaned).strip()
