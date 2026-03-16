# ============================================================
# YaqeenAI — Arabic Text Normalization & Preprocessing
# ============================================================
# Production-grade Arabic text normalization for:
#   1. BM25 indexing (conservative document normalization)
#   2. Query normalization (cautious alef folding only)
#   3. Embedding input (light normalization)
#   4. Display (original preserved)
#
# Handles: tashkeel, hamza, tatweel, alef variants, and ayah
# references without destructively collapsing higher-risk forms.

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from loguru import logger


# ───────────────────────────────────────────────────────────
# Unicode Ranges for Arabic Diacritics (Tashkeel)
# ───────────────────────────────────────────────────────────

# Arabic diacritics (harakat) — fathah, dammah, kasrah, sukun, shadda, tanween, etc.
_DIACRITICS_PATTERN = re.compile(
    r"[\u0610-\u061A"   # Arabic signs
    r"\u064B-\u065F"    # Arabic fathatan → hamza below  (excludes \u0670 — handled separately)
    r"\u06D6-\u06DC"    # Quran annotation marks
    r"\u06DF-\u06E4"    # More Quran marks
    r"\u06E7-\u06E8"    # Yeh barree marks
    r"\u06EA-\u06ED"    # More marks
    r"\uFE70-\uFE7F"    # Arabic presentation forms
    r"]"
)

# Superscript alef (\u0670) — dagger alef.
# In Uthmani Quran script this character represents a long vowel ā that is written above
# the consonant (e.g. كِتَٰبُ = كتاب).  It must be REPLACED by a full alef (ا),
# not removed, otherwise كتاب ≠ كتب after normalization.
_SUPERSCRIPT_ALEF = "\u0670"
_ALEF = "\u0627"

# Some very common Uthmani words where the dagger alef is decorative (not phonemic).
# After normalize_alef these become "ذالك" / "هاذا" etc — we map them back to
# the canonical modern-Arabic spelling so BM25 queries without diacritics match.
_UTHMANI_WORD_MAP = {
    "ذالك": "ذلك",
    "هاذا": "هذا",
    "هاذه": "هذه",
    "هاؤلاء": "هؤلاء",
    "لاكن": "لكن",
}

# Tanwin-alef pattern: accusative tanwin (ًا / ًٰ) leaves a trailing ا after removing diacritics.
# e.g.  مِصْرًا → after diacritics removal → مصرا  (the ا is NOT a diacritic, it's a letter)
# We need to remove that residual ا from words that end with it after tanwin removal.
#
# The pattern must NOT match:
#   - Short function words: لا، ما، ذا، إذا (the ا is part of the word)
#   - Words that naturally end in ا: هدا، كذا
#
# Safe heuristic:
#   - Word must be ≥ 3 chars (so لا = 2 chars is safe)
#   - The ا must be preceded by a non-alef Arabic letter (catches tanwin-al case)
#   - We use a lookbehind that asserts at least 2 Arabic letters before the trailing ا
_TANWIN_ALEF_PATTERN = re.compile(
    r"(?<=[\u0600-\u06FF][\u0600-\u06FF])"  # preceded by ≥2 Arabic letters
    r"\u0627"                                # the trailing alef
    r"(?=\s|$)"                              # at word boundary
)

# Tatweel (kashida) — decorative elongation ـ
_TATWEEL = "\u0640"

# Alef variants → normalized alef
_ALEF_VARIANTS = {
    "\u0622": "\u0627",  # آ (alef with madda) → ا
    "\u0623": "\u0627",  # أ (alef with hamza above) → ا
    "\u0625": "\u0627",  # إ (alef with hamza below) → ا
    "\u0671": "\u0627",  # ٱ (alef wasla) → ا
    "\u0672": "\u0627",  # ٲ (alef with wavy hamza above) → ا
    "\u0673": "\u0627",  # ٳ (alef with wavy hamza below) → ا
    "\u0675": "\u0627",  # ٵ (high hamza alef) → ا
}

# Final ya / alef maqsurah normalization
_YA_MAQSURAH = "\u0649"   # ى (alef maqsurah)
_YA_NORMAL = "\u064A"      # ي (ya)

# Taa marbuta → haa
_TAA_MARBUTA = "\u0629"    # ة
_HAA = "\u0647"             # ه

_ZERO_WIDTH_TRANSLATION = str.maketrans(
    "",
    "",
    "\ufeff\u200b\u200c\u200d\u200e\u200f\u2066\u2067\u2068\u2069",
)
_WORD_PATTERN = re.compile(r"[\u0621-\u063A\u0641-\u064A0-9]+")
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_RETRIEVAL_BOILERPLATE_AR = {
    "اية",
    "ايات",
    "تفسير",
    "التفسير",
    "شرح",
    "معنى",
    "تاويل",
    "تأويل",
    "سورة",
    "سور",
    "تتحدث",
    "يتحدث",
    "تتكلم",
    "يتكلم",
    "تذكر",
    "يذكر",
    "تدل",
    "يدل",
    "تشير",
    "يشير",
    "تتناول",
    "يتناول",
    "تخص",
    "يتحدثون",
    "عن",
    "الى",
    "في",
    "من",
    "ما",
    "ماذا",
    "ماهي",
    "هي",
    "هل",
    "هناك",
    "اذكر",
    "اعطني",
    "هات",
    "ابحث",
    "ابحثلي",
    "اخبرني",
    "اريد",
    "نص",
}
_TAFSIR_QUERY_KEYWORDS_AR = {
    "تفسير",
    "التفسير",
    "شرح",
    "معنى",
    "تاويل",
    "تأويل",
}
_EXPLICIT_VERSE_WRAPPERS_AR = _RETRIEVAL_BOILERPLATE_AR | {
    "الاية",
    "الايه",
    "الايات",
    "الايات",
    "تفسيرها",
    "شرحها",
}
_RETRIEVAL_BOILERPLATE_EN = {
    "verse",
    "verses",
    "ayah",
    "ayat",
    "surah",
    "chapter",
    "about",
    "mention",
    "mentions",
    "mentioning",
    "talk",
    "talks",
    "talking",
    "speaks",
    "says",
    "what",
    "which",
    "show",
    "give",
    "find",
}
_ARABIC_SINGLE_LETTER_PREFIXES = ("و", "ف", "ب", "ك", "ل")
_ARABIC_PRONOUN_SUFFIXES = (
    "كما",
    "هما",
    "كم",
    "كن",
    "هم",
    "هن",
    "ها",
    "نا",
)


class ArabicTextNormalizer:
    """
    Production-grade Arabic text normalizer with multiple levels:
    
    - Level 0 (display):     Original text preserved with diacritics
    - Level 1 (embedding):   Light normalization (remove tashkeel, tatweel, normalize alef)
    - Level 2 (BM25 docs):   Conservative lexical normalization
    - Level 3 (BM25 query):  BM25 doc normalization + cautious alef folding
    """

    @staticmethod
    def remove_diacritics(text: str) -> str:
        """Remove all Arabic diacritical marks (tashkeel)."""
        return _DIACRITICS_PATTERN.sub("", text)

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize invisible Unicode artifacts that commonly leak from API payloads.

        Quran API payloads occasionally include BOM markers or bidi controls,
        especially in the first ayah of a surah.  They should never survive into
        the embedding or BM25 text because they fragment tokens invisibly.
        """
        text = unicodedata.normalize("NFKC", text)
        return text.translate(_ZERO_WIDTH_TRANSLATION)

    @staticmethod
    def remove_tatweel(text: str) -> str:
        """Remove kashida/tatweel elongation character."""
        return text.replace(_TATWEEL, "")

    @staticmethod
    def normalize_alef(text: str) -> str:
        """
        Normalize all alef variants (آ أ إ ٱ) to bare alef (ا).

        Also converts the Uthmani superscript alef (ٰ / \\u0670) to a full alef.
        In Uthmani script this character represents a long vowel 'ā', e.g.:
            ٱلْكِتَٰبُ  (Al-Kitābu)  should normalize to  الكتاب  not  الكتب

        NOTE: Some Uthmani words use \\u0670 decoratively (e.g. ذَٰلِكَ = ذلك in
        modern spelling).  The net effect after alef-normalization + tanwin-alef
        removal is that BM25 tokens will be ذالك vs ذلك.  This is an acceptable
        approximation: the semantically important words (الكتاب، الرحمن) are
        correctly preserved, and ذلك/ذالك will still score positively via BM25
        partial-match because most tokens in a query like "ذلك الكتاب لا ريب فيه"
        will fully match.
        """
        # Replace superscript alef (long-vowel marker) with full alef first
        text = text.replace(_SUPERSCRIPT_ALEF, _ALEF)
        # Then normalize all other alef variants
        for variant, replacement in _ALEF_VARIANTS.items():
            text = text.replace(variant, replacement)
        return text

    @staticmethod
    def normalize_ya(text: str) -> str:
        """Normalize alef maqsurah (ى) to ya (ي)."""
        return text.replace(_YA_MAQSURAH, _YA_NORMAL)

    @staticmethod
    def fix_uthmani_words(text: str) -> str:
        """
        Fix common Uthmani orthographic words that diverge from modern Arabic
        spelling after dagger-alef expansion.

        Examples: ذالك→ذلك, هاذا→هذا
        Applied only in the BM25 normalization path so that plain-Arabic queries
        (without diacritics) match the stored Uthmani texts correctly.
        """
        words = text.split()
        fixed = [_UTHMANI_WORD_MAP.get(w, w) for w in words]
        return " ".join(fixed)

    @staticmethod
    def normalize_taa_marbuta(text: str) -> str:
        """Normalize taa marbuta (ة) to haa (ه) — aggressive, use for BM25 only."""
        return text.replace(_TAA_MARBUTA, _HAA)

    @staticmethod
    def remove_tanwin_alef(text: str) -> str:
        """
        Remove the trailing alef that is left after stripping tanwin fathatan (ً).

        In Arabic, accusative indefinite nouns use tanwin fathatan written as ًا
        (diacritic ً + letter ا).  Removing only the diacritic leaves a naked ا
        attached to the word stem, causing BM25 token mismatch:
            query: مصر  ≠  indexed: مصرا

        This method removes that residual ا so both sides normalize to مصر.
        Applied AFTER diacritics removal, only in the BM25 normalization path.
        """
        return _TANWIN_ALEF_PATTERN.sub("", text)

    @staticmethod
    def remove_non_arabic_non_space(text: str) -> str:
        """Remove characters that are not Arabic, spaces, or basic punctuation."""
        # Keep: Arabic chars, spaces, basic Latin (for mixed text), digits
        return re.sub(r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s\w.,;:!?()0-9]", " ", text)

    @staticmethod
    def collapse_whitespace(text: str) -> str:
        """Collapse multiple spaces/newlines into single space."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def remove_bismillah(text: str) -> str:
        """
        Optionally remove Bismillah from start of surah text.
        Useful when Bismillah is already a separate ayah (Surah Al-Fatiha ayah 1).
        """
        bismillah_pattern = re.compile(
            r"^[\s]*بِسْمِ\s+اللَّهِ\s+الرَّحْمَٰنِ\s+الرَّحِيمِ[\s]*"
        )
        return bismillah_pattern.sub("", text).strip()

    @classmethod
    def normalize_for_embedding(cls, text: str) -> str:
        """
        Level 1 — Light normalization for embedding input.
        Removes diacritics and tatweel, normalizes alef.
        Preserves word structure for semantic meaning.
        """
        text = cls.normalize_unicode(text)
        text = cls.remove_diacritics(text)
        text = cls.remove_tatweel(text)
        text = cls.normalize_alef(text)
        text = cls.collapse_whitespace(text)
        return text

    @classmethod
    def normalize_for_bm25_document(cls, text: str) -> str:
        """
        Conservative document-side normalization used by the notebook corpus.

        Risky folds such as ة↔ه and ى↔ي stay disabled so canonical distinctions
        are preserved in lexical search while still stripping noisy diacritics.
        """
        text = cls.normalize_unicode(text)
        text = cls.remove_diacritics(text)
        text = cls.remove_tatweel(text)
        text = cls.collapse_whitespace(text)
        return text

    @classmethod
    def normalize_for_bm25_query(cls, text: str) -> str:
        """Query-side normalization with cautious alef-family folding."""
        text = cls.normalize_for_bm25_document(text)
        text = cls.normalize_alef(text)
        text = cls.collapse_whitespace(text)
        return text

    @classmethod
    def normalize_for_bm25(cls, text: str) -> str:
        """Backward-compatible alias for BM25 document normalization."""
        return cls.normalize_for_bm25_document(text)

    @classmethod
    def normalize_query(cls, query: str) -> str:
        """Normalize a user query for retrieval."""
        return cls.normalize_for_bm25_query(query)

    @classmethod
    def tokenize_document_for_bm25(cls, text: str) -> list[str]:
        """Tokenize stored BM25 document text and include lexical variants."""
        base_tokens = _WORD_PATTERN.findall(cls.normalize_for_bm25_document(text))
        output = []
        for token in base_tokens:
            for variant in cls.expand_retrieval_token_variants(token):
                if variant not in output:
                    output.append(variant)
        return output

    @classmethod
    def tokenize_query_for_bm25(cls, text: str) -> list[str]:
        """Tokenize normalized user queries for BM25 lookup."""
        base_tokens = _WORD_PATTERN.findall(cls.normalize_for_bm25_query(text))
        output = []
        for token in base_tokens:
            for variant in cls.expand_retrieval_token_variants(token):
                if variant not in output:
                    output.append(variant)
        return output

    @classmethod
    def tokenize_arabic(cls, text: str) -> list[str]:
        """
        Simple whitespace tokenizer for normalized Arabic text.
        For production, consider CAMeL Tools morphological analyzer.
        """
        return cls.tokenize_query_for_bm25(text)

    @classmethod
    def extract_retrieval_focus(cls, query: str) -> str:
        """
        Remove query-wrapper words so retrieval targets the actual topic.

        Example:
            "اية تتحدث عن الصبر" -> "الصبر"
        """
        detected_lang = cls.detect_language(query)

        if detected_lang == "ar":
            base_tokens = _WORD_PATTERN.findall(cls.normalize_for_bm25_query(query))
            focused_tokens = []
            for token in base_tokens:
                surface = cls._strip_arabic_single_letter_prefixes(token)
                canonical = cls._canonical_arabic_retrieval_token(surface)
                if canonical in _RETRIEVAL_BOILERPLATE_AR:
                    continue
                focused_tokens.append(canonical or surface)
            if focused_tokens:
                return " ".join(focused_tokens)
            return cls.normalize_for_bm25_query(query)

        normalized = cls.collapse_whitespace(query.lower())
        tokens = re.findall(r"[a-z0-9]+", normalized)
        focused_tokens = [
            token for token in tokens if token not in _RETRIEVAL_BOILERPLATE_EN
        ]
        if focused_tokens:
            return " ".join(focused_tokens)
        return normalized

    @classmethod
    def is_tafsir_query(cls, query: str) -> bool:
        tokens = _WORD_PATTERN.findall(cls.normalize_for_bm25_query(query))
        return any(token in _TAFSIR_QUERY_KEYWORDS_AR for token in tokens)

    @classmethod
    def extract_explicit_verse_text(cls, query: str) -> str:
        """
        Pull out the likely verse text from a tafsir-style question.

        Example:
            "ما تفسير قل هو الله احد؟" -> "قل هو الله احد"
        """
        tokens = _WORD_PATTERN.findall(cls.normalize_for_bm25_query(query))
        filtered = [token for token in tokens if token not in _EXPLICIT_VERSE_WRAPPERS_AR]
        return " ".join(filtered)

    @classmethod
    def expand_retrieval_token_variants(cls, token: str) -> list[str]:
        """
        Generate conservative lexical variants for BM25 and exact-token matching.

        The goal is to bridge small orthographic and inflectional differences such
        as:
          - وفضلها -> فضل
          - الصدقات -> صدقات / صدقة / صدق
          - بصدقة -> صدقة / صدق

        while keeping derivationally different words like الصادقين distinct from
        الصدقة.
        """
        token = cls.normalize_for_bm25_query(token)
        variants: list[str] = []

        def add(value: str) -> None:
            value = cls.collapse_whitespace(value)
            if len(value) > 1 and value not in variants:
                variants.append(value)

        add(token)
        stripped_prefix = cls._strip_arabic_single_letter_prefixes(token)
        add(stripped_prefix)

        without_article = stripped_prefix
        if without_article.startswith("ال") and len(without_article) > 4:
            without_article = without_article[2:]
            add(without_article)

        baseline_candidates = list(variants)
        for candidate in baseline_candidates:
            pronoun_stripped = cls._strip_arabic_pronoun_suffix(candidate)
            add(pronoun_stripped)

        baseline_candidates = list(variants)
        for candidate in baseline_candidates:
            singularized = cls._strip_arabic_number_suffix(candidate)
            add(singularized)
            if candidate.endswith("ات") and len(candidate) > 4:
                add(candidate[:-2] + "ة")

        return variants

    @classmethod
    def expand_anchor_token_variants(cls, token: str) -> list[str]:
        """
        Generate stricter anchor variants for the primary topic term.

        This keeps close nominal forms such as:
          - الصدقات -> صدقات / الصدقة / صدقة
        but avoids collapsing to bare stems like صدق that drift into different
        concepts such as الصادقين / صدقهم.
        """
        token = cls.normalize_for_bm25_query(token)
        variants: list[str] = []

        def add(value: str) -> None:
            value = cls.collapse_whitespace(value)
            if len(value) > 1 and value not in variants:
                variants.append(value)

        add(token)
        stripped_prefix = cls._strip_arabic_single_letter_prefixes(token)
        add(stripped_prefix)

        without_article = stripped_prefix
        if without_article.startswith("ال") and len(without_article) > 4:
            without_article = without_article[2:]
            add(without_article)

        baseline_candidates = list(variants)
        for candidate in baseline_candidates:
            pronoun_stripped = cls._strip_arabic_pronoun_suffix(candidate)
            add(pronoun_stripped)
            if pronoun_stripped.endswith("ات") and len(pronoun_stripped) > 4:
                add(pronoun_stripped[:-2] + "ة")

        return variants

    @classmethod
    def _strip_arabic_single_letter_prefixes(cls, token: str) -> str:
        token = cls.normalize_for_bm25_query(token)
        if len(token) <= 3:
            return token

        if token.startswith("و"):
            return token[1:]
        if token.startswith("ف") and len(token) > 4 and token[1:3] == "ال":
            return token[1:]
        if token[0] in {"ب", "ك", "ل"}:
            return token[1:]
        return token

    @classmethod
    def _strip_arabic_pronoun_suffix(cls, token: str) -> str:
        for suffix in _ARABIC_PRONOUN_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token

    @classmethod
    def _strip_arabic_number_suffix(cls, token: str) -> str:
        suffixes = ("ون", "ين", "ان")
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[: -len(suffix)]
        return token

    @classmethod
    def _canonical_arabic_retrieval_token(cls, token: str) -> str:
        token = cls._strip_arabic_single_letter_prefixes(token)
        token = cls._strip_arabic_pronoun_suffix(token)
        return token

    @classmethod
    def normalize_ayah_ref(cls, query: str) -> Optional[str]:
        """
        Parse common ayah reference forms such as 2:255 or ٢:٢٥٥.
        """
        compact = cls.normalize_unicode(query).translate(_ARABIC_INDIC_DIGITS)
        compact = cls.collapse_whitespace(compact)
        match = re.search(r"\b(\d{1,3})\s*[:\-/]\s*(\d{1,3})\b", compact)
        if not match:
            return None
        return f"{int(match.group(1))}:{int(match.group(2))}"

    @classmethod
    def detect_language(cls, text: str) -> str:
        """
        Simple heuristic language detection: Arabic vs English.
        Counts Arabic characters vs Latin characters.
        """
        arabic_count = len(re.findall(r"[\u0600-\u06FF]", text))
        latin_count = len(re.findall(r"[a-zA-Z]", text))

        if arabic_count > latin_count:
            return "ar"
        elif latin_count > arabic_count:
            return "en"
        else:
            return "ar"  # Default to Arabic for Islamic content
