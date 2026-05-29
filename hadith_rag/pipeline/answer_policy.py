import re
from enum import Enum


class AnswerIntent(Enum):
    EXPLANATORY = "explanatory"
    VERIFICATION = "verification"
    COLLECTION = "collection"
    LOOKUP = "lookup"


_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]+")
_TATWEEL = re.compile(r"\u0640+")
_WHITESPACE = re.compile(r"\s+")
_ALEF_VARIANTS = re.compile(r"[أإآٱ]")

_VERIFICATION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ما\s+صح(?:ة)?",
        r"ما\s+درجة",
        r"هل\s+صح",
        r"حكم\s+حديث",
        r"تصحيح",
        r"تضعيف",
        r"authentic",
        r"authenticity",
        r"\bgrade\b",
        r"\bweak\b",
        r"\bfabricated\b",
    )
]

_COLLECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bاحاديث\b",
        r"\bالأحاديث\b",
        r"\bجميع\b",
        r"\bكل\b.*\bاحاديث\b",
        r"\bاجمع\b",
        r"\bاذكر\b.*\bحديث\b.*\b(?:عن|في|حول|بخصوص)\b",
        r"\bاذكر\b.*\bاحاديث\b",
        r"\bهات\b.*\bحديث\b.*\b(?:عن|في|حول|بخصوص)\b",
        r"\bهات\b.*\bاحاديث\b",
        r"\bاعطني\b.*\bحديث\b.*\b(?:عن|في|حول|بخصوص)\b",
        r"\blist\b",
        r"\bcollection\b",
        r"\ball narrations\b",
    )
]

_EXPLANATORY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bفضل\b",
        r"\bفضائل\b",
        r"\bفوائد\b",
        r"\bاهمية\b",
        r"\bأهمية\b",
        r"\bحكمه\b",
        r"\bحكمة\b",
        r"\bbenefits?\b",
        r"\bvirtues?\b",
        r"\bimportance\b",
    )
]

_SPECIFIC_HADITH_MARKERS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bحديث\b",
        r"قال\s+(رسول\s+الله|النبي)",
        r"ما\s+معنى\s+حديث",
        r"اشرح\s+حديث",
        r"فسر\s+حديث",
        r"فسر\s+لي\s+حديث",
        r"من\s+رواه",
        r"في\s+اي\s+كتاب",
        r"في\s+أي\s+كتاب",
        r"هل\s+(?:قال|ذكر)\s+(?:النبي|رسول\s+الله)",
        r"«|\"",
    )
]

# Refusal message returned when no context chunks are available.
_NO_CONTEXT_REPLY = "لا تتوفر معلومات كافية في المصادر المتاحة للإجابة على هذا السؤال."


def _normalize(text: str) -> str:
    text = _TASHKEEL.sub("", str(text or "").strip().lower())
    text = _TATWEEL.sub("", text)
    text = _ALEF_VARIANTS.sub("ا", text)
    return _WHITESPACE.sub(" ", text).strip()


def _looks_like_specific_hadith_request(normalized: str) -> bool:
    """Return True when the user appears to be asking about a particular hadith text."""
    return any(pattern.search(normalized) for pattern in _SPECIFIC_HADITH_MARKERS)


def classify_answer_intent(
    query: str,
    query_type: str = "general",
    metadata_fields: list[str] | None = None,
) -> AnswerIntent:
    """
    Map the query into one of four answer-policy categories.

    This sits above the lower-level query classifier used for retrieval.
    """
    normalized = _normalize(query)
    metadata_fields = set(metadata_fields or [])

    if query_type == "ruling" or "grade" in metadata_fields:
        return AnswerIntent.VERIFICATION

    if any(pattern.search(normalized) for pattern in _VERIFICATION_PATTERNS):
        return AnswerIntent.VERIFICATION

    if query_type == "narrator":
        return AnswerIntent.COLLECTION

    if any(pattern.search(normalized) for pattern in _COLLECTION_PATTERNS):
        return AnswerIntent.COLLECTION

    if any(pattern.search(normalized) for pattern in _EXPLANATORY_PATTERNS):
        return AnswerIntent.EXPLANATORY

    if query_type in {"explain_hadith", "topic", "general"}:
        if _looks_like_specific_hadith_request(normalized):
            return AnswerIntent.LOOKUP
        return AnswerIntent.EXPLANATORY

    if query_type in {"hadith_lookup", "metadata"}:
        return AnswerIntent.LOOKUP

    return AnswerIntent.EXPLANATORY


def check_context(chunks: list) -> str | None:
    """
    Call this in generate.py BEFORE sending anything to the LLM.

    Returns a ready-made refusal string when there are no retrieved chunks,
    so the LLM is never invoked and cannot hallucinate.
    Returns None when chunks are present and generation should proceed normally.

    Usage in generate.py:
        refusal = check_context(retrieved_chunks)
        if refusal:
            return refusal
        # ... proceed to call LLM
    """
    if not chunks:
        return _NO_CONTEXT_REPLY
    return None


def grade_priority(grade: str) -> int:
    """Smaller value means higher display priority."""
    return {
        "sahih": 0,
        "hasan": 1,
        "daif": 2,
        "mawdu": 3,
        "unknown": 4,
    }.get(str(grade or "").strip(), 4)
