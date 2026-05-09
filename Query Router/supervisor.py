"""
Keyword-based multi-query router.

Replaces LLM-based routing with deterministic keyword matching.
Runs in microseconds — no model inference for routing decisions.
Supports selecting MULTIPLE agents for cross-domain queries.
"""

import re
from state import AgentState

# ═══════════════════════════════════════════════════════════════════════════════
# Keyword sets — Arabic + English, covering common query patterns per domain
# ═══════════════════════════════════════════════════════════════════════════════

QURAN_KEYWORDS = {
    # English
    "quran", "quranic", "ayah", "ayat", "surah", "sura", "verse", "verses",
    "tafsir", "revelation", "juz", "hizb", "makkah", "madinah", "recitation",
    "tilawah", "mushaf", "makki", "madani",
    # Arabic
    "قرآن", "قرآنية", "قرآني", "آية", "آيات", "سورة", "سور",
    "تفسير", "وحي", "تلاوة", "مصحف", "جزء", "حزب", "تنزيل",
    "مكية", "مدنية", "الكتاب",
}

HADITH_KEYWORDS = {
    # English
    "hadith", "hadiths", "hadeeth", "prophet", "prophetic", "sunnah",
    "narration", "narrated", "narrator", "sahih", "bukhari", "muslim",
    "tirmidhi", "dawud", "nasa'i", "ibn majah", "musnad", "isnad",
    "chain", "rawi", "muhaddith", "athar", "sanad",
    # Arabic
    "حديث", "أحاديث", "نبي", "النبي", "الرسول", "رسول", "نبوي",
    "سنة", "سنن", "رواية", "راوي", "رواة", "صحيح", "بخاري",
    "مسلم", "ترمذي", "داود", "نسائي", "ماجه", "مسند", "إسناد",
    "سند", "محدث", "أثر", "آثار",
}

FIQH_KEYWORDS = {
    # English
    "fiqh", "ruling", "rulings", "halal", "haram", "fatwa", "fatwas",
    "jurisprudence", "sharia", "shariah", "madhab", "madhhab", "hanafi",
    "maliki", "shafii", "hanbali", "wajib", "obligatory", "mustahab",
    "recommended", "makruh", "disliked", "mubah", "permissible",
    "forbidden", "ibadah", "worship", "muamalat", "prayer", "salah",
    "zakat", "fasting", "hajj", "nikah", "talaq", "inheritance",
    "purification", "wudu", "ghusl", "tayammum",
    # Arabic
    "فقه", "فقهي", "فقهية", "حكم", "أحكام", "حلال", "حرام",
    "فتوى", "فتاوى", "شريعة", "مذهب", "مذاهب", "حنفي", "مالكي",
    "شافعي", "حنبلي", "واجب", "مستحب", "مكروه", "مباح",
    "عبادة", "عبادات", "معاملات", "صلاة", "زكاة", "صيام", "صوم",
    "حج", "نكاح", "طلاق", "ميراث", "إرث", "فرائض",
    "طهارة", "وضوء", "غسل", "تيمم", "الموسوعة",
}

GREETING_PATTERNS = {
    # English
    "hello", "hi", "hey", "good morning", "good evening", "good night",
    "thanks", "thank you", "bye", "goodbye", "who are you", "what are you",
    "help",
    # Arabic
    "سلام", "السلام", "مرحبا", "أهلا", "شكرا", "جزاك", "بارك",
    "وداعا", "صباح", "مساء", "من أنت", "ما أنت", "مساعدة",
}


def _tokenize(text: str) -> set[str]:
    """Split query into lowercase word tokens + bigrams for phrase matching."""
    text = text.lower().strip()
    # Remove punctuation except Arabic characters
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    words = text.split()
    tokens = set(words)
    # Add bigrams for multi-word keywords (e.g., "ibn majah", "good morning")
    for i in range(len(words) - 1):
        tokens.add(f"{words[i]} {words[i+1]}")
    return tokens


def supervisor_node(state: AgentState):
    """
    Fast keyword-based multi-query router.

    Returns a list of selected agents based on keyword matches.
    Runs in microseconds — no LLM call.
    """
    query = state["question"]
    tokens = _tokenize(query)

    # ── Check for greetings / general queries first ──
    if tokens & GREETING_PATTERNS:
        return {
            "selected_agents": ["direct_answer"],
            "current_agent": "direct_answer",
            "loop_step": state.get("loop_step", 0) + 1,
        }

    # ── Match against domain keyword sets ──
    agents = []

    if tokens & QURAN_KEYWORDS:
        agents.append("quran_agent")

    if tokens & HADITH_KEYWORDS:
        agents.append("hadith_agent")

    if tokens & FIQH_KEYWORDS:
        agents.append("fiqh_agent")

    # ── Fallback: if no keywords matched, search all sources ──
    if not agents:
        agents = ["quran_agent", "hadith_agent", "fiqh_agent"]

    return {
        "selected_agents": agents,
        "current_agent": agents[0],  # backward-compat: first agent as primary
        "loop_step": state.get("loop_step", 0) + 1,
    }