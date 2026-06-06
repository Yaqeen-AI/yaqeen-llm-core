"""
Unified keyword-based query router.

Replaces the old supervisor + decomposer two-step pipeline with a single
node that handles both routing AND decomposition in one pass:

  1. Split compound questions into sub-queries (regex-based, μs)
  2. Route each sub-query to domain agents via keyword matching (μs)
  3. Return the merged agent list + per-sub-query routing map

No LLM inference — runs in microseconds.
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

# ═══════════════════════════════════════════════════════════════════════════════
# Splitting patterns — Arabic + English conjunctions, punctuation, numbering
# ═══════════════════════════════════════════════════════════════════════════════

_SPLIT_PATTERN = re.compile(
    r"""
    (?:                          # Non-capturing group for alternation
        \s*[؟?]\s*             |  # Question marks (Arabic + English)
        \s*،\s*                |  # Arabic comma
        \s+و(?:كذلك|أيضا|أيضاً)?\s+  |  # Arabic standalone 'and' / 'and also'
        \s+و(?=ما|ماذا|هل|كيف|لماذا|أين|متى|من) | # Arabic attached 'waw' before question words
        \s+ثم\s+              |  # Arabic 'then'
        \s+and\s+              |  # English 'and'
        \s+also\s+             |  # English 'also'
        \s+additionally\s+     |  # English 'additionally'
        \s*\n\s*               |  # Newlines
        \s*\d+[.)]\s+            # Numbered lists: "1. " or "1) "
    )
    """,
    re.VERBOSE | re.UNICODE,
)

# Minimum token count for a sub-query to be considered meaningful
_MIN_SUB_QUERY_TOKENS = 2


# ═══════════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════════

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


def _route_tokens(tokens: set[str]) -> list[str]:
    """Determine which agents a token set matches."""
    # Check greeting first
    if tokens & GREETING_PATTERNS:
        return ["direct_answer"]

    agents = []
    if tokens & QURAN_KEYWORDS:
        agents.append("quran_agent")
    if tokens & HADITH_KEYWORDS:
        agents.append("hadith_agent")
    if tokens & FIQH_KEYWORDS:
        agents.append("fiqh_agent")

    return agents


def _split_query(question: str) -> list[str]:
    """
    Split a complex question into candidate sub-queries.

    Returns a list of cleaned sub-query strings.  If no split points
    are found, returns [question] unchanged.
    """
    parts = _SPLIT_PATTERN.split(question)
    # Clean up and filter out empty / too-short fragments
    sub_queries = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        # Skip fragments that are too short to be meaningful
        word_count = len(cleaned.split())
        if word_count < _MIN_SUB_QUERY_TOKENS:
            continue
        sub_queries.append(cleaned)

    return sub_queries if sub_queries else [question]


# ═══════════════════════════════════════════════════════════════════════════════
# Graph node
# ═══════════════════════════════════════════════════════════════════════════════

def router_node(state: AgentState):
    """
    Single-pass keyword router: split → route → done.

    Replaces the old supervisor + decomposer two-node pipeline.
    Handles both simple queries (single agent) and compound queries
    (multiple sub-queries, each routed independently).

    Sets:
      - sub_queries: list of sub-query strings
      - sub_query_agents: dict mapping each sub-query to its agent list
      - selected_agents: merged unique list of all agents across sub-queries
      - current_agent: first agent (backward compat)
      - loop_step: incremented by 1
    """
    question = state["question"]
    tokens = _tokenize(question)

    # ── Quick path: greetings / general queries ──
    if tokens & GREETING_PATTERNS:
        print(f"   [Router] -> Greeting detected, routing to direct_answer")
        return {
            "selected_agents": ["direct_answer"],
            "current_agent": "direct_answer",
            "sub_queries": [question],
            "sub_query_agents": {question: ["direct_answer"]},
            "loop_step": state.get("loop_step", 0) + 1,
        }

    # ── Split compound query ──
    sub_queries = _split_query(question)

    # ── Single query (no decomposition) ──
    if len(sub_queries) <= 1:
        agents = _route_tokens(tokens)
        if not agents:
            # Fallback: search all sources
            agents = ["quran_agent", "hadith_agent", "fiqh_agent"]

        print(f"   [Router] -> Single query, agents: [{', '.join(agents)}]")
        return {
            "selected_agents": agents,
            "current_agent": agents[0],
            "sub_queries": [question],
            "sub_query_agents": {question: agents},
            "loop_step": state.get("loop_step", 0) + 1,
        }

    # ── Multi-query: route each sub-query independently ──
    print(f"   [Router] -> Split into {len(sub_queries)} sub-queries:")
    sub_query_agents = {}
    all_agents = set()

    for i, sq in enumerate(sub_queries, 1):
        sq_tokens = _tokenize(sq)
        agents = _route_tokens(sq_tokens)

        if not agents:
            # No keywords match for this sub-query → use all domain agents
            agents = ["quran_agent", "hadith_agent", "fiqh_agent"]

        sub_query_agents[sq] = agents
        all_agents.update(agents)
        print(f"      [{i}] '{sq[:50]}...' -> [{', '.join(agents)}]")

    # Remove direct_answer if any domain agent is also selected
    domain_agents = all_agents - {"direct_answer"}
    if domain_agents:
        all_agents.discard("direct_answer")

    merged_agents = sorted(all_agents)

    return {
        "selected_agents": merged_agents,
        "current_agent": merged_agents[0],
        "sub_queries": list(sub_query_agents.keys()),
        "sub_query_agents": sub_query_agents,
        "loop_step": state.get("loop_step", 0) + 1,
    }
