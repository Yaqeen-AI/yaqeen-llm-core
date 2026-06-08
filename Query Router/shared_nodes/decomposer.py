"""
Keyword-based query decomposer for multi-query routing.

Splits complex, multi-part user questions into focused sub-queries,
then routes each sub-query independently through the keyword matcher.
Runs in microseconds — no LLM inference.
"""

import re
from state import AgentState

try:
    from supervisor import (
        _tokenize,
        QURAN_KEYWORDS,
        HADITH_KEYWORDS,
        FIQH_KEYWORDS,
        GREETING_PATTERNS,
    )
except ImportError:
    from router import (
        _tokenize,
        QURAN_KEYWORDS,
        HADITH_KEYWORDS,
        FIQH_KEYWORDS,
        GREETING_PATTERNS,
    )

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


def _route_sub_query(tokens: set[str]) -> list[str]:
    """Determine which agents a sub-query's tokens match."""
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
    Returns a list of cleaned sub-query strings.
    """
    parts = _SPLIT_PATTERN.split(question)
    sub_queries = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        word_count = len(cleaned.split())
        if word_count < _MIN_SUB_QUERY_TOKENS:
            continue
        sub_queries.append(cleaned)

    return sub_queries if sub_queries else [question]


def decomposer_node(state: AgentState):
    """
    Decompose a complex query into sub-queries and route each independently.
    """
    question = state["question"]
    existing_agents = state.get("selected_agents", [])

    if existing_agents == ["direct_answer"]:
        return {
            "sub_queries": [question],
            "sub_query_agents": {question: ["direct_answer"]},
        }

    sub_queries = _split_query(question)

    if len(sub_queries) <= 1:
        return {
            "sub_queries": [question],
            "sub_query_agents": {question: existing_agents},
        }

    print(f"   [Decomposer] -> Split into {len(sub_queries)} sub-queries:")
    sub_query_agents = {}
    all_agents = set()

    for i, sq in enumerate(sub_queries, 1):
        tokens = _tokenize(sq)
        agents = _route_sub_query(tokens)

        if not agents:
            agents = ["quran_agent", "hadith_agent", "fiqh_agent"]

        sub_query_agents[sq] = agents
        all_agents.update(agents)
        agent_str = ", ".join(agents)
        print(f"      [{i}] '{sq[:50]}...' -> [{agent_str}]")

    domain_agents = all_agents - {"direct_answer"}
    if domain_agents:
        all_agents.discard("direct_answer")

    merged_agents = sorted(all_agents)

    return {
        "sub_queries": sub_queries,
        "sub_query_agents": sub_query_agents,
        "selected_agents": merged_agents,
    }
