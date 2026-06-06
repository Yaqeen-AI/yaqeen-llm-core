"""
Manager — Task Decomposition node.

Receives an unfulfilled query (cache MISS) and decomposes it into
parallelizable sub-queries using regex-based splitting.  Each sub-query
will be assigned to its own isolated Colony.

No LLM inference — runs in microseconds.
"""

import re
import logging
from state import AgentState
from colony import tokenize, is_greeting

logger = logging.getLogger("mmas.manager")

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

_MIN_SUB_QUERY_TOKENS = 2


def _split_query(question: str) -> list:
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


# ═══════════════════════════════════════════════════════════════════════════════
# Manager Graph Node
# ═══════════════════════════════════════════════════════════════════════════════

def manager_node(state: AgentState):
    """
    Manager: Decompose the original query into parallelizable sub-queries.

    If the query is a greeting, it passes through as a single sub-query.
    Complex multi-part questions are split into focused sub-queries,
    each of which will get its own Colony.

    Sets:
      - sub_queries: list of sub-query strings
      - loop_step: incremented by 1
    """
    question = state["question"]
    tokens = tokenize(question)

    # ── Quick path: greetings ──
    if is_greeting(tokens):
        logger.info(f"[Manager] Greeting detected: '{question[:40]}...'")
        print(f"   [Manager] -> Greeting detected, single sub-query")
        return {
            "sub_queries": [question],
            "loop_step": state.get("loop_step", 0) + 1,
        }

    # ── Decompose ──
    sub_queries = _split_query(question)

    if len(sub_queries) <= 1:
        logger.info(f"[Manager] Single query (no decomposition): '{question[:40]}...'")
        print(f"   [Manager] -> Single query, no decomposition needed")
    else:
        logger.info(
            f"[Manager] Decomposed into {len(sub_queries)} sub-queries: "
            f"{[sq[:30] for sq in sub_queries]}"
        )
        print(f"   [Manager] -> Decomposed into {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"      [{i}] '{sq[:60]}...'")

    return {
        "sub_queries": sub_queries,
        "loop_step": state.get("loop_step", 0) + 1,
    }
