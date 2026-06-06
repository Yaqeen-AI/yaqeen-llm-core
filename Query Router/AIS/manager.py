"""
AIS Manager (Task Decomposition).

Decomposes the incoming antigen (User Query) into parallelizable sub-queries.
No LLM inference is used for decomposition to ensure sub-millisecond execution.
"""

import re
import logging
from state import AgentState

logger = logging.getLogger("ais.manager")

# Splitting patterns for Arabic & English conjunctions, punctuation, and numbering
_SPLIT_PATTERN = re.compile(
    r"""
    (?:
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

def split_query(question: str) -> list:
    """Split a complex question into candidate sub-queries."""
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

def manager_node(state: AgentState) -> dict:
    """
    Manager node: Receives user query (antigen) and decomposes it.
    """
    question = state["question"]
    sub_queries = split_query(question)

    logger.info(f"Antigen decomposed into {len(sub_queries)} sub-queries.")
    print(f"   [Manager] -> Decomposed antigen into {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries, 1):
        print(f"      [{i}] '{sq[:60]}...'")

    return {
        "sub_queries": sub_queries,
        "loop_step": state.get("loop_step", 0) + 1
    }
