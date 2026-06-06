"""
AIS Maturation Module (Phase 3).

1. Extracts semantic signals/key entities from Clones.
2. Rewrites and expands the initial sub-queries using these signals.
3. Executes a secondary retrieval pass using these matured sub-queries.
"""

import asyncio
import logging
import re
from typing import List, Set
from langchain_core.documents import Document
from state import AgentState
from dispatcher import route_sub_query, _run_worker_async

logger = logging.getLogger("ais.maturation")

# Basic Stopwords for Arabic & English
STOPWORDS = {
    # English
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "once", "here", "there",
    "what", "where", "who", "which", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "can", "will", "just", "should",
    "now", "ruling", "rulings", "say", "says", "about", "what",
    
    # Arabic
    "من", "إلى", "عن", "على", "في", "مع", "ثم", "أو", "أم", "بل", "لا", "ما", "من",
    "ذا", "هو", "هي", "هم", "هن", "هذا", "هذه", "ذلك", "الذين", "التي", "الذي",
    "كان", "كانت", "يكون", "أن", "إن", "هل", "كيف", "لماذا", "أين", "متى", "كم",
    "كل", "بعض", "غير", "فوق", "تحت", "قبل", "بعد", "مع", "عند", "بين", "منذ",
    "حكم", "أحكام", "ماذا", "قول", "يقول", "عنه", "عنها", "فيها", "فيه", "عليه",
}

def extract_semantic_signals(clones: List[Document], top_n: int = 5) -> List[str]:
    """
    Extract key content-rich terms from the selected clones to act as expansion signals.
    """
    word_counts = {}
    for doc in clones:
        # Simple tokenization
        text = doc.page_content.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
        words = text.split()
        
        for word in words:
            # Filter by word length and stopwords
            if len(word) >= 3 and word not in STOPWORDS and not word.isdigit():
                word_counts[word] = word_counts.get(word, 0) + 1
                
    # Sort and pick top_n terms
    sorted_terms = sorted(word_counts.items(), key=lambda x: -x[1])
    return [term for term, count in sorted_terms[:top_n]]

async def maturation_node(state: AgentState) -> dict:
    """
    Maturation Node (Phase 3):
    1. Extracts semantic signals from the clones.
    2. Rewrites the initial sub-queries to expand them.
    3. Runs a secondary targeted retrieval pass using these expanded queries.
    """
    clones = state.get("clones", [])
    sub_queries = state.get("sub_queries", [state["question"]])
    
    if not clones:
        print("   [Maturation] -> No clones found for maturation. Skipping secondary retrieval.")
        return {
            "matured_sub_queries": sub_queries,
            "secondary_context": []
        }
        
    # Step 1: Extract semantic signals
    signals = extract_semantic_signals(clones, top_n=5)
    print(f"   [Maturation] -> Extracted semantic signals: {signals}")
    
    # Step 2: Expand sub-queries
    matured_sub_queries = []
    for sq in sub_queries:
        if signals:
            expanded = f"{sq} " + " ".join(signals)
        else:
            expanded = sq
        matured_sub_queries.append(expanded)
        
    print("   [Maturation] -> Matured sub-queries:")
    for i, msq in enumerate(matured_sub_queries, 1):
        print(f"      [{i}] '{msq[:70]}...'")
        
    # Step 3: Execute secondary retrieval pass
    tasks = []
    for msq in matured_sub_queries:
        workers = route_sub_query(msq)
        for w in workers:
            tasks.append(_run_worker_async(w, msq))
            
    print(f"   [Maturation] -> Executing secondary retrieval with {len(tasks)} parallel workers...")
    
    try:
        # Run with a 15-second timeout to handle potential API hangs
        secondary_results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=15.0)
        
        secondary_context = []
        for docs in secondary_results:
            secondary_context.extend(docs)
            
        print(f"   [Maturation] -> Retrieved {len(secondary_context)} matured antibodies")
        
    except Exception as e:
        # Graceful fallback to initial clones if secondary retrieval fails or times out
        logger.error(f"Secondary retrieval failed or timed out: {e}. Falling back to clonal selection clones.")
        print(f"   [Maturation] -> WARNING: Secondary retrieval failed ({e}). Falling back to clones.")
        secondary_context = []
        
    return {
        "matured_sub_queries": matured_sub_queries,
        "secondary_context": secondary_context
    }
