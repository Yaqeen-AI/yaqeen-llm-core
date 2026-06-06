"""
AIS Suppression Module (Phase 4).

Implements a network suppression mechanism to deduplicate chunks
from both initial and secondary retrieval phases to maximize context diversity.
"""

import os
import logging
from typing import List, Set
from langchain_core.documents import Document
from state import AgentState

logger = logging.getLogger("ais.suppression")

def tokenize_to_set(text: str) -> Set[str]:
    """Tokenize text into lowercase alphanumeric word tokens, filtering short words."""
    text = text.lower().strip()
    words = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text).split()
    return {w for w in words if len(w) > 2}

import re

def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return float(intersection) / union

def suppression_node(state: AgentState) -> dict:
    """
    Suppression Node (Phase 4):
    1. Gathers chunks from both initial clones and secondary retrieval.
    2. Performs Jaccard-based deduplication using SUPPRESSION_SIMILARITY_CUTOFF.
    3. Keeps only diverse chunks to maximize context richness.
    """
    clones = state.get("clones", [])
    secondary_context = state.get("secondary_context", [])
    
    # Pool all candidate documents
    # Give priority to Clones (high-affinity selection)
    candidates: List[Document] = list(clones)
    
    for doc in secondary_context:
        # Avoid duplicate objects
        if doc not in candidates:
            candidates.append(doc)
            
    if not candidates:
        print("   [Suppression] -> No candidate chunks to suppress.")
        return {"suppressed_context": []}

    # Load configurable threshold (default to 0.70)
    try:
        cutoff = float(os.getenv("SUPPRESSION_SIMILARITY_CUTOFF", "0.70"))
    except ValueError:
        cutoff = 0.70
        
    print(f"   [Suppression] -> Regulatory suppression threshold set to {cutoff:.2f}")

    # Prepare token sets for each document
    token_sets = [tokenize_to_set(doc.page_content) for doc in candidates]
    
    suppressed_context: List[Document] = []
    
    for i, doc in enumerate(candidates):
        set_i = token_sets[i]
        
        # Check if doc is a near-duplicate of any already approved document
        is_duplicate = False
        for approved_doc in suppressed_context:
            set_approved = tokenize_to_set(approved_doc.page_content)
            sim = compute_jaccard_similarity(set_i, set_approved)
            
            if sim >= cutoff:
                is_duplicate = True
                logger.info(f"Suppressed duplicate document (Jaccard similarity: {sim:.3f})")
                break
                
        if not is_duplicate:
            suppressed_context.append(doc)
            
    print(
        f"   [Suppression] -> Suppressed {len(candidates) - len(suppressed_context)} chunks. "
        f"Approved {len(suppressed_context)} unique context antibodies."
    )
    for i, doc in enumerate(suppressed_context, 1):
        source = doc.metadata.get("source", "Unknown")
        print(f"      [{i}] [{source}] '{doc.page_content[:65]}...'")

    return {
        "suppressed_context": suppressed_context
    }
