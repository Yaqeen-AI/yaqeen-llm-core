"""
Context compressor — fits retrieved documents into the LLM's context window
while preserving document boundaries and source metadata for proper citations.

Uses sentence-aware truncation instead of raw token cutting.
"""

import re
import logging
from langchain_core.documents import Document
from state import AgentState

logger = logging.getLogger("shared.compressor")

# Maximum tokens to feed to the writer LLM as context
MAX_CONTEXT_TOKENS = 750

# Sentence-ending patterns for clean truncation
_SENTENCE_END = re.compile(r'[.!?؟]\s+')


def _truncate_at_sentence(text: str, max_tokens: int, tokenizer) -> str:
    """Truncate text to fit max_tokens, ending at a valid sentence boundary."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    match = None
    for m in _SENTENCE_END.finditer(truncated):
        match = m

    if match and match.end() > len(truncated) // 3:
        return truncated[:match.end()]
    return truncated


def compression_node(state: AgentState):
    """
    Compress retrieved context to fit within token budget.
    """
    docs = state.get("retrieved_context", [])
    if not docs:
        return {"retrieved_context": []}

    try:
        from shared_nodes.models.llm_loader import get_tokenizer
        tokenizer = get_tokenizer()
        if tokenizer is None:
            raise ValueError("get_tokenizer() returned None")
    except Exception as e:
        logger.warning(f"Tokenizer unavailable ({e}), using fallback")
        print(f"   [Compressor] -> Tokenizer unavailable ({e}), using fallback tokenizer.")
        try:
            from shared_nodes.models.llm_loader import HeuristicTokenizer
            tokenizer = HeuristicTokenizer()
        except Exception:
            class _FallbackTokenizer:
                def encode(self, t): return t.split()
                def decode(self, tokens, **kw): return " ".join(tokens)
            tokenizer = _FallbackTokenizer()

    compressed_docs = []
    tokens_used = 0

    for doc in docs:
        content = doc.page_content
        block = f"[المصدر: {doc.metadata.get('source', 'Unknown')}]\n{content}"
        block_tokens = len(tokenizer.encode(block))

        if tokens_used + block_tokens <= MAX_CONTEXT_TOKENS:
            compressed_docs.append(doc)
            tokens_used += block_tokens
        else:
            remaining_budget = MAX_CONTEXT_TOKENS - tokens_used
            if remaining_budget > 50:
                truncated_content = _truncate_at_sentence(content, remaining_budget - 20, tokenizer)
                if truncated_content.strip():
                    compressed_docs.append(Document(
                        page_content=truncated_content + "\n[...truncated]",
                        metadata={**doc.metadata, "truncated": True},
                    ))
            break

    if not compressed_docs:
        first = docs[0]
        truncated = _truncate_at_sentence(first.page_content, MAX_CONTEXT_TOKENS - 50, tokenizer)
        compressed_docs = [Document(
            page_content=truncated + "\n[...truncated]",
            metadata={**first.metadata, "truncated": True},
        )]
        tokens_used = len(tokenizer.encode(truncated))

    print(f"   [Compressor] -> {len(docs)} docs in, {len(compressed_docs)} docs out, "
          f"~{tokens_used} tokens used (budget: {MAX_CONTEXT_TOKENS})")

    return {"retrieved_context": compressed_docs}
