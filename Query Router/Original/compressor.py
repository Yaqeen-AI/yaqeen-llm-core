"""
Context compressor — fits retrieved documents into the LLM's context window
while preserving document boundaries and source metadata for proper citations.

Uses sentence-aware truncation instead of raw token cutting.
"""

import re
from langchain_core.documents import Document
from state import AgentState
# Lazy import of tokenizer – will be performed inside compression_node

# Maximum tokens to feed to the writer LLM as context
MAX_CONTEXT_TOKENS = 750

# Sentence-ending patterns for clean truncation
_SENTENCE_END = re.compile(r'[.。!؟?\n]')


def _truncate_at_sentence(text: str, max_tokens: int, tokenizer) -> str:
    """Truncate text at the last complete sentence within the token budget."""
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    # Decode the truncated tokens
    truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)

    # Find the last sentence boundary
    match = None
    for m in _SENTENCE_END.finditer(truncated):
        match = m

    if match and match.end() > len(truncated) // 3:
        # Cut at last sentence end (only if we keep at least 1/3 of the text)
        return truncated[:match.end()]

    return truncated


def compression_node(state: AgentState):
    """
    Compress retrieved context to fit within token budget.

    Preserves individual document boundaries and source metadata
    so the writer can produce proper citations.
    """
    docs = state.get("retrieved_context", [])
    if not docs:
        return {"retrieved_context": []}

    # Lazy import tokenizer – falls back to a simple whitespace tokenizer if torch is unavailable
    try:
        from models.llm_loader import get_tokenizer
        tokenizer = get_tokenizer()
    except Exception as e:
        print(f"   [Compressor] -> Tokenizer unavailable ({e}), using fallback tokenizer.")
        class _FallbackTokenizer:
            def encode(self, text):
                return text.split()
            def decode(self, tokens, skip_special_tokens=True):
                return " ".join(tokens)
        tokenizer = _FallbackTokenizer()

    # ── Build context blocks preserving source boundaries ──
    compressed_docs = []
    tokens_used = 0

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content.strip()
        block = f"[Source: {source}]\n{content}"

        block_tokens = len(tokenizer.encode(block))

        if tokens_used + block_tokens <= MAX_CONTEXT_TOKENS:
            # Entire document fits
            compressed_docs.append(doc)
            tokens_used += block_tokens
        else:
            # Partial fit — truncate at sentence boundary
            remaining_budget = MAX_CONTEXT_TOKENS - tokens_used
            if remaining_budget > 50:  # Only include if we can fit something meaningful
                truncated_content = _truncate_at_sentence(content, remaining_budget - 20, tokenizer)
                if truncated_content.strip():
                    truncated_doc = Document(
                        page_content=truncated_content + "\n[...truncated]",
                        metadata={**doc.metadata, "truncated": True},
                    )
                    compressed_docs.append(truncated_doc)
            break  # Budget exhausted

    if not compressed_docs:
        # Edge case: even the first doc exceeds budget, force-truncate it
        first = docs[0]
        truncated = _truncate_at_sentence(first.page_content, MAX_CONTEXT_TOKENS - 50, tokenizer)
        compressed_docs = [Document(
            page_content=truncated + "\n[...truncated]",
            metadata={**first.metadata, "truncated": True},
        )]

    print(f"   [Compressor] -> {len(docs)} docs in, {len(compressed_docs)} docs out, "
          f"~{tokens_used} tokens used (budget: {MAX_CONTEXT_TOKENS})")

    return {"retrieved_context": compressed_docs}