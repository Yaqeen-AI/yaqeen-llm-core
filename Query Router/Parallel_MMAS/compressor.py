"""
Context Compressor — fits highest-scored chunks into the LLM context window.

Receives Inspector-approved documents (sorted by quality) and applies
sentence-aware truncation to fit within the token budget while preserving
document boundaries and source metadata for proper citations.
"""

import re
import logging
from langchain_core.documents import Document
from state import AgentState

logger = logging.getLogger("mmas.compressor")

# Maximum tokens to feed to the Writer LLM as context
MAX_CONTEXT_TOKENS = 750

# Sentence-ending patterns for clean truncation
_SENTENCE_END = re.compile(r'[.。!؟?\n]')


def _truncate_at_sentence(text: str, max_tokens: int, tokenizer) -> str:
    """Truncate text at the last complete sentence within the token budget."""
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

    Preserves individual document boundaries and source metadata
    so the writer can produce proper citations.  Documents are already
    sorted by Inspector quality score (highest first).
    """
    docs = state.get("retrieved_context", [])
    if not docs:
        return {"retrieved_context": []}

    # Lazy tokenizer import
    try:
        from models.llm_loader import get_tokenizer
        tokenizer = get_tokenizer()
    except Exception as e:
        logger.warning(f"Tokenizer unavailable ({e}), using fallback")
        print(f"   [Compressor] -> Tokenizer unavailable ({e}), using fallback")
        class _FallbackTokenizer:
            def encode(self, text):
                return text.split()
            def decode(self, tokens, skip_special_tokens=True):
                return " ".join(tokens)
        tokenizer = _FallbackTokenizer()

    # ── Build compressed context ──
    compressed_docs = []
    tokens_used = 0

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content.strip()
        block = f"[Source: {source}]\n{content}"

        block_tokens = len(tokenizer.encode(block))

        if tokens_used + block_tokens <= MAX_CONTEXT_TOKENS:
            compressed_docs.append(doc)
            tokens_used += block_tokens
        else:
            remaining = MAX_CONTEXT_TOKENS - tokens_used
            if remaining > 50:
                truncated = _truncate_at_sentence(content, remaining - 20, tokenizer)
                if truncated.strip():
                    compressed_docs.append(Document(
                        page_content=truncated + "\n[...truncated]",
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

    print(
        f"   [Compressor] -> {len(docs)} docs in, {len(compressed_docs)} docs out, "
        f"~{tokens_used} tokens used (budget: {MAX_CONTEXT_TOKENS})"
    )

    return {"retrieved_context": compressed_docs}