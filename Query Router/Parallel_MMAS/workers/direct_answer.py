"""
Direct Answer Worker — handles greetings and general conversational queries.
"""

from langchain_core.documents import Document


def direct_answer_node(state: dict) -> dict:
    """Return a prompt-guiding document for greeting/conversational queries."""
    dummy_doc = Document(
        page_content=(
            "This is a general conversational query or greeting. "
            "Respond politely and directly to the user as Yaqeen, "
            "an Islamic knowledge assistant."
        ),
        metadata={"source": "Direct Answer"},
    )
    return {"retrieved_context": [dummy_doc]}