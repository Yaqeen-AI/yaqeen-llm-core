"""
Negative Selection Algorithm (NSA) query filter.
Located inside the Query Router directory to keep it tracked under the current git branch.
Classifies user queries into:
1. Islamic (Self) -> proceed to semantic cache & RAG pipeline
2. Borderline / General -> answer directly with LLM (no RAG) + suffix recommendation
3. Harmful / Irrelevant -> reject with out-of-scope response
"""

import os
import sys
import pickle
import requests
import numpy as np
import re

# Resolve path relative to this file
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ANCHORS_FILE = os.path.join(_SCRIPT_DIR, "nsa_anchors.pkl")

# Messages
HARMFUL_MESSAGE = (
    "This question is outside the scope of this system, which specializes in Islamic knowledge. "
    "Please ask about Quran, Hadith, Fiqh, or related Islamic topics."
)

GENERAL_SUFFIX = (
    "\n\nI answered this as a general question. If your question relates to Islamic rulings, "
    "halal/haram, or religious guidance, I can provide a much more detailed and sourced "
    "answer using Quran, Hadith, and Fiqh references. Would you like me to do that?"
)

# Configuration
_JINA_EMBED_MODEL = "jina-embeddings-v3"
_EMBED_DIM = 1024

# Cache anchor embeddings
_anchors = None
_local_embed_model = None
_http = requests.Session()


def _load_anchors():
    """Load the precomputed anchors and their embeddings."""
    global _anchors
    if _anchors is not None:
        return _anchors

    if not os.path.exists(_ANCHORS_FILE):
        raise FileNotFoundError(
            f"NSA anchors file not found at {_ANCHORS_FILE}. "
            f"Please run generate_nsa_anchors.py first."
        )

    with open(_ANCHORS_FILE, "rb") as f:
        _anchors = pickle.load(f)

    return _anchors


def _get_local_model():
    """Lazy loader for local SentenceTransformer model (fallback)."""
    global _local_embed_model
    if _local_embed_model is not None:
        return _local_embed_model

    try:
        from sentence_transformers import SentenceTransformer
        print("   [NSA Filter] -> Loading local SentenceTransformer fallback model...")
        _local_embed_model = SentenceTransformer(
            "jinaai/jina-embeddings-v3", trust_remote_code=True
        )
        return _local_embed_model
    except Exception as e:
        print(f"   [NSA Filter] -> Failed to load local SentenceTransformer fallback model: {e}")
        return None


def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a given text, trying Jina API first, then local fallback."""
    # Try Jina API
    jina_key = os.getenv("JINA_API_KEY")
    if jina_key:
        try:
            resp = _http.post(
                "https://api.jina.ai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {jina_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": _JINA_EMBED_MODEL,
                    "input": [text],
                    "dimensions": _EMBED_DIM,
                    "task": "retrieval.query"
                },
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
            else:
                print(f"   [NSA Filter] -> Jina API returned code {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"   [NSA Filter] -> Jina API call failed: {e}")

    # Fallback to local SentenceTransformer
    model = _get_local_model()
    if model is not None:
        try:
            kwargs = {
                "normalize_embeddings": True,
                "task": "retrieval.query",
                "truncate_dim": _EMBED_DIM
            }
            emb = model.encode(text, **kwargs)
            return emb.tolist()
        except Exception as e:
            print(f"   [NSA Filter] -> Local ST encoding failed: {e}")

    raise RuntimeError("Unable to generate query embedding (both Jina API and local ST failed).")


def classify_query(query: str) -> str:
    """
    Classify a query into 'islamic', 'general', or 'harmful'.
    Uses semantic similarity with class anchors.
    """
    try:
        anchors_data = _load_anchors()
        query_vector = np.array(get_embedding(query))

        scores = {}
        for category, cat_data in anchors_data.items():
            embeddings = np.array(cat_data["embeddings"])
            similarities = np.dot(embeddings, query_vector)
            
            similarities = sorted(similarities, reverse=True)
            
            max_sim = similarities[0]
            top3_mean = np.mean(similarities[:3])
            
            scores[category] = 0.6 * max_sim + 0.4 * top3_mean

        print(f"   [NSA Filter] -> Classification scores:")
        for cat, score in scores.items():
            print(f"      - {cat}: {score:.4f}")

        decision = max(scores, key=scores.get)
        
        # If the query is weak (very low similarity to both Islamic and Harmful classes),
        # treat it as a general query to avoid loading heavy RAG pipelines.
        if scores.get("islamic", 0.0) < 0.35 and scores.get("harmful", 0.0) < 0.35:
            decision = "general"
            
        print(f"   [NSA Filter] -> Classified query as '{decision}'")
        return decision

    except Exception as e:
        print(f"   [NSA Filter] -> ERROR: Classification failed: {e}. Defaulting to 'islamic'.")
        return "islamic"


def generate_direct_answer(query: str) -> str:
    """Generate a direct answer to a general query using the configured LLM."""
    print("   [NSA Filter] -> Generating direct answer using LLM...")
    
    try:
        from shared_nodes.models.llm_loader import get_llm
        llm = get_llm()
        response = llm.invoke(query)
        ans_text = ""
        if hasattr(response, "content"):
            ans_text = response.content
        else:
            ans_text = str(response)
        ans_text = strip_think_tags(ans_text)
        return clean_arabic_text(ans_text)
    except Exception as e:
        # Fallback to local LM Studio if possible
        try:
            from core.generator import _client, LM_STUDIO_MODEL, MAX_OUTPUT_TOKENS
            response = _client.chat.completions.create(
                model=LM_STUDIO_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the user's question directly, clearly, and concisely."},
                    {"role": "user", "content": query}
                ],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.7,
            )
            ans_text = strip_think_tags(response.choices[0].message.content)
            return clean_arabic_text(ans_text)
        except Exception as e2:
            return f"Error generating direct answer: {e} / {e2}"


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> tags and their contents from the generated output."""
    import re
    if not text:
        return text
    # Remove closed <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove any unclosed <think> tag blocks to prevent leakage if truncated
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def clean_arabic_text(text: str) -> str:
    """
    Cleans Arabic text.
    Removes zero-width characters, tatweel, mid-word spaces, and duplicate letters.
    """
    if not text:
        return text
    # Remove zero-width characters, invisible characters, and tatweel
    text = re.sub(r'[\u200B-\u200D\uFEFFـ]', '', text)
    # Removed mid-word spaces regex because it concatenates all Arabic words
    # Remove duplicate consecutive Arabic characters (e.g., ندمم -> ندم)
    text = re.sub(r'([ا-ي])\1+', r'\1', text)
    # Normalize extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

