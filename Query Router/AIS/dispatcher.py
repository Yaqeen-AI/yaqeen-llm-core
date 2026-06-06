"""
AIS Dispatcher (Innate Response - Antibody Generation).

Dispatches sub-queries to workers in parallel using async/await.
Aggregates initial context chunks (antibodies) into a single, unified context array.
"""

import asyncio
import logging
import re
from typing import Dict, List, Set
from langchain_core.documents import Document
from state import AgentState

# Import synchronous worker nodes
from workers.quran_agent import quran_agent_node
from workers.hadith_agent import hadith_agent_node
from workers.fiqh_agent import fiqh_agent_node
from workers.direct_answer import direct_answer_node

logger = logging.getLogger("ais.dispatcher")

# Worker registry
WORKER_MAP = {
    "quran_agent": quran_agent_node,
    "hadith_agent": hadith_agent_node,
    "fiqh_agent": fiqh_agent_node,
    "direct_answer": direct_answer_node,
}

# Heuristic keywords for routing
QURAN_KEYWORDS = {
    "quran", "quranic", "ayah", "ayat", "surah", "sura", "verse", "verses",
    "tafsir", "revelation", "juz", "hizb", "makkah", "madinah", "recitation",
    "قرآن", "قرآنية", "قرآني", "آية", "آيات", "سورة", "سور",
    "تفسير", "وحي", "تلاوة", "مصحف", "جزء",
}

HADITH_KEYWORDS = {
    "hadith", "hadiths", "hadeeth", "prophet", "prophetic", "sunnah",
    "narration", "narrated", "narrator", "sahih", "bukhari", "muslim",
    "tirmidhi", "dawud", "nasa'i", "ibn majah",
    "حديث", "أحاديث", "نبي", "النبي", "الرسول", "رسول", "نبوي",
    "سنة", "سنن", "رواية", "صحيح", "بخاري", "مسلم",
}

FIQH_KEYWORDS = {
    "fiqh", "ruling", "rulings", "halal", "haram", "fatwa", "fatwas",
    "jurisprudence", "sharia", "shariah", "madhab", "madhhab", "hanafi",
    "maliki", "shafii", "hanbali", "wajib", "obligatory", "mustahab",
    "recommended", "makruh", "disliked", "mubah", "permissible",
    "forbidden", "ibadah", "worship", "muamalat", "prayer", "salah",
    "zakat", "fasting", "hajj", "nikah", "talaq", "purification",
    "فقه", "فقهي", "حكم", "أحكام", "حلال", "حرام", "فتوى", "فتاوى",
    "شريعة", "مذهب", "حنفي", "مالكي", "شافعي", "حنبلي", "واجب",
    "مستحب", "مكروه", "مباح", "صلاة", "زكاة", "صيام", "حج",
}

GREETINGS = {
    "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye", "help",
    "سلام", "السلام", "مرحبا", "أهلا", "شكرا", "وداعا", "مساعدة",
}

def tokenize(text: str) -> Set[str]:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return set(text.split())

def route_sub_query(sq: str) -> List[str]:
    """Determine which workers are relevant for a given sub-query."""
    tokens = tokenize(sq)
    
    # Check for direct conversational greeting
    if tokens & GREETINGS:
        return ["direct_answer"]
        
    agents = []
    if tokens & QURAN_KEYWORDS:
        agents.append("quran_agent")
    if tokens & HADITH_KEYWORDS:
        agents.append("hadith_agent")
    if tokens & FIQH_KEYWORDS:
        agents.append("fiqh_agent")
        
    # Default to all if no keywords match
    if not agents:
        agents = ["quran_agent", "hadith_agent", "fiqh_agent"]
        
    return agents

async def _run_worker_async(worker_name: str, sub_query: str) -> List[Document]:
    """Run a worker synchronously in a separate thread using asyncio.to_thread."""
    fn = WORKER_MAP.get(worker_name)
    if not fn:
        return []
        
    try:
        # Wrap state matching the worker expectation
        state = {"question": sub_query}
        result = await asyncio.to_thread(fn, state)
        docs = result.get("retrieved_context", [])
        
        # Tag documents with origin information
        for doc in docs:
            try:
                doc.metadata["sub_query"] = sub_query
                doc.metadata["worker"] = worker_name
            except (AttributeError, TypeError):
                pass
        return docs
    except Exception as e:
        logger.error(f"Worker {worker_name} failed on '{sub_query[:30]}...': {e}")
        return []

async def dispatcher_node(state: AgentState) -> dict:
    """
    Dispatcher Node (Phase 2):
    Runs workers in parallel for all decomposed sub-queries using async/await.
    Aggregates the retrieved chunks (antibodies).
    """
    sub_queries = state.get("sub_queries", [state["question"]])
    
    tasks = []
    dispatch_plan = {}
    
    for sq in sub_queries:
        workers = route_sub_query(sq)
        dispatch_plan[sq] = workers
        for w in workers:
            tasks.append(_run_worker_async(w, sq))
            
    print(f"   [Dispatcher] -> Innate response: dispatching {len(tasks)} parallel workers...")
    
    # Execute all retrievals in parallel using async/await
    results = await asyncio.gather(*tasks)
    
    # Flatten results into initial_context array
    initial_context = []
    for docs in results:
        initial_context.extend(docs)
        
    print(f"   [Dispatcher] -> Aggregated {len(initial_context)} initial antibodies")
    
    return {
        "initial_context": initial_context,
        "loop_step": state.get("loop_step", 0) + 1
    }
