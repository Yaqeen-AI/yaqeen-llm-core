"""
FiqhRAG — Full Arabic end-to-end system.

    python main.py

Flow:
    Arabic query → LangGraph (hybrid retrieval: BM25 + Jina v3 → RRF → Jina reranker)
               → Gemini 2.0 Flash (Google API) generates grounded Arabic answer
               → Answer + source citations displayed
"""

import threading

from llama_index.core import Settings

from core.embeddings import JinaEmbedding
from core.generator import GeminiLLM, generate_answer
from core.graph import fiqh_graph
from fiqh_rag.core.retriever import nodes_to_results
from core.cache import TwoTierCache

Settings.embed_model = JinaEmbedding()
Settings.llm = GeminiLLM()

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       نظام الفقه الإسلامي — بحث واسترجاع بالذكاء الاصطناعي  ║
║  بحث هجين: BM25 + Jina v3     |  إعادة ترتيب: Jina v2      ║
║  قاعدة بيانات: Qdrant          |  توليد: Gemini 2.0 Flash    ║
║  المصدر: الموسوعة الفقهية الكويتية — ٤٦ مجلداً               ║
╚══════════════════════════════════════════════════════════════╝
اكتب سؤالك بالعربية.  للخروج: اكتب  خروج
"""


def show(answer: str, results: list) -> None:
    print("\n" + "═" * 68)
    print("الإجابة:")
    print("═" * 68)
    print(answer)
    print("\n" + "─" * 68)
    print("المصادر المُسترجعة:")
    print("─" * 68)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r.volume_id}  |  {r.book_page}  |  {r.chunk_page}"
              f"   rerank={r.rerank_score:.3f}")
    print("═" * 68)


def main() -> None:
    print(BANNER)

    cache = TwoTierCache()

    while True:
        try:
            query = input("\nالسؤال › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nمع السلامة.")
            break

        if not query:
            continue
        if query in {"خروج", "quit", "exit", "q"}:
            print("مع السلامة.")
            break

        # ── Cache check (Tier 1 + Tier 2) ────────────────────────────────
        cached, vec = cache.get(query)  # vec reused below to skip re-embed
        if cached is not None:
            print("\n⚡ إجابة من الذاكرة المؤقتة:")
            print("═" * 68)
            print(cached)
            print("═" * 68)
            continue

        print("جارٍ البحث في الموسوعة الفقهية الكويتية...")
        try:
            state = fiqh_graph.invoke({"query": query, "precomputed_embedding": vec})
            results = nodes_to_results(state["documents"])
        except Exception as e:
            print(f"خطأ في الاسترجاع: {e}")
            continue

        if not results:
            print("لم يُعثر على نتائج ذات صلة.")
            continue

        print(f"تم استرجاع {len(results)} مقطع. جارٍ توليد الإجابة عبر Gemini...")
        try:
            answer = generate_answer(query, results)
        except Exception as e:
            print(f"خطأ في التوليد — تحقق من GOOGLE_API_KEY: {e}")
            continue

        threading.Thread(target=cache.set, args=(query, answer, _vec), daemon=True).start()
        show(answer, results)


if __name__ == "__main__":
    main()
