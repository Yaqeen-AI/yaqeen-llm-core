"""
FiqhRAG — Full Arabic end-to-end system.

    python main.py

Flow:
    Arabic query → Hybrid retrieval (TF-IDF + Jina v3 → RRF → Jina reranker)
               → Gemma 4 (LM Studio) generates grounded Arabic answer
               → Answer + source citations displayed
"""

from core.retriever import FiqhRetriever
from core.generator import generate_answer
from core.cache import TwoTierCache

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       نظام الفقه الإسلامي — بحث واسترجاع بالذكاء الاصطناعي  ║
║  بحث هجين: TF-IDF + Jina v3  |  إعادة ترتيب: Jina v2       ║
║  قاعدة بيانات: Qdrant          |  توليد: Gemma 4 (LM Studio) ║
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

    try:
        retriever = FiqhRetriever()
    except SystemExit as e:
        print(f"خطأ: {e}")
        return

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
        cached = cache.get(query)
        if cached is not None:
            print("\n⚡ إجابة من الذاكرة المؤقتة:")
            print("═" * 68)
            print(cached)
            print("═" * 68)
            continue

        print("جارٍ البحث في الموسوعة الفقهية الكويتية...")
        try:
            results = retriever.retrieve(query)
        except Exception as e:
            print(f"خطأ في الاسترجاع: {e}")
            continue

        if not results:
            print("لم يُعثر على نتائج ذات صلة.")
            continue

        print(f"تم استرجاع {len(results)} مقطع. جارٍ توليد الإجابة عبر Gemma 4...")
        try:
            answer = generate_answer(query, results)
        except Exception as e:
            print(f"خطأ في التوليد (هل LM Studio يعمل؟): {e}")
            continue

        cache.set(query, answer)
        show(answer, results)


if __name__ == "__main__":
    main()
