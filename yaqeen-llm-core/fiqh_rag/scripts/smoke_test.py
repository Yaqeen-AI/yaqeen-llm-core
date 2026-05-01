"""
Smoke test — 10 Fiqh questions end-to-end.
Run:  python -m scripts.smoke_test
Output is printed to console AND saved to smoke_report.md in the project root.
"""

import io
import sys
import time
from datetime import datetime
from pathlib import Path

QUESTIONS = [
    "متى يجوز جمع الصلاة؟",
    "ما حكم الصلاة في الأرض المغصوبة؟",
    "ما شروط صحة عقد البيع عند المذاهب الأربعة؟",
    "ما حكم الزكاة على الذهب والفضة؟",
    "ما حكم الطلاق في حالة الغضب الشديد؟",
    "ما حكم صوم من أفطر ناسياً في رمضان؟",
    "ما حكم الوضوء بالماء المستعمل؟",
    "ما حكم صلاة الجماعة؟",
    "ما شروط وجوب الحج؟",
    "ما حكم قراءة القرآن للحائض؟",
]

SEP  = "─" * 70
SEP2 = "═" * 70
REPORT_PATH = Path(__file__).parent.parent / "smoke_report.md"


class _Tee(io.TextIOBase):
    """Write to both a real stream and an in-memory buffer."""
    def __init__(self, real, buf):
        self._real = real
        self._buf  = buf

    def write(self, s):
        self._real.write(s)
        self._buf.write(s)
        return len(s)

    def flush(self):
        self._real.flush()


def main() -> None:
    buf = io.StringIO()
    sys.stdout = _Tee(sys.__stdout__, buf)

    print(f"# FiqhRAG Smoke Report\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("Loading retriever…")

    try:
        from core.retriever import FiqhRetriever
        retriever = FiqhRetriever()
    except SystemExit as e:
        print(f"[FATAL] Retriever init failed: {e}")
        sys.stdout = sys.__stdout__
        sys.exit(1)

    try:
        from core.generator import generate_answer
        _gen_available = True
    except Exception as e:
        print(f"[WARN] Generator unavailable ({e}) — retrieval-only mode")
        _gen_available = False

    passed = 0
    failed = 0

    for idx, q in enumerate(QUESTIONS, 1):
        print(f"\n{SEP}")
        print(f"[{idx:02d}/10] {q}")
        print(SEP)
        t0 = time.perf_counter()
        try:
            results = retriever.retrieve(q)
            elapsed_r = time.perf_counter() - t0

            if not results:
                print("  ⚠  No results returned")
                failed += 1
                continue

            print(f"  ✓  Retrieval  {elapsed_r:.2f}s  — {len(results)} chunks")
            for i, r in enumerate(results[:3], 1):
                print(f"     [{i}] {r.short_ref()}  rerank={r.rerank_score:.3f}  "
                      f"{'  '.join(r.mazhabs) or '—'}")

            if _gen_available:
                t1 = time.perf_counter()
                answer = generate_answer(q, results)
                elapsed_g = time.perf_counter() - t1
                print(f"  ✓  Generation {elapsed_g:.2f}s")
                print()
                print(f"**Answer:**\n\n{answer}")
            passed += 1

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  ✗  FAILED after {elapsed:.2f}s: {e}")
            failed += 1

    print(f"\n{SEP2}")
    print(f"  Results: {passed} passed  /  {failed} failed  /  {len(QUESTIONS)} total")
    print(f"{SEP2}")

    sys.stdout = sys.__stdout__

    report = buf.getvalue()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {REPORT_PATH}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
