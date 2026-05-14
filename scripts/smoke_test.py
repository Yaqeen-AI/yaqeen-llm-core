"""
Smoke test — RAG evaluation across 20 Fiqh questions + adversarial set.
Run:  python -m scripts.smoke_test

Evaluation metrics (TABLE II):
  Retrieval  — Recall@5, Precision@5, Hit Rate@5, MRR, NDCG@5
               (relevance proxy: Jina rerank_score ≥ 0.5)
  Reranking  — NDCG (same NDCG@5 signal, rerank scores as graded relevance)
  Generation — Sufficiency, Faithfulness, Completeness, Answer Relevance,
               Coverage Score, Hallucination Rate  (Gemini-as-judge, single call/query)
  Safety     — Rejection Rate  (adversarial query set, top-1 rerank < 0.3)
  System     — Latency P95  (end-to-end: retrieval + generation + eval)

Output is printed to console AND saved to smoke_report.md in the project root.
"""

import io
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

import os
from core.config import GOOGLE_API_KEY

# ── Constants ──────────────────────────────────────────────────────────────────

RELEVANCE_THRESHOLD = 0.0   # rerank_score >= this → "relevant" (Jina v3 scores range ~−0.2 to +0.5)
REJECTION_THRESHOLD = 0.0   # top-1 rerank_score < this → "correctly rejected"

# Separate model for the LLM judge — decoupled from GEMINI_MODEL so evaluation
# works even if the production generation model is unavailable via this API.
JUDGE_MODEL = os.getenv("SMOKE_JUDGE_MODEL", "gemini-2.0-flash")

# ── Questions ─────────────────────────────────────────────────────────────────

QUESTIONS = [
    # Original set
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
    # New set
    "هل الزنا ينقد الوضوء",
    "هل النوم لمده قصيره يننقض الوضوء",
    "ما هي اركان الايمانن",
    "هل يصح رفع الاصبع في بدايه التشهد ام عند قول اشهد ان لا اله الا الله و اشهد ان محمدا رسول الله في الصلاه",
    "هل اغنية طلع البدر علينا حدثت ام بدعة؟",
    "هل استخدام السبحة بدعة",
    # Extended set
    "حكم صلاة الجمعه مع صلاة العيد",
    "حكم سماع الاغاني",
    "حكم تربية الكلاب",
    "حكم كفالة اليتيم",
]

ADVERSARIAL_QUESTIONS = [
    "ما عاصمة فرنسا؟",
    "كيف أصنع كعكة الشوكولاتة؟",
    "ما هو أفضل برنامج لتحرير الصور؟",
    "كيف أعالج ضغط الدم المرتفع؟",
    "من هو مخترع الهاتف؟",
    "ما هي أسرع سيارة في العالم؟",
]

TOTAL     = len(QUESTIONS)
TOTAL_ADV = len(ADVERSARIAL_QUESTIONS)
SEP       = "─" * 72
SEP2      = "═" * 72
REPORT_PATH  = Path(__file__).parent.parent / "smoke_report.md"
SNIPPET_LEN  = 300   # chars of chunk_text shown per result


# ── Gemini judge client ────────────────────────────────────────────────────────

_judge_client: Optional[genai.Client] = None


def _get_judge_client() -> genai.Client:
    global _judge_client
    if _judge_client is None:
        _judge_client = genai.Client(api_key=GOOGLE_API_KEY)
    return _judge_client


# ── Per-query result container ─────────────────────────────────────────────────

@dataclass
class QueryResult:
    question: str
    is_adversarial: bool = False
    # Retrieval
    hit: bool = False
    result_count: int = 0
    top1_rerank: float = 0.0
    mean_rerank_top3: float = 0.0
    score_spread: float = 0.0
    mazhab_coverage: int = 0
    topic_filter: Optional[list] = field(default=None)
    retrieval_latency: float = 0.0
    # TABLE II — Retrieval ranking metrics
    recall_at_5: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    # Generation
    answer: Optional[str] = None
    generation_latency: float = 0.0
    # TABLE II — Generation quality metrics (Gemini-as-judge)
    faithfulness: float = 0.0
    sufficiency: float = 0.0
    completeness: float = 0.0
    answer_relevance: float = 0.0
    coverage_score: float = 0.0
    hallucination_rate: float = 0.0
    eval_latency: float = 0.0
    # Error
    error: Optional[str] = None

    @property
    def total_latency(self) -> float:
        return self.retrieval_latency + self.generation_latency + self.eval_latency


# ── Retrieval ranking metric helpers ──────────────────────────────────────────

def _precision_at_5(scores: list[float]) -> float:
    top5 = scores[:5]
    if not top5:
        return 0.0
    return sum(1 for s in top5 if s >= RELEVANCE_THRESHOLD) / 5


def _recall_at_5(scores: list[float]) -> float:
    return 1.0 if any(s >= RELEVANCE_THRESHOLD for s in scores[:5]) else 0.0


def _mrr_at_5(scores: list[float]) -> float:
    for i, s in enumerate(scores[:5]):
        if s >= RELEVANCE_THRESHOLD:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_5(scores: list[float]) -> float:
    rel  = [1 if s >= RELEVANCE_THRESHOLD else 0 for s in scores[:5]]
    dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    n_rel = sum(rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    idx = max(0, math.ceil(0.95 * len(values)) - 1)
    return sorted(values)[idx]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ── LLM judge (Gemini) ─────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
أنت محكم متخصص في تقييم جودة أنظمة الإجابة عن الأسئلة الفقهية.

بناءً على السؤال والسياق المسترجع والإجابة المولدة، قيّم الإجابة وأعطِ درجات من 0.0 إلى 1.0 لكل مقياس.

السؤال:
{query}

السياق المسترجع (أفضل 5 وثائق):
{context}

الإجابة المولدة:
{answer}

تعريفات المقاييس:
- faithfulness: نسبة الادعاءات في الإجابة المدعومة صراحةً بالسياق المسترجع (1.0 = كل الادعاءات مدعومة)
- sufficiency: مدى كفاية السياق المسترجع للإجابة الكاملة على السؤال (1.0 = كافٍ تماماً)
- completeness: مدى تغطية الإجابة لجميع جوانب السؤال المهمة (1.0 = شاملة تماماً)
- answer_relevance: مدى إجابة النص المولد على السؤال مباشرةً (1.0 = ذو صلة تامة)
- coverage_score: مدى استخدام الإجابة للمعلومات في السياق المسترجع (1.0 = يستخدم السياق بالكامل)
- hallucination_rate: نسبة الادعاءات في الإجابة غير المدعومة بالسياق (0.0 = لا هلوسة)

أجب بـ JSON فقط، بدون أي نص إضافي:
{{"faithfulness": 0.0, "sufficiency": 0.0, "completeness": 0.0, "answer_relevance": 0.0, "coverage_score": 0.0, "hallucination_rate": 0.0}}
"""

_JUDGE_DEFAULTS = {
    "faithfulness": 0.0, "sufficiency": 0.0, "completeness": 0.0,
    "answer_relevance": 0.0, "coverage_score": 0.0, "hallucination_rate": 0.0,
}


def _eval_generation(query: str, results: list, answer: str) -> dict[str, float]:
    """Call Gemini to score generation quality; returns dict of 6 metrics (0–1)."""
    if not answer or not results:
        return dict(_JUDGE_DEFAULTS)

    context_parts = [f"[{i}] {r.chunk_text[:400]}" for i, r in enumerate(results, 1)]
    context = "\n\n".join(context_parts)
    prompt  = _JUDGE_PROMPT.format(query=query, context=context, answer=answer[:1200])

    try:
        client   = _get_judge_client()
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=256, temperature=0.0),
        )
        raw = response.text or ""
        m = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            return {k: float(parsed.get(k, 0.0)) for k in _JUDGE_DEFAULTS}
    except Exception as e:
        print(f"  [eval] Gemini judge failed: {e}")
    return dict(_JUDGE_DEFAULTS)


# ── Display helpers ────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float = 1.0, width: int = 12) -> str:
    filled = int(round(value / max_val * width)) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _grade(score: float) -> str:
    if score >= 0.8: return "EXCELLENT"
    if score >= 0.6: return "GOOD"
    if score >= 0.4: return "FAIR"
    return "POOR"


class _Tee(io.TextIOBase):
    def __init__(self, real, buf):
        if hasattr(real, "buffer"):
            real = io.TextIOWrapper(real.buffer, encoding="utf-8", errors="replace")
        self._real = real
        self._buf  = buf

    def write(self, s):
        self._real.write(s)
        self._buf.write(s)
        return len(s)

    def flush(self):
        self._real.flush()


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate_query(
    idx: int,
    question: str,
    total: int,
    is_adversarial: bool = False,
) -> QueryResult:
    qr = QueryResult(question=question, is_adversarial=is_adversarial)

    label = f"[ADV {idx:02d}/{total}]" if is_adversarial else f"[{idx:02d}/{total}]"
    print(f"\n{SEP}")
    print(f"{label} {question}")
    print(SEP)

    t0 = time.perf_counter()
    try:
        from core.graph import fiqh_graph
        from core.llamaindex_retriever import nodes_to_results
        state   = fiqh_graph.invoke({"query": question})
        results = nodes_to_results(state["documents"])
        qr.retrieval_latency = time.perf_counter() - t0

        mazhab_filter = state.get("mazhab_filter")
        topic_filter  = state.get("topic_filter")
        print(f"  Filters — mazhab: {mazhab_filter or '—'}  topic: {topic_filter or '—'}")
    except Exception as e:
        qr.retrieval_latency = time.perf_counter() - t0
        qr.error = str(e)
        print(f"  FAILED ({qr.retrieval_latency:.2f}s): {e}")
        return qr

    qr.hit          = len(results) > 0
    qr.result_count = len(results)

    if not qr.hit:
        print(f"  No results  ({qr.retrieval_latency:.2f}s)")
        return qr

    scores = [r.rerank_score for r in results]
    qr.top1_rerank      = scores[0]
    qr.mean_rerank_top3 = sum(scores[:3]) / min(3, len(scores))
    qr.score_spread     = scores[0] - scores[-1]
    all_mazhabs         = {m for r in results for m in (r.mazhabs or [])}
    qr.mazhab_coverage  = len(all_mazhabs)
    topics = list({r.fiqh_topic for r in results if r.fiqh_topic})
    qr.topic_filter = topics or None

    # TABLE II — retrieval ranking metrics
    qr.recall_at_5    = _recall_at_5(scores)
    qr.precision_at_5 = _precision_at_5(scores)
    qr.mrr            = _mrr_at_5(scores)
    qr.ndcg_at_5      = _ndcg_at_5(scores)

    print(f"  Retrieval   {qr.retrieval_latency:.2f}s  —  {qr.result_count} chunks")
    print(f"  Top-1 rerank    : {qr.top1_rerank:.4f}  {_bar(qr.top1_rerank)}")
    print(f"  Mean rerank top3: {qr.mean_rerank_top3:.4f}  {_bar(qr.mean_rerank_top3)}")
    print(f"  Score spread    : {qr.score_spread:.4f}")
    print(f"  Recall@5        : {qr.recall_at_5:.4f}  Precision@5: {qr.precision_at_5:.4f}")
    print(f"  MRR             : {qr.mrr:.4f}  NDCG@5: {qr.ndcg_at_5:.4f}")
    print(f"  Mazhab coverage : {qr.mazhab_coverage}  {sorted(all_mazhabs)}")
    print(f"  Topics in docs  : {topics or '—'}")
    print()

    for i, r in enumerate(results, 1):
        snippet  = r.chunk_text[:SNIPPET_LEN].replace("\n", " ")
        if len(r.chunk_text) > SNIPPET_LEN:
            snippet += "…"
        mazhab_s = "  ".join(r.mazhabs) if r.mazhabs else "—"
        print(f"  ── Doc {i} ──────────────────────────────────────────────────────")
        print(f"  Ref    : {r.short_ref()}")
        print(f"  Rerank : {r.rerank_score:.4f}  |  Qdrant: {r.qdrant_score:.4f}")
        print(f"  Mazhab : {mazhab_s}  |  Topic: {r.fiqh_topic or '—'}")
        print(f"  Text   : {snippet}")
        print()

    # Adversarial queries: retrieval only (no generation cost)
    if is_adversarial:
        return qr

    # Generation
    print("  Generating answer…")
    from core.generator import generate_answer
    t_gen = time.perf_counter()
    try:
        qr.answer = generate_answer(question, results)
        qr.generation_latency = time.perf_counter() - t_gen
    except Exception as e:
        qr.generation_latency = time.perf_counter() - t_gen
        print(f"  Generation FAILED ({qr.generation_latency:.2f}s): {e}")
        return qr

    ans_snippet = (qr.answer or "")[:200].replace("\n", " ")
    print(f"  Generation  {qr.generation_latency:.2f}s")
    print(f"  Answer      : {ans_snippet}…")

    # LLM evaluation (Gemini-as-judge)
    print("  Evaluating answer quality…")
    t_eval = time.perf_counter()
    gen_scores = _eval_generation(question, results[:5], qr.answer or "")
    qr.eval_latency = time.perf_counter() - t_eval

    qr.faithfulness      = gen_scores.get("faithfulness",     0.0)
    qr.sufficiency       = gen_scores.get("sufficiency",      0.0)
    qr.completeness      = gen_scores.get("completeness",     0.0)
    qr.answer_relevance  = gen_scores.get("answer_relevance", 0.0)
    qr.coverage_score    = gen_scores.get("coverage_score",   0.0)
    qr.hallucination_rate = gen_scores.get("hallucination_rate", 0.0)

    print(f"  Eval        {qr.eval_latency:.2f}s")
    print(f"  Faithfulness: {qr.faithfulness:.2f}  Sufficiency: {qr.sufficiency:.2f}  "
          f"Completeness: {qr.completeness:.2f}")
    print(f"  Ans.Rel: {qr.answer_relevance:.2f}  Coverage: {qr.coverage_score:.2f}  "
          f"Hallucination: {qr.hallucination_rate:.2f}")
    print(f"  E2E total   {qr.total_latency:.2f}s")

    return qr


# ── Aggregate report ───────────────────────────────────────────────────────────

def print_aggregate(fiqh_results: list[QueryResult], adv_results: list[QueryResult]) -> None:
    hits   = [r for r in fiqh_results if r.hit]
    errors = [r for r in fiqh_results if r.error]
    gen_ok = [r for r in fiqh_results if r.answer]

    # Retrieval
    hit_rate    = len(hits) / len(fiqh_results) if fiqh_results else 0
    avg_recall5 = _avg([r.recall_at_5    for r in fiqh_results])
    avg_prec5   = _avg([r.precision_at_5 for r in fiqh_results])
    avg_mrr     = _avg([r.mrr            for r in fiqh_results])
    avg_ndcg5   = _avg([r.ndcg_at_5      for r in fiqh_results])
    avg_top1    = _avg([r.top1_rerank    for r in hits])
    avg_mean3   = _avg([r.mean_rerank_top3 for r in hits])
    avg_spread  = _avg([r.score_spread   for r in hits])
    avg_mazhab  = _avg([r.mazhab_coverage for r in hits])

    # Generation
    avg_faith = _avg([r.faithfulness      for r in gen_ok])
    avg_suff  = _avg([r.sufficiency       for r in gen_ok])
    avg_comp  = _avg([r.completeness      for r in gen_ok])
    avg_rel   = _avg([r.answer_relevance  for r in gen_ok])
    avg_cov   = _avg([r.coverage_score    for r in gen_ok])
    avg_hall  = _avg([r.hallucination_rate for r in gen_ok])

    # Safety
    def _is_rejected(r: QueryResult) -> bool:
        return not r.hit or r.top1_rerank < REJECTION_THRESHOLD

    n_rejected     = sum(1 for r in adv_results if _is_rejected(r))
    rejection_rate = n_rejected / len(adv_results) if adv_results else 0.0

    # Latency
    avg_r_lat     = _avg([r.retrieval_latency  for r in fiqh_results])
    avg_g_lat     = _avg([r.generation_latency for r in gen_ok])
    e2e_latencies = [r.total_latency for r in fiqh_results if r.answer]
    p95_e2e       = _p95(e2e_latencies)

    print(f"\n\n{SEP2}")
    print("  AGGREGATE EVALUATION REPORT  —  TABLE II Metrics")
    print(SEP2)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    print("\n### Retrieval Metrics\n")
    print(f"  {'Metric':<38} {'Value':>9}  Visual          Grade")
    print(f"  {'─'*38} {'─'*9}  {'─'*14}  {'─'*9}")
    print(f"  {'Recall@5':<38} {avg_recall5:>8.1%}  {_bar(avg_recall5)}  {_grade(avg_recall5)}")
    print(f"  {'Precision@5':<38} {avg_prec5:>8.1%}  {_bar(avg_prec5)}  {_grade(avg_prec5)}")
    print(f"  {'Hit Rate@5':<38} {hit_rate:>8.1%}  {_bar(hit_rate)}  {_grade(hit_rate)}")
    print(f"  {'MRR':<38} {avg_mrr:>9.4f}  {_bar(avg_mrr)}  {_grade(avg_mrr)}")

    # ── Reranking (NDCG) ───────────────────────────────────────────────────────
    print("\n### Reranking Metrics\n")
    print(f"  {'Metric':<38} {'Value':>9}  Visual          Grade")
    print(f"  {'─'*38} {'─'*9}  {'─'*14}  {'─'*9}")
    print(f"  {'NDCG@5':<38} {avg_ndcg5:>9.4f}  {_bar(avg_ndcg5)}  {_grade(avg_ndcg5)}")
    print(f"  {'Avg Top-1 Rerank Score':<38} {avg_top1:>9.4f}  {_bar(avg_top1)}")
    print(f"  {'Avg Mean Rerank Score (top-3)':<38} {avg_mean3:>9.4f}  {_bar(avg_mean3)}")
    print(f"  {'Avg Score Spread':<38} {avg_spread:>9.4f}")
    print(f"  {'Avg Mazhab Coverage':<38} {avg_mazhab:>9.1f}  madhabs/query")

    # ── Generation ─────────────────────────────────────────────────────────────
    print("\n### Generation Metrics\n")
    if gen_ok:
        print(f"  {'Metric':<38} {'Value':>9}  Visual          Grade")
        print(f"  {'─'*38} {'─'*9}  {'─'*14}  {'─'*9}")
        print(f"  {'Sufficiency':<38} {avg_suff:>9.4f}  {_bar(avg_suff)}  {_grade(avg_suff)}")
        print(f"  {'Faithfulness Score':<38} {avg_faith:>9.4f}  {_bar(avg_faith)}  {_grade(avg_faith)}")
        print(f"  {'Completeness':<38} {avg_comp:>9.4f}  {_bar(avg_comp)}  {_grade(avg_comp)}")
        print(f"  {'Answer Relevance':<38} {avg_rel:>9.4f}  {_bar(avg_rel)}  {_grade(avg_rel)}")
        print(f"  {'Coverage Score':<38} {avg_cov:>9.4f}  {_bar(avg_cov)}  {_grade(avg_cov)}")
        hall_grade = "LOW ✓" if avg_hall < 0.2 else _grade(1.0 - avg_hall)
        print(f"  {'Hallucination Rate':<38} {avg_hall:>9.4f}  {_bar(avg_hall)}  {hall_grade}")
        print(f"\n  (based on {len(gen_ok)}/{TOTAL} queries where generation succeeded)")
    else:
        print("  No answers generated — generation metrics unavailable.")

    # ── Safety ─────────────────────────────────────────────────────────────────
    print("\n### Adversarial / Safety Metrics\n")
    print(f"  {'Rejection Rate':<38} {rejection_rate:>8.1%}  {_bar(rejection_rate)}  "
          f"({n_rejected}/{len(adv_results)} correctly rejected)")
    print()
    for r in adv_results:
        status = "REJECTED ✓" if _is_rejected(r) else "PASSED   ✗"
        q_s = r.question[:50]
        print(f"    {status}  top-1={r.top1_rerank:.3f}  {q_s}")

    # ── System / Latency ───────────────────────────────────────────────────────
    print("\n### System Metrics\n")
    print(f"  {'Metric':<38} {'Value':>9}")
    print(f"  {'─'*38} {'─'*9}")
    print(f"  {'Avg Retrieval Latency':<38} {avg_r_lat:>8.2f}s")
    print(f"  {'Avg Generation Latency':<38} {avg_g_lat:>8.2f}s")
    print(f"  {'Latency P95 (E2E)':<38} {p95_e2e:>8.2f}s")

    # ── Per-query summary table ────────────────────────────────────────────────
    print(f"\n### Per-Query Summary\n")
    hdr = (f"  {'#':>3}  {'Hit':>3}  {'R@5':>4}  {'P@5':>4}  {'MRR':>4}  "
           f"{'NDCG':>4}  {'Faith':>5}  {'Hall':>4}  {'R-lat':>5}  {'G-lat':>5}  Question")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for i, r in enumerate(fiqh_results, 1):
        hit_s   = "Y" if r.hit else "N"
        q_short = r.question[:34] + ("…" if len(r.question) > 34 else "")
        print(f"  {i:>3}  {hit_s:>3}  {r.recall_at_5:>4.2f}  {r.precision_at_5:>4.2f}  "
              f"{r.mrr:>4.2f}  {r.ndcg_at_5:>4.2f}  {r.faithfulness:>5.2f}  "
              f"{r.hallucination_rate:>4.2f}  {r.retrieval_latency:>4.1f}s  "
              f"{r.generation_latency:>4.1f}s  {q_short}")

    print(f"\n{SEP2}")
    print(f"  RESULT: {len(hits)} retrieval hits  /  {len(gen_ok)} answers generated  "
          f"/  {len(errors)} errors  /  {TOTAL} total")
    print(f"  SAFETY: {n_rejected}/{len(adv_results)} adversarial queries correctly rejected")
    print(SEP2)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    buf = io.StringIO()
    _real_stdout = sys.__stdout__
    sys.stdout = _Tee(sys.__stdout__, buf)

    print(f"# FiqhRAG Smoke Report — Full Pipeline Evaluation\n"
          f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"**Questions:** {TOTAL} Fiqh + {TOTAL_ADV} adversarial  "
          f"|  **Mode:** retrieval + generation + LLM-eval (Gemini-as-judge)\n")
    print(f"**Metrics (TABLE II):** Recall@5, Precision@5, Hit Rate@5, MRR, NDCG@5 · "
          f"Sufficiency, Faithfulness, Completeness, Answer Relevance, Coverage Score, "
          f"Hallucination Rate · Rejection Rate · Latency P95\n")
    print(f"**Relevance threshold:** rerank_score ≥ {RELEVANCE_THRESHOLD}  "
          f"|  **Rejection threshold:** top-1 rerank < {REJECTION_THRESHOLD}\n")
    print("Loading pipeline…")

    try:
        from core.graph import fiqh_graph  # noqa: F401  — warm up graph + retriever
        print("Pipeline: ready\n")
    except SystemExit as e:
        print(f"[FATAL] Pipeline init failed: {e}")
        sys.stdout = sys.__stdout__
        sys.exit(1)

    # 12s between queries keeps Jina reranker under its ~6 RPM free-tier limit.
    INTER_QUERY_DELAY = 12

    # ── Fiqh evaluation loop ───────────────────────────────────────────────────
    fiqh_results: list[QueryResult] = []
    for idx, q in enumerate(QUESTIONS, 1):
        qr = evaluate_query(idx, q, TOTAL)
        fiqh_results.append(qr)
        if idx < TOTAL:
            print(f"  waiting {INTER_QUERY_DELAY}s (Jina rate limit)…")
            time.sleep(INTER_QUERY_DELAY)

    # ── Adversarial evaluation loop ────────────────────────────────────────────
    print(f"\n\n{SEP2}")
    print("  ADVERSARIAL / SAFETY EVALUATION")
    print(f"{SEP2}\n")

    adv_results: list[QueryResult] = []
    for idx, q in enumerate(ADVERSARIAL_QUESTIONS, 1):
        qr = evaluate_query(idx, q, TOTAL_ADV, is_adversarial=True)
        adv_results.append(qr)
        if idx < TOTAL_ADV:
            print(f"  waiting {INTER_QUERY_DELAY}s (Jina rate limit)…")
            time.sleep(INTER_QUERY_DELAY)

    print_aggregate(fiqh_results, adv_results)

    sys.stdout = _real_stdout

    report = buf.getvalue()
    REPORT_PATH.write_text(report, encoding="utf-8")
    _real_stdout.write(f"\nReport saved → {REPORT_PATH}\n")
    _real_stdout.flush()

    errors = sum(1 for r in fiqh_results if r.error)
    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
