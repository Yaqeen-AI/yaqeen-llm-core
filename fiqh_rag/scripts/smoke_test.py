"""
Smoke test — RAG evaluation across 35 Fiqh questions (5 difficulty tiers)
+ 12 adversarial out-of-domain queries.

Run:  python -m scripts.smoke_test

Evaluation metrics (TABLE II):
  Retrieval  — Recall@5, Precision@5, Hit Rate@5, MRR
               95% bootstrap confidence intervals on every retrieval metric
  Reranking  — NDCG
  System     — Latency P95  (end-to-end: retrieval + generation)

Reports written to:
  smoke_report.md          — human-readable Markdown
  eval_output/eval_results_<ts>.csv
  eval_output/eval_aggregate_<ts>.json
"""

import csv
import io
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.config import GOOGLE_API_KEY, JINA_API_KEY, RELEVANCE_THRESHOLD as _CONFIG_RELEVANCE_THRESHOLD
from scripts.benchmark_data import (
    ALL_FIQH_QUERIES,
    ADVERSARIAL_QUERIES,
    BenchmarkQuery,
)

# ── Constants ──────────────────────────────────────────────────────────────────

RELEVANCE_THRESHOLD   = _CONFIG_RELEVANCE_THRESHOLD   # shared with retriever pipeline
REJECTION_THRESHOLD   = 0.22   # top-1 < this → adversarial query correctly rejected

MAX_CONSECUTIVE_ERRORS = 3    # abort early after this many consecutive retrieval exceptions

BOOTSTRAP_SAMPLES     = 1000   # iterations for 95% CI estimation
OUTPUT_DIR            = Path(__file__).parent.parent / "eval_output"
REPORT_PATH           = Path(__file__).parent.parent / "smoke_report.md"
SNIPPET_LEN           = 300    # chars of chunk_text shown per result in console

# ── Module-level mutable state ─────────────────────────────────────────────────

_inter_query_delay = 12   # start at Jina free-tier minimum; bumped on 429s

# ── Benchmark data ─────────────────────────────────────────────────────────────

QUESTIONS             = [q.question for q in ALL_FIQH_QUERIES]
ADVERSARIAL_QUESTIONS = [q.question for q in ADVERSARIAL_QUERIES]
QUERY_META: dict[str, BenchmarkQuery] = {
    q.question: q for q in ALL_FIQH_QUERIES + ADVERSARIAL_QUERIES
}

TOTAL     = len(QUESTIONS)
TOTAL_ADV = len(ADVERSARIAL_QUESTIONS)
SEP       = "─" * 72
SEP2      = "═" * 72

# ── Per-query result container ─────────────────────────────────────────────────

@dataclass
class QueryResult:
    question: str
    is_adversarial: bool = False
    difficulty: str = ""
    # Retrieval
    hit: bool = False
    result_count: int = 0
    top1_rerank: float = 0.0
    mean_rerank_top3: float = 0.0
    score_spread: float = 0.0
    mazhab_coverage: int = 0
    chunk_diversity: float = 0.0
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
    # Error
    error: Optional[str] = None

    @property
    def total_latency(self) -> float:
        return self.retrieval_latency + self.generation_latency


# ── Graded relevance ───────────────────────────────────────────────────────────

def _graded_relevance(score: float) -> int:
    """3-tier graded relevance calibrated to Jina v3 reranker output range."""
    if score >= 0.40:
        return 2   # highly relevant
    if score >= RELEVANCE_THRESHOLD:
        return 1   # marginally relevant — aligned with binary threshold (was 0.20)
    return 0       # not relevant


def _is_relevant(score: float) -> bool:
    return score >= RELEVANCE_THRESHOLD


# ── Retrieval ranking metric helpers ──────────────────────────────────────────

def _precision_at_k(scores: list[float], k: int = 5) -> float:
    top = scores[:k]
    if not top:
        return 0.0
    return sum(1 for s in top if _is_relevant(s)) / k


def _recall_at_k(scores: list[float], k: int = 5) -> float:
    return 1.0 if any(_is_relevant(s) for s in scores[:k]) else 0.0


def _mrr_at_k(scores: list[float], k: int = 5) -> float:
    for i, s in enumerate(scores[:k]):
        if _is_relevant(s):
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k_graded(scores: list[float], k: int = 5) -> float:
    """NDCG@k using 3-tier graded relevance (0 / 1 / 2)."""
    rel   = [_graded_relevance(s) for s in scores[:k]]
    dcg   = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg  = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _chunk_diversity(results: list) -> float:
    """Fraction of unique (volume, page) pairs — detects duplicate chunk inflation."""
    if not results:
        return 0.0
    unique = len({(r.volume_id, r.book_page) for r in results})
    return unique / len(results)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    idx = max(0, math.ceil(0.95 * len(values)) - 1)
    return sorted(values)[idx]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bootstrap_ci(
    values: list[float],
    n_boot: int = BOOTSTRAP_SAMPLES,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Return (lower, upper) 95% CI via bootstrap resampling (stdlib only)."""
    if len(values) < 2:
        m = values[0] if values else 0.0
        return m, m
    rng   = random.Random(42)
    means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(n_boot)
    )
    lo = int((1 - ci) / 2 * n_boot)
    hi = min(int((1 + ci) / 2 * n_boot), n_boot - 1)
    return means[lo], means[hi]


# ── Display helpers ────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float = 1.0, width: int = 12) -> str:
    filled = int(round(value / max_val * width)) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _grade(score: float) -> str:
    if score >= 0.8: return "EXCELLENT"
    if score >= 0.6: return "GOOD"
    if score >= 0.4: return "FAIR"
    return "POOR"


def _ci_str(lo: float, hi: float) -> str:
    return f"[{lo:.3f}–{hi:.3f}]"


class _Tee(io.TextIOBase):
    def __init__(self, real, buf):
        # Keep the original reference directly — creating a new TextIOWrapper
        # sharing the same buffer causes "I/O operation on closed file" on
        # Windows when sys.stdout is later restored.
        self._real = real
        self._buf  = buf

    def write(self, s):
        try:
            self._real.write(s)
            self._real.flush()
        except Exception:
            pass
        self._buf.write(s)
        return len(s)

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass


# ── Core evaluation ────────────────────────────────────────────────────────────

def evaluate_query(
    idx: int,
    question: str,
    total: int,
    is_adversarial: bool = False,
) -> QueryResult:
    meta = QUERY_META.get(question)
    qr   = QueryResult(
        question=question,
        is_adversarial=is_adversarial,
        difficulty=meta.difficulty if meta else ("adversarial" if is_adversarial else ""),
    )

    label = f"[ADV {idx:02d}/{total}]" if is_adversarial else f"[{idx:02d}/{total}]"
    diff_tag = f"  ({qr.difficulty})" if qr.difficulty else ""
    print(f"\n{SEP}")
    print(f"{label}{diff_tag}  {question}")
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
    qr.chunk_diversity  = _chunk_diversity(results)
    topics = list({r.fiqh_topic for r in results if r.fiqh_topic})
    qr.topic_filter = topics or None

    # TABLE II — retrieval ranking metrics (graded NDCG, raised threshold)
    qr.recall_at_5    = _recall_at_k(scores)
    qr.precision_at_5 = _precision_at_k(scores)
    qr.mrr            = _mrr_at_k(scores)
    qr.ndcg_at_5      = _ndcg_at_k_graded(scores)

    print(f"  Retrieval   {qr.retrieval_latency:.2f}s  —  {qr.result_count} chunks")
    print(f"  Top-1 rerank    : {qr.top1_rerank:.4f}  {_bar(qr.top1_rerank)}")
    print(f"  Mean rerank top3: {qr.mean_rerank_top3:.4f}  {_bar(qr.mean_rerank_top3)}")
    print(f"  Score spread    : {qr.score_spread:.4f}  |  Chunk diversity: {qr.chunk_diversity:.3f}")
    print(f"  Recall@5        : {qr.recall_at_5:.4f}  Precision@5: {qr.precision_at_5:.4f}")
    print(f"  MRR             : {qr.mrr:.4f}  NDCG@5 (graded): {qr.ndcg_at_5:.4f}")
    print(f"  Mazhab coverage : {qr.mazhab_coverage}  {sorted(all_mazhabs)}")
    print(f"  Topics in docs  : {topics or '—'}")
    print()

    for i, r in enumerate(results, 1):
        snippet  = r.chunk_text[:SNIPPET_LEN].replace("\n", " ")
        if len(r.chunk_text) > SNIPPET_LEN:
            snippet += "…"
        mazhab_s = "  ".join(r.mazhabs) if r.mazhabs else "—"
        grade    = "✓" if _is_relevant(r.rerank_score) else "✗"
        print(f"  ── Doc {i} ({grade}) ──────────────────────────────────────────────")
        print(f"  Ref    : {r.short_ref()}")
        print(f"  Rerank : {r.rerank_score:.4f} (grade={_graded_relevance(r.rerank_score)})"
              f"  |  Qdrant: {r.qdrant_score:.4f}")
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
    print(f"  E2E total   {qr.total_latency:.2f}s")

    return qr


# ── CSV / JSON export ──────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "question", "difficulty", "hit", "result_count",
    "recall_at_5", "precision_at_5", "mrr", "ndcg_at_5",
    "chunk_diversity", "top1_rerank", "mean_rerank_top3", "score_spread",
    "retrieval_latency", "generation_latency",
]


def _export_results(
    fiqh_results: list[QueryResult],
    adv_results:  list[QueryResult],
    ci_map:       dict,
    timestamp:    str,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Per-query CSV
    csv_path = OUTPUT_DIR / f"eval_results_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in fiqh_results:
            writer.writerow({field: getattr(r, field, "") for field in _CSV_FIELDS})

    # Aggregate JSON with confidence intervals
    def _ci_dict(key: str) -> dict:
        lo, hi = ci_map.get(key, (0.0, 0.0))
        return {"ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    aggregate = {
        "timestamp":     timestamp,
        "n_fiqh":        len(fiqh_results),
        "n_adversarial": len(adv_results),
        "thresholds": {
            "relevance": RELEVANCE_THRESHOLD,
            "rejection": REJECTION_THRESHOLD,
        },
        "retrieval": {
            "recall_at_5":     {"mean": round(_avg([r.recall_at_5    for r in fiqh_results]), 4), **_ci_dict("recall")},
            "precision_at_5":  {"mean": round(_avg([r.precision_at_5 for r in fiqh_results]), 4), **_ci_dict("prec")},
            "mrr":             {"mean": round(_avg([r.mrr             for r in fiqh_results]), 4), **_ci_dict("mrr")},
            "ndcg_at_5":       {"mean": round(_avg([r.ndcg_at_5      for r in fiqh_results]), 4), **_ci_dict("ndcg")},
            "hit_rate":        round(sum(1 for r in fiqh_results if r.hit) / max(len(fiqh_results), 1), 4),
            "chunk_diversity": round(_avg([r.chunk_diversity for r in fiqh_results if r.hit]), 4),
        },
        "safety": {
            "rejection_rate": round(
                sum(1 for r in adv_results if not r.hit or r.top1_rerank < REJECTION_THRESHOLD)
                / max(len(adv_results), 1), 4
            ),
        },
        "per_difficulty": {},
    }

    for diff in ("easy", "medium", "hard", "colloquial"):
        group = [r for r in fiqh_results if r.difficulty == diff]
        if group:
            aggregate["per_difficulty"][diff] = {
                "n":          len(group),
                "recall_at_5":    round(_avg([r.recall_at_5    for r in group]), 4),
                "precision_at_5": round(_avg([r.precision_at_5 for r in group]), 4),
                "mrr":            round(_avg([r.mrr             for r in group]), 4),
                "ndcg_at_5":      round(_avg([r.ndcg_at_5      for r in group]), 4),
            }

    json_path = OUTPUT_DIR / f"eval_aggregate_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    print(f"\n  Reports -> {csv_path}")
    print(f"           {json_path}")


# ── Aggregate report ───────────────────────────────────────────────────────────

def print_aggregate(fiqh_results: list[QueryResult], adv_results: list[QueryResult]) -> dict:
    hits   = [r for r in fiqh_results if r.hit]
    errors = [r for r in fiqh_results if r.error]

    # Collect raw metric lists for CI computation
    recall_vals  = [r.recall_at_5    for r in fiqh_results]
    prec_vals    = [r.precision_at_5 for r in fiqh_results]
    mrr_vals     = [r.mrr            for r in fiqh_results]
    ndcg_vals    = [r.ndcg_at_5      for r in fiqh_results]

    # Bootstrap 95% confidence intervals
    ci_recall = _bootstrap_ci(recall_vals)
    ci_prec   = _bootstrap_ci(prec_vals)
    ci_mrr    = _bootstrap_ci(mrr_vals)
    ci_ndcg   = _bootstrap_ci(ndcg_vals)

    ci_map = {
        "recall": ci_recall, "prec": ci_prec, "mrr": ci_mrr, "ndcg": ci_ndcg,
    }

    # Aggregated averages
    hit_rate    = len(hits) / len(fiqh_results) if fiqh_results else 0
    avg_recall5 = _avg(recall_vals)
    avg_prec5   = _avg(prec_vals)
    avg_mrr     = _avg(mrr_vals)
    avg_ndcg5   = _avg(ndcg_vals)

    # Latency P95
    e2e_latencies = [r.total_latency for r in fiqh_results if r.answer]
    p95_e2e       = _p95(e2e_latencies)

    print(f"\n\n{SEP2}")
    print("  AGGREGATE EVALUATION REPORT  —  TABLE II Metrics")
    print(f"  Relevance threshold: {RELEVANCE_THRESHOLD}")
    print(SEP2)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    print("\n### Retrieval Metrics  (95% CI via bootstrap)\n")
    print(f"  {'Metric':<38} {'Value':>9}  {'95% CI':<16}  Visual          Grade")
    print(f"  {'─'*38} {'─'*9}  {'─'*16}  {'─'*14}  {'─'*9}")
    print(f"  {'Recall@5':<38} {avg_recall5:>8.1%}  {_ci_str(*ci_recall):<16}  "
          f"{_bar(avg_recall5)}  {_grade(avg_recall5)}")
    print(f"  {'Precision@5':<38} {avg_prec5:>8.1%}  {_ci_str(*ci_prec):<16}  "
          f"{_bar(avg_prec5)}  {_grade(avg_prec5)}")
    print(f"  {'Hit Rate@5':<38} {hit_rate:>8.1%}  {'':16}  "
          f"{_bar(hit_rate)}  {_grade(hit_rate)}")
    print(f"  {'MRR':<38} {avg_mrr:>9.4f}  {_ci_str(*ci_mrr):<16}  "
          f"{_bar(avg_mrr)}  {_grade(avg_mrr)}")

    # ── Reranking (NDCG) ───────────────────────────────────────────────────────
    print("\n### Reranking Metrics\n")
    print(f"  {'Metric':<38} {'Value':>9}  {'95% CI':<16}  Visual          Grade")
    print(f"  {'─'*38} {'─'*9}  {'─'*16}  {'─'*14}  {'─'*9}")
    print(f"  {'NDCG':<38} {avg_ndcg5:>9.4f}  {_ci_str(*ci_ndcg):<16}  "
          f"{_bar(avg_ndcg5)}  {_grade(avg_ndcg5)}")

    # ── Per-difficulty breakdown ───────────────────────────────────────────────
    print("\n### Per-Difficulty Breakdown\n")
    print(f"  {'Difficulty':<14} {'n':>3}  {'R@5':>6}  {'P@5':>6}  {'HR@5':>6}  "
          f"{'MRR':>6}  {'NDCG':>6}")
    print("  " + "─" * 58)
    for diff in ("easy", "medium", "hard", "colloquial"):
        group = [r for r in fiqh_results if r.difficulty == diff]
        if not group:
            continue
        hr_group = sum(1 for r in group if r.hit) / len(group)
        print(f"  {diff:<14} {len(group):>3}  "
              f"{_avg([r.recall_at_5 for r in group]):>6.3f}  "
              f"{_avg([r.precision_at_5 for r in group]):>6.3f}  "
              f"{hr_group:>6.3f}  "
              f"{_avg([r.mrr for r in group]):>6.3f}  "
              f"{_avg([r.ndcg_at_5 for r in group]):>6.3f}")

    # ── System / Latency ───────────────────────────────────────────────────────
    print("\n### System Metrics\n")
    print(f"  {'Metric':<38} {'Value':>9}")
    print(f"  {'─'*38} {'─'*9}")
    print(f"  {'Latency P95 (E2E)':<38} {p95_e2e:>8.2f}s")

    # ── Per-query summary table ────────────────────────────────────────────────
    print(f"\n### Per-Query Summary\n")
    hdr = (f"  {'#':>3}  {'Diff':<12}  {'Hit':>3}  {'R@5':>4}  {'P@5':>4}  "
           f"{'MRR':>4}  {'NDCG':>4}  {'Lat':>5}  Question")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for i, r in enumerate(fiqh_results, 1):
        hit_s   = "Y" if r.hit else "N"
        q_short = r.question[:40] + ("…" if len(r.question) > 40 else "")
        print(f"  {i:>3}  {r.difficulty:<12}  {hit_s:>3}  {r.recall_at_5:>4.2f}  "
              f"{r.precision_at_5:>4.2f}  {r.mrr:>4.2f}  {r.ndcg_at_5:>4.2f}  "
              f"{r.total_latency:>4.1f}s  {q_short}")

    print(f"\n{SEP2}")
    print(f"  RESULT: {len(hits)} retrieval hits  /  {len(errors)} errors  /  {TOTAL} total")
    print(SEP2)

    return ci_map


# ── Main ───────────────────────────────────────────────────────────────────────

def _preflight() -> bool:
    """Fail fast if API keys are missing before spending time on pipeline init."""
    missing = [name for name, val in [("GOOGLE_API_KEY", GOOGLE_API_KEY),
                                       ("JINA_API_KEY",   JINA_API_KEY)]
               if not val]
    if missing:
        print(f"[FATAL] Missing env vars: {', '.join(missing)} — set them in .env and retry")
        return False
    return True


def _run_loop(
    questions: list[str],
    total: int,
    is_adversarial: bool,
) -> list[QueryResult]:
    """Run evaluate_query for a list of questions with adaptive rate-limit handling.

    Aborts early if MAX_CONSECUTIVE_ERRORS retrieval exceptions occur back-to-back
    (distinguishes hard errors from low-relevance misses, which are not errors).
    """
    global _inter_query_delay
    results: list[QueryResult] = []
    consecutive_errors = 0

    for idx, q in enumerate(questions, 1):
        qr = evaluate_query(idx, q, total, is_adversarial=is_adversarial)
        results.append(qr)

        if qr.error:
            consecutive_errors += 1
            err_lower = qr.error.lower()
            # Bump delay on rate-limit signals from retrieval/reranking
            if any(x in err_lower for x in ("429", "rate", "quota", "resource_exhausted")):
                _inter_query_delay = min(_inter_query_delay * 2, 120)
                print(f"  [rate-limit] Increasing inter-query delay to {_inter_query_delay}s")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\n[ABORT] {consecutive_errors} consecutive retrieval errors — "
                      f"check API keys / network and restart.")
                print(f"  Last error: {qr.error}")
                break
        else:
            consecutive_errors = 0

        if idx < total:
            print(f"  waiting {_inter_query_delay}s (API rate limit)…")
            time.sleep(_inter_query_delay)

    return results


def main() -> None:
    global _inter_query_delay

    # ── Pre-flight: fail fast before touching any API ──────────────────────────
    if not _preflight():
        sys.exit(1)

    buf = io.StringIO()
    _real_stdout = sys.__stdout__
    sys.stdout = _Tee(sys.__stdout__, buf)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"# FiqhRAG Smoke Report — Full Pipeline Evaluation\n"
          f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"**Questions:** {TOTAL} Fiqh (easy/medium/hard/colloquial)"
          f" + {TOTAL_ADV} adversarial  "
          f"|  **Mode:** retrieval + generation\n")
    print(f"**Metrics (TABLE II):** Recall@5, Precision@5, Hit Rate@5, MRR (Retrieval)  ·  "
          f"NDCG (Reranking)  ·  Latency P95 (System)\n")
    print(f"**Relevance threshold:** rerank_score ≥ {RELEVANCE_THRESHOLD}  "
          f"|  **Rejection threshold:** top-1 rerank < {REJECTION_THRESHOLD}\n")
    print("Loading pipeline…")

    try:
        from core.graph import fiqh_graph  # noqa: F401  — warm up graph + retriever
        print("Pipeline: ready\n")
    except Exception as e:
        print(f"[FATAL] Pipeline init failed: {e}")
        sys.stdout = _real_stdout
        sys.exit(1)

    # ── Fiqh evaluation loop ───────────────────────────────────────────────────
    fiqh_results = _run_loop(QUESTIONS, TOTAL, is_adversarial=False)

    # ── Adversarial evaluation loop ────────────────────────────────────────────
    print(f"\n\n{SEP2}")
    print("  ADVERSARIAL / SAFETY EVALUATION")
    print(f"{SEP2}\n")

    adv_results = _run_loop(ADVERSARIAL_QUESTIONS, TOTAL_ADV, is_adversarial=True)

    ci_map = print_aggregate(fiqh_results, adv_results)

    sys.stdout = _real_stdout

    report = buf.getvalue()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved -> {REPORT_PATH}")

    _export_results(fiqh_results, adv_results, ci_map, timestamp)
    sys.exit(0)   # always exit clean — per-query errors are logged, not fatal


if __name__ == "__main__":
    main()
