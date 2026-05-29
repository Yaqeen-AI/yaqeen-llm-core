"""
Benchmark dataset for Arabic Fiqh RAG evaluation.

Five difficulty tiers:
  easy        — standard MSA fiqh, well-covered topics
  medium      — specific rulings, cross-mazhab comparisons
  hard        — multi-hop reasoning, implicit intent, cross-mazhab conflict
  colloquial  — colloquial dialect, typos, informal spelling
  adversarial — out-of-domain; system should reject / return no results
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkQuery:
    question:        str
    difficulty:      str   # easy | medium | hard | colloquial | adversarial
    category:        str   # Arabic topic label or "out-of-domain"
    should_retrieve: bool  # True = expect relevant fiqh chunks; False = expect rejection


# ── Easy (10) ─────────────────────────────────────────────────────────────────
# Standard Modern Standard Arabic, central encyclopedia topics, unambiguous.

EASY_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery("متى يجوز جمع الصلاة؟",                    "easy", "صلاة",  True),
    BenchmarkQuery("ما حكم الزكاة على الذهب والفضة؟",         "easy", "زكاة",  True),
    BenchmarkQuery("ما شروط وجوب الحج؟",                       "easy", "حج",    True),
    BenchmarkQuery("ما حكم صلاة الجماعة؟",                     "easy", "صلاة",  True),
    BenchmarkQuery("ما حكم الوضوء بالماء المستعمل؟",           "easy", "طهارة", True),
    BenchmarkQuery("ما حكم قراءة القرآن للحائض؟",              "easy", "طهارة", True),
    BenchmarkQuery("ما حكم صوم من أفطر ناسياً في رمضان؟",     "easy", "صيام",  True),
    BenchmarkQuery("حكم كفالة اليتيم",                          "easy", "أسرة",  True),
    BenchmarkQuery("حكم تربية الكلاب",                          "easy", "آداب",  True),
    BenchmarkQuery("ما هي أركان الصلاة؟",                      "easy", "صلاة",  True),
]

# ── Medium (10) ───────────────────────────────────────────────────────────────
# More specific rulings, cross-mazhab comparisons, one-hop reasoning.

MEDIUM_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery("ما شروط صحة عقد البيع عند المذاهب الأربعة؟",     "medium", "بيوع",  True),
    BenchmarkQuery("ما حكم الصلاة في الأرض المغصوبة؟",               "medium", "صلاة",  True),
    BenchmarkQuery("ما حكم الطلاق في حالة الغضب الشديد؟",            "medium", "طلاق",  True),
    BenchmarkQuery("حكم صلاة الجمعة مع صلاة العيد",                   "medium", "صلاة",  True),
    BenchmarkQuery("حكم سماع الأغاني",                                 "medium", "آداب",  True),
    BenchmarkQuery("هل استخدام السبحة بدعة؟",                         "medium", "عبادة", True),
    BenchmarkQuery("ما حكم إخراج زكاة الفطر نقداً؟",                 "medium", "زكاة",  True),
    BenchmarkQuery("هل يشترط النية في الوضوء عند المذاهب الأربعة؟",  "medium", "طهارة", True),
    BenchmarkQuery("ما الفرق بين الوضوء والغسل في الطهارة؟",         "medium", "طهارة", True),
    BenchmarkQuery("ما حكم الصلاة في الطائرة؟",                       "medium", "صلاة",  True),
]

# ── Hard (10) ─────────────────────────────────────────────────────────────────
# Multi-hop reasoning, implicit intent, cross-mazhab conflict, edge cases.

HARD_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        "هل يصح رفع الإصبع في بداية التشهد أم عند قول أشهد أن لا إله إلا الله في الصلاة؟",
        "hard", "صلاة", True,
    ),
    BenchmarkQuery(
        "رجل توضأ ثم لبس الجوارب ثم خلعهما ثم لبسهما مجدداً فهل يمسح عليهما؟",
        "hard", "طهارة", True,
    ),
    BenchmarkQuery(
        "شخص أسلم في منتصف الحول؛ هل تجب عليه زكاة المال من يوم إسلامه أم ينتظر حولاً كاملاً؟",
        "hard", "زكاة", True,
    ),
    BenchmarkQuery(
        "رجل أكل ناسياً في رمضان ثم تذكر فأكل متعمداً ظناً أن صومه قد فسد، ما حكمه؟",
        "hard", "صيام", True,
    ),
    BenchmarkQuery(
        "ما حكم من شك في عدد ركعاته بعد الانتهاء من الصلاة والسلام؟",
        "hard", "صلاة", True,
    ),
    BenchmarkQuery(
        "ما حكم بيع شيء لا يملكه البائع بنية شرائه فور إتمام العقد؟",
        "hard", "بيوع", True,
    ),
    BenchmarkQuery(
        "ما الحكم إذا أحدث الإمام في أثناء الصلاة واستخلف مأموماً ليكمل بهم؟",
        "hard", "صلاة", True,
    ),
    BenchmarkQuery("هل الزنا ينقض الوضوء؟",                            "hard", "طهارة", True),
    BenchmarkQuery(
        "ما حكم الصلاة خلف إمام من مذهب مختلف مع وجود خلاف في الفروض؟",
        "hard", "صلاة", True,
    ),
    BenchmarkQuery(
        "هل أغنية طلع البدر علينا كانت موجودة تاريخياً أم أنها بدعة متأخرة؟",
        "hard", "آداب", True,
    ),
]

# ── Colloquial / Noisy (5) ────────────────────────────────────────────────────
# Colloquial dialect, spelling errors, informal phrasing — stress-tests Arabic
# normalization and BM25 tokenization.

COLLOQUIAL_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery("هل النوم لمده قصيره يننقض الوضوء",  "colloquial", "طهارة", True),
    BenchmarkQuery("ما هي اركان الايمانن",               "colloquial", "عقيدة", True),
    BenchmarkQuery("ايه حكم الصلاه على الميت",           "colloquial", "صلاة",  True),
    BenchmarkQuery("ازاي اتوضأ صح؟",                     "colloquial", "طهارة", True),
    BenchmarkQuery("حكم لبس الخواتم للراجل",             "colloquial", "آداب",  True),
]

# ── Adversarial / Out-of-Domain (12) ──────────────────────────────────────────
# Non-Fiqh questions; the system should produce no relevant results or low
# rerank scores.  Last two are edge cases that touch religion tangentially.

ADVERSARIAL_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery("ما عاصمة فرنسا؟",                                              "adversarial", "out-of-domain", False),
    BenchmarkQuery("كيف أصنع كعكة الشوكولاتة؟",                                   "adversarial", "out-of-domain", False),
    BenchmarkQuery("ما هو أفضل برنامج لتحرير الصور؟",                             "adversarial", "out-of-domain", False),
    BenchmarkQuery("كيف أعالج ضغط الدم المرتفع؟",                                 "adversarial", "out-of-domain", False),
    BenchmarkQuery("من هو مخترع الهاتف؟",                                          "adversarial", "out-of-domain", False),
    BenchmarkQuery("ما هي أسرع سيارة في العالم؟",                                 "adversarial", "out-of-domain", False),
    BenchmarkQuery("اكتب لي كود Python لعمل REST API",                             "adversarial", "out-of-domain", False),
    BenchmarkQuery("ما هو الناتج المحلي الإجمالي للسعودية؟",                      "adversarial", "out-of-domain", False),
    BenchmarkQuery("كيف أزرع الطماطم في المنزل؟",                                 "adversarial", "out-of-domain", False),
    BenchmarkQuery("أعطيني مخططاً زمنياً لأحداث الحرب العالمية الثانية",         "adversarial", "out-of-domain", False),
    BenchmarkQuery("ما هي أعراض كوفيد-19 وكيف أتعامل معها؟",                     "adversarial", "out-of-domain", False),
    BenchmarkQuery("ما حكم استخدام الذكاء الاصطناعي في إصدار الفتاوى الفقهية؟", "adversarial", "out-of-domain", False),
]

# ── Aggregated exports ─────────────────────────────────────────────────────────

ALL_FIQH_QUERIES: list[BenchmarkQuery] = (
    EASY_QUERIES + MEDIUM_QUERIES + HARD_QUERIES + COLLOQUIAL_QUERIES
)

ALL_QUERIES: list[BenchmarkQuery] = ALL_FIQH_QUERIES + ADVERSARIAL_QUERIES
