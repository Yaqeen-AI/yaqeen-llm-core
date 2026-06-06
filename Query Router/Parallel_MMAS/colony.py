"""
Colony — An isolated MMAS colony per sub-query.

Each Colony owns:
  - A PheromoneTable for its worker pool
  - A heuristic function (keyword-based relevance estimation)
  - The logic for probabilistic worker selection using pheromone trails

Colonies are instantiated by the Dispatcher, one per sub-query.
"""

import re
import random
import logging
from typing import Dict, List, Optional
from pheromone import PheromoneTable, MMASConfig

logger = logging.getLogger("mmas.colony")

# ═══════════════════════════════════════════════════════════════════════════════
# Keyword sets — used as heuristic η(j) in the ACO probability formula
# ═══════════════════════════════════════════════════════════════════════════════

QURAN_KEYWORDS = {
    "quran", "quranic", "ayah", "ayat", "surah", "sura", "verse", "verses",
    "tafsir", "revelation", "juz", "hizb", "makkah", "madinah", "recitation",
    "tilawah", "mushaf", "makki", "madani",
    "قرآن", "قرآنية", "قرآني", "آية", "آيات", "سورة", "سور",
    "تفسير", "وحي", "تلاوة", "مصحف", "جزء", "حزب", "تنزيل",
    "مكية", "مدنية", "الكتاب",
}

HADITH_KEYWORDS = {
    "hadith", "hadiths", "hadeeth", "prophet", "prophetic", "sunnah",
    "narration", "narrated", "narrator", "sahih", "bukhari", "muslim",
    "tirmidhi", "dawud", "nasa'i", "ibn majah", "musnad", "isnad",
    "chain", "rawi", "muhaddith", "athar", "sanad",
    "حديث", "أحاديث", "نبي", "النبي", "الرسول", "رسول", "نبوي",
    "سنة", "سنن", "رواية", "راوي", "رواة", "صحيح", "بخاري",
    "مسلم", "ترمذي", "داود", "نسائي", "ماجه", "مسند", "إسناد",
    "سند", "محدث", "أثر", "آثار",
}

FIQH_KEYWORDS = {
    "fiqh", "ruling", "rulings", "halal", "haram", "fatwa", "fatwas",
    "jurisprudence", "sharia", "shariah", "madhab", "madhhab", "hanafi",
    "maliki", "shafii", "hanbali", "wajib", "obligatory", "mustahab",
    "recommended", "makruh", "disliked", "mubah", "permissible",
    "forbidden", "ibadah", "worship", "muamalat", "prayer", "salah",
    "zakat", "fasting", "hajj", "nikah", "talaq", "inheritance",
    "purification", "wudu", "ghusl", "tayammum",
    "فقه", "فقهي", "فقهية", "حكم", "أحكام", "حلال", "حرام",
    "فتوى", "فتاوى", "شريعة", "مذهب", "مذاهب", "حنفي", "مالكي",
    "شافعي", "حنبلي", "واجب", "مستحب", "مكروه", "مباح",
    "عبادة", "عبادات", "معاملات", "صلاة", "زكاة", "صيام", "صوم",
    "حج", "نكاح", "طلاق", "ميراث", "إرث", "فرائض",
    "طهارة", "وضوء", "غسل", "تيمم", "الموسوعة",
}

GREETING_PATTERNS = {
    "hello", "hi", "hey", "good morning", "good evening", "good night",
    "thanks", "thank you", "bye", "goodbye", "who are you", "what are you",
    "help",
    "سلام", "السلام", "مرحبا", "أهلا", "شكرا", "جزاك", "بارك",
    "وداعا", "صباح", "مساء", "من أنت", "ما أنت", "مساعدة",
}

WORKER_KEYWORD_MAP = {
    "quran_agent": QURAN_KEYWORDS,
    "hadith_agent": HADITH_KEYWORDS,
    "fiqh_agent": FIQH_KEYWORDS,
}

DOMAIN_WORKERS = ["quran_agent", "hadith_agent", "fiqh_agent"]
ALL_WORKERS = DOMAIN_WORKERS + ["direct_answer"]


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer — shared utility
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> set:
    """Split query into lowercase word tokens + bigrams for phrase matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    words = text.split()
    tokens = set(words)
    for i in range(len(words) - 1):
        tokens.add(f"{words[i]} {words[i+1]}")
    return tokens


def is_greeting(tokens: set) -> bool:
    """Check if the token set contains greeting patterns."""
    return bool(tokens & GREETING_PATTERNS)


# ═══════════════════════════════════════════════════════════════════════════════
# Colony
# ═══════════════════════════════════════════════════════════════════════════════

class Colony:
    """
    Isolated MMAS colony for a single sub-query.

    Responsibilities:
      1. Compute heuristic values η(j) from keyword matching
      2. Select workers probabilistically: P(j) ∝ τ(j)^α · η(j)^β
      3. Own a PheromoneTable with independent trails
      4. Accept pheromone updates from Inspector scores
    """

    def __init__(
        self,
        sub_query: str,
        colony_id: str = "",
        config: Optional[MMASConfig] = None,
    ):
        self.sub_query = sub_query
        self.colony_id = colony_id or sub_query[:30]
        self.tokens = tokenize(sub_query)
        self.is_greeting_query = is_greeting(self.tokens)

        # Determine worker pool
        if self.is_greeting_query:
            self.worker_names = ["direct_answer"]
        else:
            self.worker_names = list(DOMAIN_WORKERS)

        # Initialize pheromone table
        self.pheromone = PheromoneTable(
            worker_names=self.worker_names,
            config=config,
        )

        logger.info(
            f"Colony '{self.colony_id}' created: "
            f"workers={self.worker_names}, greeting={self.is_greeting_query}"
        )

    def compute_heuristics(self) -> Dict[str, float]:
        """
        Compute heuristic desirability η(j) for each worker based on
        keyword overlap with the sub-query.

        Returns a dict of worker_name -> heuristic score.
        Higher score = more keywords matched = more desirable.
        """
        if self.is_greeting_query:
            return {"direct_answer": 1.0}

        heuristics = {}
        for name in self.worker_names:
            keywords = WORKER_KEYWORD_MAP.get(name, set())
            overlap = len(self.tokens & keywords)
            # Minimum heuristic of 0.1 to keep all workers explorable
            heuristics[name] = max(overlap + 0.1, 0.1)

        return heuristics

    def select_workers(self, min_workers: int = 1) -> List[str]:
        """
        Select workers using MMAS probabilistic path selection.

        At least `min_workers` are selected. Workers with probability > 0.15
        are always included. Below that threshold, they may be randomly
        included based on their probability.

        For greeting queries, always returns ["direct_answer"].
        """
        if self.is_greeting_query:
            return ["direct_answer"]

        heuristics = self.compute_heuristics()
        probabilities = self.pheromone.get_probabilities(heuristics)

        selected = []
        for name, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
            if prob > 0.15:
                # High-probability worker: always include
                selected.append(name)
            elif random.random() < prob * 3:
                # Low-probability worker: stochastic inclusion
                selected.append(name)

        # Guarantee minimum selection
        if len(selected) < min_workers:
            remaining = [w for w in self.worker_names if w not in selected]
            remaining.sort(key=lambda w: probabilities.get(w, 0), reverse=True)
            selected.extend(remaining[:min_workers - len(selected)])

        logger.info(
            f"Colony '{self.colony_id}' selected: {selected} "
            f"(probs: {', '.join(f'{k}={v:.3f}' for k, v in probabilities.items())})"
        )
        return selected

    def update_pheromones(self, scores: Dict[str, float]) -> dict:
        """
        Full MMAS pheromone update cycle:
          1. Evaporate all trails
          2. Deposit pheromone for the best-performing worker
          3. Update bounds based on best fitness
          4. Log the bounding state

        Args:
            scores: dict of worker_name -> Inspector quality score (0.0 to 1.0)

        Returns:
            A log dict capturing the pheromone state for monitoring.
        """
        if not scores:
            return self.pheromone.snapshot()

        # Step 1: Evaporation
        self.pheromone.evaporate()

        # Step 2: Find iteration-best and deposit (MMAS: only best deposits)
        best_worker = max(scores, key=scores.get)
        best_score = scores[best_worker]
        self.pheromone.deposit(best_worker, best_score)

        # Step 3: Update bounds
        if best_score > 0:
            self.pheromone.update_bounds(1.0 / best_score)

        # Step 4: Build log entry
        snapshot = self.pheromone.snapshot()
        log_entry = {
            "colony_id": self.colony_id,
            "sub_query": self.sub_query[:50],
            "scores": scores,
            "best_worker": best_worker,
            "best_score": best_score,
            **snapshot,
        }
        logger.info(
            f"Colony '{self.colony_id}' pheromone update: "
            f"best={best_worker}({best_score:.4f}), "
            f"tau_min={snapshot['tau_min']:.4f}, tau_max={snapshot['tau_max']:.4f}"
        )
        return log_entry

    def get_trails(self) -> Dict[str, float]:
        """Return current pheromone trail snapshot."""
        return self.pheromone.get_trails()

    def merge_from_colony(self, other: "Colony", weight: float = 0.3) -> None:
        """Inter-colony synchronization: blend trails from another colony."""
        other_trails = other.get_trails()
        # Only merge trails for workers we share
        self.pheromone.merge_from(other_trails, weight)
