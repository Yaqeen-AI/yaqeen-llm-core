# ============================================================
# YaqeenAI — Query Router
# ============================================================
# Routes user queries to the correct sub-RAG(s).
# V1: Keyword/regex-based (zero overhead, runs on CPU).
# V2: Can be upgraded to a small classifier model.
#
# Currently routes to Quran sub-RAG only (as per our test scope).
# Designed to be extended for Hadith, Tafsir sub-RAGs.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger

from app.preprocessing.arabic_normalizer import ArabicTextNormalizer


class SubRAG(str, Enum):
    """Available sub-RAG systems."""
    QURAN = "quran"
    HADITH = "hadith"
    TAFSIR = "tafsir"


@dataclass
class RoutingDecision:
    """Result of query routing."""
    targets: list[SubRAG]
    detected_language: str
    confidence: float
    reasoning: str
    detected_keywords: list[str] = field(default_factory=list)


class QueryRouter:
    """
    Rule-based query router for the Multi-RAG architecture.
    
    Routes user queries to the appropriate sub-RAG(s) based on:
    1. Arabic/English keyword detection
    2. Intent classification (simple rules)
    3. Language detection
    
    This is intentionally simple for V1.
    For V2, plug in a small classifier (e.g., fine-tuned DistilBERT).
    """

    # ─── Arabic Keywords ───
    _QURAN_KEYWORDS_AR = [
        "قرآن", "آية", "آيات", "سورة", "سور", "قرأ", "تلاوة",
        "مصحف", "جزء", "حزب", "ربع", "صفحة", "ركوع",
        "بسم الله", "الفاتحة", "البقرة", "آل عمران",
        "مكي", "مدني", "نزول", "أسباب النزول",
    ]

    _HADITH_KEYWORDS_AR = [
        "حديث", "أحاديث", "رواية", "رواه", "صحيح", "حسن", "ضعيف",
        "البخاري", "مسلم", "الترمذي", "أبو داود", "النسائي", "ابن ماجه",
        "إسناد", "متن", "سند", "راوي", "محدث",
        "النبي", "رسول الله", "صلى الله عليه وسلم",
    ]

    _TAFSIR_KEYWORDS_AR = [
        "تفسير", "شرح", "معنى", "تأويل", "تدبر",
        "ابن كثير", "الطبري", "القرطبي", "الجلالين",
        "البغوي", "السعدي", "الميسر",
    ]

    # ─── English Keywords ───
    _QURAN_KEYWORDS_EN = [
        "quran", "quran", "verse", "ayah", "ayat", "surah", "sura",
        "chapter", "juz", "hizb", "page", "recitation",
        "al-fatiha", "al-baqarah", "meccan", "medinan",
    ]

    _HADITH_KEYWORDS_EN = [
        "hadith", "narration", "bukhari", "muslim", "sahih",
        "hasan", "weak", "narrator", "chain", "isnad", "matn",
        "prophet", "messenger",
    ]

    _TAFSIR_KEYWORDS_EN = [
        "tafsir", "tafseer", "interpretation", "commentary",
        "explanation", "meaning", "ibn kathir", "tabari",
    ]

    def __init__(self):
        self._normalizer = ArabicTextNormalizer()

    def route(self, query: str) -> RoutingDecision:
        """
        Analyze the query and decide which sub-RAG(s) to query.
        
        Returns:
            RoutingDecision with target sub-RAGs and reasoning
        """
        detected_lang = self._normalizer.detect_language(query)
        query_lower = query.lower()
        query_normalized = self._normalizer.normalize_for_bm25(query)

        detected_keywords = []
        scores = {SubRAG.QURAN: 0.0, SubRAG.HADITH: 0.0, SubRAG.TAFSIR: 0.0}

        # ─── Check Arabic keywords ───
        for kw in self._QURAN_KEYWORDS_AR:
            if kw in query or kw in query_normalized:
                scores[SubRAG.QURAN] += 1.0
                detected_keywords.append(kw)

        for kw in self._HADITH_KEYWORDS_AR:
            if kw in query or kw in query_normalized:
                scores[SubRAG.HADITH] += 1.0
                detected_keywords.append(kw)

        for kw in self._TAFSIR_KEYWORDS_AR:
            if kw in query or kw in query_normalized:
                scores[SubRAG.TAFSIR] += 1.0
                detected_keywords.append(kw)

        # ─── Check English keywords ───
        for kw in self._QURAN_KEYWORDS_EN:
            if kw in query_lower:
                scores[SubRAG.QURAN] += 1.0
                detected_keywords.append(kw)

        for kw in self._HADITH_KEYWORDS_EN:
            if kw in query_lower:
                scores[SubRAG.HADITH] += 1.0
                detected_keywords.append(kw)

        for kw in self._TAFSIR_KEYWORDS_EN:
            if kw in query_lower:
                scores[SubRAG.TAFSIR] += 1.0
                detected_keywords.append(kw)

        # ─── Determine targets ───
        # If tafsir is requested, also include the source (Quran or Hadith)
        targets = []
        reasoning_parts = []

        if scores[SubRAG.TAFSIR] > 0:
            targets.append(SubRAG.TAFSIR)
            reasoning_parts.append("tafsir keywords detected")
            # Also add Quran if tafsir is about Quran
            if scores[SubRAG.QURAN] > 0 or scores[SubRAG.HADITH] == 0:
                targets.append(SubRAG.QURAN)
                reasoning_parts.append("+ quran source for tafsir context")

        if scores[SubRAG.QURAN] > 0 and SubRAG.QURAN not in targets:
            targets.append(SubRAG.QURAN)
            reasoning_parts.append("quran keywords detected")

        if scores[SubRAG.HADITH] > 0:
            targets.append(SubRAG.HADITH)
            reasoning_parts.append("hadith keywords detected")

        # ─── Default fallback: route to Quran ───
        if not targets:
            targets = [SubRAG.QURAN]
            reasoning_parts.append("no specific keywords → default to quran")

        total_score = sum(scores.values())
        confidence = min(total_score / 3.0, 1.0) if total_score > 0 else 0.3

        decision = RoutingDecision(
            targets=targets,
            detected_language=detected_lang,
            confidence=confidence,
            reasoning="; ".join(reasoning_parts),
            detected_keywords=detected_keywords,
        )

        logger.debug(
            f"Query routed: '{query[:50]}...' → "
            f"{[t.value for t in targets]} (confidence={confidence:.2f})"
        )

        return decision
