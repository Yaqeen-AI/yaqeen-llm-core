"""
quran_rag/data_fetcher.py

Async client for the QuranicHub REST API.
Used at ingestion time (notebooks / scripts) — NOT at query time.

The retriever (retriever.py) hits Qdrant Cloud directly;
this module is only needed when building or updating the index.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

_DEFAULT_BASE_URL = "https://api.quranhub.com"


# ── Standalone helpers ───────────────────────────────────────────────────────


async def fetch_surah_map(base_url: str = _DEFAULT_BASE_URL) -> Dict[int, Dict]:
    """
    Fetch metadata for all 114 surahs.

    Returns a dict keyed by int surah number:
        {
            1: {
                "name_arabic": "الفاتحة",
                "name_english": "Al-Fatiha",
                "name_translation": "The Opening",
                "revelation_type": "Meccan",
                "number_of_ayahs": 7,
            },
            ...
        }

    Raises RuntimeError if the endpoint is unreachable.
    """
    url = f"{base_url.rstrip('/')}/v1/surah/"
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Could not fetch surah metadata from {url}: {exc}") from exc

    items = payload.get("data", []) if isinstance(payload, dict) else payload

    surah_map: Dict[int, Dict] = {}
    for item in items:
        n = item.get("number")
        if n is None:
            continue

        name_val = item.get("name", "")
        eng_val = item.get("englishName", "")

        arabic_name = name_val if isinstance(name_val, str) else (
            name_val.get("arabic", "") if isinstance(name_val, dict) else ""
        )
        english_name = eng_val if isinstance(eng_val, str) else (
            eng_val.get("english", "") if isinstance(eng_val, dict) else ""
        )

        surah_map[int(n)] = {
            "name_arabic": arabic_name,
            "name_english": english_name,
            "name_translation": item.get("englishNameTranslation") or item.get("translation", ""),
            "revelation_type": item.get("revelationType") or item.get("revelation_type", ""),
            "number_of_ayahs": item.get("numberOfAyahs") or item.get("number_of_ayahs", 0),
        }

    if len(surah_map) != 114:
        raise ValueError(
            f"Expected 114 surahs but got {len(surah_map)}. "
            "Check the QuranicHub /v1/surah/ endpoint."
        )
    return surah_map


# ── Stateful fetcher (caches meta) ───────────────────────────────────────────


class QuranicDataFetcher(BaseModel):
    """
    Async fetcher for individual ayahs and ranges.
    Caches /v1/meta/ after first fetch to avoid redundant HTTP round-trips.

    Usage:
        fetcher = QuranicDataFetcher()
        await fetcher.fetch_quran_meta()   # pre-warm cache
        result = await fetcher.fetch_ayah_range_async(2, 255, 255, "ar.ibnkathir")
    """

    base_url: str = Field(default=_DEFAULT_BASE_URL)
    max_concurrent_requests: int = 5

    # Pydantic v2: private attrs via model_config or explicit __slots__
    # We store the cache as a plain instance attribute set in __init__
    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context) -> None:
        object.__setattr__(self, "_meta_cache", {})

    # ── Meta ────────────────────────────────────────────────────────────────

    async def fetch_quran_meta(self) -> Dict:
        """Fetch and cache structural metadata (juz / page / ruku boundaries)."""
        cache = object.__getattribute__(self, "_meta_cache")
        if cache:
            return cache

        url = f"{self.base_url.rstrip('/')}/v1/meta/"
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, timeout=15.0)
                resp.raise_for_status()
                raw = resp.json()
                data = raw.get("data", raw)
                object.__setattr__(self, "_meta_cache", data)
                return data
            except Exception as exc:
                print(f"[WARN] Could not fetch Quran meta: {exc}")
                return {}

    # ── Single ayah ─────────────────────────────────────────────────────────

    async def fetch_single_ayah(
        self,
        client: httpx.AsyncClient,
        surah: int,
        ayah: int,
        edition: str,
    ) -> Dict:
        url = f"{self.base_url.rstrip('/')}/v1/ayah/{surah}:{ayah}/{edition}"

        for attempt in range(3):
            try:
                resp = await client.get(url, timeout=10.0)
                if resp.status_code == 429:
                    await asyncio.sleep((attempt + 1) * 2)
                    continue
                resp.raise_for_status()
                data = resp.json().get("data", {})
                text = data.get("text")
                return {
                    "ayah_num": ayah,
                    "text": text.strip() if text else None,
                    "surah": surah,
                    "juz": data.get("juz"),
                    "page": data.get("page"),
                    "ruku": data.get("ruku"),
                    "hizbQuarter": data.get("hizbQuarter"),
                    "sajda": data.get("sajda", False),
                }
            except Exception:
                if attempt == 2:
                    return {"ayah_num": ayah, "text": "[ERROR]", "surah": surah}
                await asyncio.sleep(1)

        return {"ayah_num": ayah, "text": "[ERROR]", "surah": surah}

    # ── Ayah range ──────────────────────────────────────────────────────────

    async def fetch_ayah_range_async(
        self,
        surah_number: int,
        ayah_from: int,
        ayah_to: int,
        edition: str,
    ) -> Optional[Dict]:
        """
        Fetch all ayahs in [ayah_from, ayah_to] for a given edition.
        Merges back empty texts with the previous valid text
        (handles tafsir editions that group consecutive ayahs).
        """
        ayah_numbers = list(range(ayah_from, ayah_to + 1))
        limits = httpx.Limits(max_connections=self.max_concurrent_requests)

        async with httpx.AsyncClient(limits=limits) as client:
            results = await asyncio.gather(
                *[self.fetch_single_ayah(client, surah_number, n, edition) for n in ayah_numbers]
            )

        final_ayahs: List[Dict] = []
        last_valid_text = ""
        juz = page = ruku = hizb = None

        for r in results:
            current_text = r.get("text")
            if current_text and current_text != "[ERROR]":
                last_valid_text = current_text
            if juz is None and r.get("juz"):
                juz = r.get("juz")
                page = r.get("page")
                ruku = r.get("ruku")
                hizb = r.get("hizbQuarter")

            final_ayahs.append({
                "ayah_num": r["ayah_num"],
                "text": last_valid_text,
                "is_merged": not bool(current_text),
            })

        return {
            "surah": surah_number,
            "range": f"{surah_number}:{ayah_from}-{ayah_to}",
            "full_text": " ".join(a["text"] for a in final_ayahs if a["text"]),
            "ayahs": final_ayahs,
            "edition": edition,
            "is_merged_content": any(a["is_merged"] for a in final_ayahs),
            "juz": juz,
            "page": page,
            "ruku": ruku,
            "hizbQuarter": hizb,
        }
