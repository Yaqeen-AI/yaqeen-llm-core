# ============================================================
# YaqeenAI — Quran API Client (Data Ingestion)
# ============================================================
# Fetches Quran data from api.alquran.cloud for ingestion.
# Handles: surahs, ayahs, editions, translations.
# Uses retry logic for robustness.

from __future__ import annotations

from typing import Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.models.schemas import QuranSurah, QuranAyah, QuranEdition


class QuranApiClient:
    """
    Async client for the Quran Hub API.
    
    Usage:
        async with QuranApiClient() as client:
            surah = await client.get_surah(1, edition="quran-uthmani")
    """

    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self._base_url = base_url or get_settings().quran_api_base_url
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "QuranApiClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _get(self, path: str) -> dict:
        """Make a GET request with retry logic."""
        logger.debug(f"API GET: {path}")
        response = await self._client.get(path)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != 200:
            raise ValueError(f"API returned non-200 code: {data.get('code')} — {data.get('status')}")

        return data["data"]

    # ───────────────────────────────────────────────────────
    # Surah Endpoints
    # ───────────────────────────────────────────────────────

    async def get_surah(self, surah_number: int, edition: str = "quran-uthmani") -> QuranSurah:
        """
        Fetch a complete surah with all ayahs.
        
        Args:
            surah_number: 1-114
            edition: Edition identifier (default: quran-uthmani for Arabic Uthmani script)
        
        Returns:
            QuranSurah with all ayahs populated
        """
        data = await self._get(f"/surah/{surah_number}/{edition}")
        surah = QuranSurah(**data)
        if surah.edition is None:
            surah.edition = self._infer_edition_metadata(edition)
        return surah

    async def get_surah_list(self) -> list[dict]:
        """Fetch the list of all 114 surahs with metadata."""
        data = await self._get("/surah/")
        return data

    async def get_multiple_surahs(
        self,
        surah_numbers: list[int],
        edition: str = "quran-uthmani",
    ) -> list[QuranSurah]:
        """Fetch multiple surahs (for batch ingestion)."""
        surahs = []
        for num in surah_numbers:
            try:
                surah = await self.get_surah(num, edition)
                surahs.append(surah)
                logger.info(
                    f"Fetched surah {num}: {surah.english_name} "
                    f"({surah.number_of_ayahs} ayahs)"
                )
            except Exception as e:
                logger.error("Failed to fetch surah {}: {}", num, e)
                raise
        return surahs

    async def get_complete_quran(self, edition: str = "quran-uthmani") -> list[QuranSurah]:
        """
        Fetch the entire Quran in a single request when the backend supports it.

        Quran Hub exposes `/quran/{editionIdentifier}` which returns a `surahs`
        array containing the full corpus.  This is the preferred ingestion path
        for Colab because it avoids 114 sequential network round-trips.
        """
        data = await self._get(f"/quran/{edition}")
        surahs_data = data.get("surahs", [])
        inferred_edition = self._infer_edition_metadata(edition)

        surahs = []
        for surah_data in surahs_data:
            surah = QuranSurah(**surah_data)
            surah.edition = surah.edition or inferred_edition
            surahs.append(surah)

        logger.info(
            "Fetched complete Quran for edition {} with {} surahs",
            edition,
            len(surahs),
        )
        return surahs

    # ───────────────────────────────────────────────────────
    # Ayah Endpoints
    # ───────────────────────────────────────────────────────

    async def get_ayah(
        self,
        reference: str,
        edition: str = "quran-uthmani",
    ) -> dict:
        """
        Fetch a single ayah.
        
        Args:
            reference: Can be "2:255" (surah:ayah) or global number "262"
            edition: Edition identifier
        """
        return await self._get(f"/ayah/{reference}/{edition}")

    async def get_ayah_multiple_editions(
        self,
        reference: str,
        editions: list[str],
    ) -> list[dict]:
        """Fetch same ayah across multiple editions (for translations)."""
        edition_str = ",".join(editions)
        return await self._get(f"/ayah/{reference}/editions/{edition_str}")

    # ───────────────────────────────────────────────────────
    # Edition Endpoints
    # ───────────────────────────────────────────────────────

    async def get_editions(
        self,
        language: Optional[str] = None,
        edition_type: Optional[str] = None,
        format_type: Optional[str] = None,
    ) -> list[dict]:
        """
        List available editions with optional filters.
        
        Args:
            language: "ar", "en", etc.
            edition_type: "quran", "translation", "tafsir"
            format_type: "text", "audio"
        """
        params = {}
        if language:
            params["language"] = language
        if edition_type:
            params["type"] = edition_type
        if format_type:
            params["format"] = format_type

        # Build path with query params
        path = "/edition/"
        if params:
            query_str = "&".join(f"{k}={v}" for k, v in params.items())
            path = f"{path}?{query_str}"

        return await self._get(path)

    async def get_tafsir_editions(self, language: str = "ar") -> list[dict]:
        """Get all available tafsir editions for a language."""
        return await self.get_editions(language=language, edition_type="tafsir")

    # ───────────────────────────────────────────────────────
    # Juz / Page / Search
    # ───────────────────────────────────────────────────────

    async def get_juz(self, juz_number: int, edition: str = "quran-uthmani") -> dict:
        """Fetch a complete juz (1-30)."""
        return await self._get(f"/juz/{juz_number}/{edition}")

    async def search(self, keyword: str, edition: str = "quran-uthmani") -> dict:
        """Search the Quran text for a keyword."""
        return await self._get(f"/search/{keyword}")

    @staticmethod
    def _infer_edition_metadata(edition_identifier: str) -> QuranEdition:
        """
        Build a minimal edition object when the response does not include one.

        Quran Hub's full-Quran endpoint returns a `surahs` array without
        repeating edition metadata on every surah.  We only need enough
        metadata for chunk creation and language-aware retrieval.
        """
        known_tafsir_suffixes = (".tabari", ".muyassar", ".mukhtasar", ".saddi")
        known_quran_identifiers = {"quran-uthmani", "quran-simple", "quran-simple-clean"}

        if edition_identifier in known_quran_identifiers or edition_identifier.startswith("quran-"):
            language = "ar"
            edition_type = "quran"
        elif edition_identifier.endswith(known_tafsir_suffixes):
            language = edition_identifier.split(".", 1)[0]
            edition_type = "tafsir"
        elif "." in edition_identifier:
            language = edition_identifier.split(".", 1)[0]
            edition_type = "translation"
        else:
            language = "ar"
            edition_type = "quran"

        return QuranEdition(
            identifier=edition_identifier,
            language=language,
            name=edition_identifier,
            englishName=edition_identifier,
            format="text",
            type=edition_type,
            direction="rtl" if language == "ar" else "ltr",
        )
