"""
quran_rag/chunker.py

Converts raw Quranic data (fetched from QuranicHub API) into
LlamaIndex Documents with a 3-level parent-child hierarchy:

    Theme → Ayah range → Tafsir chunk

This module is used at ingestion time (notebooks / scripts).
The retriever (retriever.py) does NOT import this module.

Public API:
    build_documents(theme_entry, fetcher, editions, surah_info) → List[Document]
    ingest_themes_from_json(file_path, fetcher, editions)       → List[Document]
"""
from __future__ import annotations

import asyncio
import gc
import json
import re
import uuid as _uuid_mod
from typing import Dict, List, Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from .data_fetcher import QuranicDataFetcher
from .normalizer import normalize_arabic

# ── Constants ────────────────────────────────────────────────────────────────

MAX_CHARS_PARENT = 7_000   # parent node char limit (full theme summary)
MAX_CHARS_CHILD = 1_600    # child node char limit (single tafsir slice)

# Grouped-tafsir editions where multiple ayahs share one commentary block
_GROUPED_TAFSIR_EDITIONS = {"ar.saddi", "ar.ibnkathir"}

# Concurrency guard — prevents hammering QuranicHub
_SEM = asyncio.Semaphore(5)

# ── Sentence splitter for tafsir text ───────────────────────────────────────

_tafsir_splitter = SentenceSplitter(
    chunk_size=350,
    chunk_overlap=50,
    paragraph_separator="\n\n",
    separator=" ",
    chunking_tokenizer_fn=lambda text: re.split(r"(?<=[.،؟!\n])\s+", text),
)

# ── Text utilities ───────────────────────────────────────────────────────────


def clean_repetitive_text(text: str) -> str:
    """Drop near-duplicate sentences — common in some digital manuscripts."""
    if not text:
        return ""
    seen: set = set()
    parts: list = []
    for part in re.split(r"(?<=[.،؟!\n])\s+", text):
        part = part.strip()
        if part and part[:60] not in seen:
            seen.add(part[:60])
            parts.append(part)
    return " ".join(parts)


def truncate_to_char_limit(text: str, max_chars: int) -> str:
    """Hard-truncate at the nearest sentence or word boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = max(cut.rfind("،"), cut.rfind("."), cut.rfind("؟"), cut.rfind("!"), cut.rfind("\n"))
    if boundary > max_chars // 2:
        return cut[: boundary + 1].strip()
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > max_chars // 2 else cut).strip()


# ── LlamaIndex Document helpers ─────────────────────────────────────────────


def _ayah_anchor(surah: int, ayah_from: int, ayah_to: int, ayah_text: str) -> str:
    """Contextual header prepended to every child node for grounding."""
    rng = str(ayah_from) if ayah_from == ayah_to else f"{ayah_from}-{ayah_to}"
    header = f"[سورة {surah}، الآية {rng}]\n"
    if ayah_text:
        header += f"[نص الآية]: {truncate_to_char_limit(ayah_text, 200)}\n"
    return header


def _make_doc(
    text: str,
    meta: Dict,
    *,
    is_parent: bool,
    chunk_id: str,
    parent_chunk_id: str = "",
) -> Document:
    """
    Build a LlamaIndex Document with a deterministic stable UUID.
    Re-runs upsert rather than duplicating, thanks to uuid5.
    """
    stable_id = str(_uuid_mod.uuid5(_uuid_mod.NAMESPACE_DNS, chunk_id))
    doc = Document(
        text=text,
        id_=stable_id,
        metadata={**meta, "is_parent": is_parent, "chunk_id": chunk_id, "parent_chunk_id": parent_chunk_id},
    )
    # These metadata keys are excluded from embedding context
    # (they're kept in payload for filtering only)
    doc.excluded_embed_metadata_keys = [
        "theme", "ayah_text", "surah", "ayah_range", "ayah_from", "ayah_to",
        "edition", "is_parent", "surah_name_arabic", "surah_name_english",
        "surah_name_translation", "revelation_type", "juz", "page", "ruku",
        "hizb_quarter", "chunk_id", "parent_chunk_id", "total_ayahs_in_theme",
    ]
    return doc


# ── Theme document builder ───────────────────────────────────────────────────


async def build_documents(
    theme_entry: Dict,
    fetcher: QuranicDataFetcher,
    editions: List[str],
    surah_info: Optional[Dict] = None,
) -> List[Document]:
    """
    Build all LlamaIndex Documents for one theme entry.

    Each edition produces:
      - 1 parent document  (full ayah range + tafsir summary, ≤ MAX_CHARS_PARENT)
      - N child documents  (individual tafsir slices, ≤ MAX_CHARS_CHILD each)

    Args:
        theme_entry:  One entry from Quran-Themes.json BookContent list.
        fetcher:      QuranicDataFetcher instance (pre-warmed meta cache).
        editions:     List of tafsir edition codes, e.g. ["ar.jalalayn", "ar.ibnkathir"].
        surah_info:   Pre-fetched surah metadata dict (from fetch_surah_map).
    """
    surah = theme_entry["surah_number"]
    theme_start = theme_entry["ayah_from"]
    theme_end = theme_entry["ayah_to"]
    theme_desc = theme_entry.get("content", "")
    total_ayahs = theme_end - theme_start + 1

    si = surah_info or {}
    ayah_numbers = list(range(theme_start, theme_end + 1))

    # ── Fetch all required texts concurrently ────────────────────────────────

    async def _fetch(ayah: int, edition: str):
        return ayah, edition, await fetcher.fetch_ayah_range_async(surah, ayah, ayah, edition)

    fetch_tasks = []
    for ayah in ayah_numbers:
        fetch_tasks.append(_fetch(ayah, "quran-simple"))   # Uthmani text
        for ed in editions:
            fetch_tasks.append(_fetch(ayah, ed))

    raw = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    ayah_data: Dict[int, Dict[str, Optional[Dict]]] = {a: {} for a in ayah_numbers}
    for item in raw:
        if isinstance(item, Exception):
            continue
        ayah_num, edition, result = item
        ayah_data[ayah_num][edition] = None if isinstance(result, Exception) else result

    # ── Structural metadata from first Uthmani ayah ──────────────────────────

    structural: Dict = {}
    first = ayah_data.get(theme_start, {}).get("quran-simple")
    if first:
        structural = {
            "juz": first.get("juz"),
            "page": first.get("page"),
            "ruku": first.get("ruku"),
            "hizb_quarter": first.get("hizbQuarter"),
        }

    theme_base_meta = {
        "surah": surah,
        "ayah_from": theme_start,
        "ayah_to": theme_end,
        "ayah_range": f"{theme_start}-{theme_end}",
        "theme": theme_desc,
        "surah_name_arabic": si.get("name_arabic", ""),
        "surah_name_english": si.get("name_english", ""),
        "surah_name_translation": si.get("name_translation", ""),
        "revelation_type": si.get("revelation_type", ""),
        "total_ayahs_in_theme": total_ayahs,
        **structural,
    }

    docs: List[Document] = []
    theme_loc_key = f"S{surah}_A{theme_start}-{theme_end}"

    # ── Per-edition document building ─────────────────────────────────────────

    for edition in editions:
        ayah_texts: Dict[int, str] = {}
        tafsir_texts: Dict[int, str] = {}

        for ayah in ayah_numbers:
            q = ayah_data[ayah].get("quran-simple")
            if q:
                ayah_texts[ayah] = normalize_arabic(q.get("full_text", ""))
            t = ayah_data[ayah].get(edition)
            if t:
                tafsir_texts[ayah] = clean_repetitive_text(normalize_arabic(t.get("full_text", "")))

        if not any(tafsir_texts.values()):
            continue

        ed_key = f"{theme_loc_key}_{edition}"

        # Parent document — full range summary
        dynamic_limit = max(300, MAX_CHARS_PARENT // max(1, len(ayah_numbers)))
        parent_parts: List[str] = []
        for ayah in ayah_numbers:
            q_text = ayah_texts.get(ayah, "")
            t_text = tafsir_texts.get(ayah, "")
            anchor = _ayah_anchor(surah, ayah, ayah, q_text)
            excerpt = truncate_to_char_limit(t_text, dynamic_limit)
            if excerpt:
                parent_parts.append(anchor + f"[التفسير]:\n{excerpt}")

        parent_text = truncate_to_char_limit("\n\n".join(parent_parts), MAX_CHARS_PARENT)
        if parent_text.strip():
            parent_meta = {**theme_base_meta, "edition": edition, "ayah_text": ""}
            docs.append(_make_doc(parent_text, parent_meta, is_parent=True, chunk_id=f"{ed_key}_parent"))

        # Child documents — individual tafsir slices
        is_grouped = edition in _GROUPED_TAFSIR_EDITIONS or (
            len(tafsir_texts.get(theme_start, "")) > 500
            and any(len(tafsir_texts.get(a, "")) == 0 for a in ayah_numbers[1:])
        )

        if is_grouped:
            combined = " ".join(
                tafsir_texts.get(a, "") for a in ayah_numbers if tafsir_texts.get(a, "")
            ).strip()
            if not combined:
                continue

            nodes = _tafsir_splitter.get_nodes_from_documents([Document(text=combined)])
            for i, node in enumerate(nodes):
                slice_text = node.get_content().strip()
                if not slice_text:
                    continue
                baseline_q = ayah_texts.get(theme_start, "")
                anchor = _ayah_anchor(surah, theme_start, theme_end, baseline_q)
                final = truncate_to_char_limit(anchor + slice_text, MAX_CHARS_CHILD)
                child_meta = {
                    **theme_base_meta,
                    "edition": edition,
                    "ayah_from": theme_start,
                    "ayah_to": theme_end,
                    "ayah_range": f"{theme_start}-{theme_end}",
                    "ayah_text": baseline_q,
                }
                docs.append(_make_doc(
                    final, child_meta, is_parent=False,
                    chunk_id=f"{ed_key}_range_child{i}",
                    parent_chunk_id=f"{ed_key}_parent",
                ))
        else:
            for ayah in ayah_numbers:
                q_text = ayah_texts.get(ayah, "")
                t_text = tafsir_texts.get(ayah, "")
                if not t_text:
                    continue

                nodes = _tafsir_splitter.get_nodes_from_documents([Document(text=t_text)])
                for i, node in enumerate(nodes):
                    slice_text = node.get_content().strip()
                    if not slice_text:
                        continue
                    anchor = _ayah_anchor(surah, ayah, ayah, q_text)
                    final = truncate_to_char_limit(anchor + slice_text, MAX_CHARS_CHILD)
                    child_meta = {
                        **theme_base_meta,
                        "edition": edition,
                        "ayah_from": ayah,
                        "ayah_to": ayah,
                        "ayah_range": str(ayah),
                        "ayah_text": q_text,
                    }
                    docs.append(_make_doc(
                        final, child_meta, is_parent=False,
                        chunk_id=f"{ed_key}_ayah{ayah}_child{i}",
                        parent_chunk_id=f"{ed_key}_parent",
                    ))

    return docs


async def _process_theme_safe(
    theme_entry: Dict,
    fetcher: QuranicDataFetcher,
    editions: List[str],
    surah_info: Optional[Dict] = None,
) -> List[Document]:
    """Semaphore-guarded wrapper to limit concurrent QuranicHub requests."""
    async with _SEM:
        try:
            return await build_documents(theme_entry, fetcher, editions, surah_info)
        except Exception as exc:
            s = theme_entry.get("surah_number")
            a = f"{theme_entry.get('ayah_from')}-{theme_entry.get('ayah_to')}"
            print(f"[CHUNKER ERROR] Surah {s}, Ayah {a}: {type(exc).__name__}: {exc}")
            return []


async def ingest_themes_from_json(
    file_path: str,
    fetcher: QuranicDataFetcher,
    editions: List[str],
    surah_map: Optional[Dict[int, Dict]] = None,
    target_surahs: Optional[List[int]] = None,
) -> List[Document]:
    """
    Main ingestion entry point.

    Reads Quran-Themes.json, processes each theme entry concurrently,
    and returns the full list of LlamaIndex Documents ready for indexing.

    Args:
        file_path:     Path to Quran-Themes.json.
        fetcher:       QuranicDataFetcher (pre-warm meta cache first).
        editions:      Tafsir edition codes, e.g. ["ar.jalalayn", "ar.ibnkathir"].
        surah_map:     Output of fetch_surah_map() — pass to avoid re-fetching.
        target_surahs: Optional list of surah numbers to process (subset ingestion).
    """
    try:
        from tqdm.asyncio import tqdm as atqdm
    except ImportError:
        atqdm = None

    with open(file_path, "r", encoding="utf-8") as f:
        themes_list = json.load(f).get("BookContent", [])

    if target_surahs:
        themes_list = [t for t in themes_list if t.get("surah_number") in target_surahs]

    # Pre-warm meta cache once before all concurrent tasks
    if not object.__getattribute__(fetcher, "_meta_cache"):
        await fetcher.fetch_quran_meta()

    tasks = [
        _process_theme_safe(
            entry,
            fetcher,
            editions,
            (surah_map or {}).get(int(entry["surah_number"]), {}),
        )
        for entry in themes_list
    ]

    nested: List[List[Document]] = []
    if atqdm is not None:
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Themes"):
            nested.append(await coro)
    else:
        nested = await asyncio.gather(*tasks)

    docs = [d for sublist in nested for d in sublist]

    parents = sum(1 for d in docs if d.metadata.get("is_parent"))
    children = len(docs) - parents
    print(f"\nIngestion complete: {len(docs)} docs — {parents} parents, {children} children")
    gc.collect()
    return docs
