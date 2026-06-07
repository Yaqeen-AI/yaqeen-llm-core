#!/usr/bin/env python3
"""
clean_ayah_themes.py — Deduplicate ayah-themes.db
==================================================
Reads the source ayah-themes.db, removes exact duplicate rows,
and writes a clean ayah-themes-clean.db.

Usage:
    python clean_ayah_themes.py
"""

import sqlite3
import sys
from pathlib import Path


def main():
    src = Path("ayah-themes.db")
    dst = Path("ayah-themes-clean.db")

    if not src.exists():
        print(f"ERROR: Source database '{src}' not found.")
        sys.exit(1)

    if dst.exists():
        dst.unlink()
        print(f"Removed existing '{dst}'.")

    # ── Read source ──
    conn_src = sqlite3.connect(src)
    cur_src = conn_src.cursor()

    cur_src.execute("SELECT COUNT(*) FROM themes")
    total_rows = cur_src.fetchone()[0]

    cur_src.execute(
        "SELECT DISTINCT theme, surah_number, ayah_from, ayah_to, keywords, total_ayahs "
        "FROM themes ORDER BY surah_number, ayah_from"
    )
    distinct_rows = cur_src.fetchall()
    conn_src.close()

    duplicates_removed = total_rows - len(distinct_rows)

    # ── Write clean DB ──
    conn_dst = sqlite3.connect(dst)
    cur_dst = conn_dst.cursor()

    cur_dst.execute("""
        CREATE TABLE themes (
            theme       TEXT,
            surah_number INTEGER,
            ayah_from   INTEGER,
            ayah_to     INTEGER,
            keywords    TEXT,
            total_ayahs INTEGER
        )
    """)

    cur_dst.executemany(
        "INSERT INTO themes (theme, surah_number, ayah_from, ayah_to, keywords, total_ayahs) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        distinct_rows,
    )

    # Add an index for common query patterns
    cur_dst.execute("CREATE INDEX idx_themes_surah ON themes(surah_number, ayah_from)")

    conn_dst.commit()

    # ── Verify ──
    cur_dst.execute("SELECT COUNT(*) FROM themes")
    clean_count = cur_dst.fetchone()[0]

    cur_dst.execute("SELECT COUNT(DISTINCT theme) FROM themes")
    distinct_themes = cur_dst.fetchone()[0]

    cur_dst.execute("SELECT MIN(total_ayahs), MAX(total_ayahs), AVG(total_ayahs) FROM themes")
    min_a, max_a, avg_a = cur_dst.fetchone()

    conn_dst.close()

    # ── Report ──
    print("=" * 50)
    print("ayah-themes.db Cleanup Report")
    print("=" * 50)
    print(f"  Source rows:       {total_rows}")
    print(f"  Clean rows:        {clean_count}")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Distinct themes:   {distinct_themes}")
    print(f"  Ayahs range:       {min_a} – {max_a} (avg {avg_a:.1f})")
    print(f"  Output:            {dst}")
    print("=" * 50)


if __name__ == "__main__":
    main()
