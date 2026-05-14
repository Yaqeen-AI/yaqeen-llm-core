"""
One-time script: add 'mazhabs' field to every existing Qdrant point.

Run once after ingest.py has already populated the collection:
    python enrich_payloads.py

Uses set_payload — no re-embedding, no API calls. Pure local operation.
"""

import pathlib
import sys

# Ensure the project root is on sys.path so `core.*` imports resolve
# regardless of which directory the script is run from.
_PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import SetPayload, Filter

from core.config import COLLECTION_NAME, QDRANT_PATH
from core.arabic_utils import detect_mazhabs, detect_fiqh_topic

BATCH_SIZE = 256


def main() -> None:
    client = QdrantClient(path=QDRANT_PATH)

    if not client.collection_exists(COLLECTION_NAME):
        sys.exit(f"Collection '{COLLECTION_NAME}' not found — run ingest.py first.")

    for field in ("mazhabs", "fiqh_topic"):
        try:
            from qdrant_client.models import PayloadSchemaType
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    info = client.get_collection(COLLECTION_NAME)
    total = info.points_count
    print(f"Enriching {total:,} points in '{COLLECTION_NAME}' with mazhab + fiqh_topic tags...\n")

    offset = None
    processed = 0
    mazhab_counts: dict[str, int] = {}

    with tqdm(total=total, unit="pts") as bar:
        while True:
            records, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not records:
                break

            for rec in records:
                text    = rec.payload.get("chunk_text", "")
                mazhabs = detect_mazhabs(text)
                _t      = detect_fiqh_topic(text)
                topic   = _t[0] if _t and len(_t) == 1 else ""

                client.set_payload(
                    collection_name=COLLECTION_NAME,
                    payload={"mazhabs": mazhabs, "fiqh_topic": topic},
                    points=[rec.id],
                )

                for m in mazhabs:
                    mazhab_counts[m] = mazhab_counts.get(m, 0) + 1

            processed += len(records)
            bar.update(len(records))

            if next_offset is None:
                break
            offset = next_offset

    print(f"\nDone — {processed:,} points updated.")
    print("\nMazhab mention counts:")
    for name, count in sorted(mazhab_counts.items(), key=lambda x: -x[1]):
        pct = count / processed * 100
        print(f"  {name:10s}  {count:6,}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
