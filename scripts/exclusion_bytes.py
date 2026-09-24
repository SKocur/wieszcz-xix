"""Byte size of the excluded set, recovered from the exclusion list and the cleaned corpus.

The exclusion report records which documents the rule removed and under which arm, but not
how much text that came to. The paper states the figure, so it needs a file behind it:
this sums the cleaned documents named by the list, which is the same quantity the exclusion
percentage of the freeze is computed from.

    python scripts/exclusion_bytes.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXCLUSIONS = REPO / "exclusions/exclusions_2026-08-03.json"
CLEAN = REPO / "data/clean"
OUT = REPO / "metrics/exclusion_bytes_2026-08-03.json"


def main() -> None:
    ids = json.loads(EXCLUSIONS.read_text())["ids"]
    total = missing = 0
    for doc in ids:
        path = CLEAN / f"{doc}.txt"
        if path.exists():
            total += path.stat().st_size
        else:
            missing += 1
    if missing:
        raise SystemExit(f"{missing} of {len(ids)} excluded documents are not in {CLEAN}, "
                         "so the total would be a partial sum")
    report = {"meta": {"script": "scripts/exclusion_bytes.py",
                       "exclusions": EXCLUSIONS.name, "corpus": str(CLEAN.relative_to(REPO))},
              "documents": len(ids), "bytes": total}
    OUT.write_text(json.dumps(report, indent=1))
    print(f"{len(ids):,} documents, {total:,} bytes -> {OUT.name}")


if __name__ == "__main__":
    main()
