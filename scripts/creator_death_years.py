"""Count Internet Archive documents whose catalogue creator died too recently for life+70.

Publication before 1919 does not put a text in the public domain where the term is the
author's life plus 70 years. The catalogue records pulled by `ia_catalogue_metadata.py`
give life dates for part of the creators ("Staff, Leopold, 1878-1957"). This reads them
and reports, for the documents of the provenance ledger:

- how many name a creator at all, and how many of those carry a death year;
- how many name a creator who died in or after the first year still protected
  (`--as-of` minus 70, since the term runs to the end of the calendar year);
- how many name a creator with a birth year and no death year, which the catalogue
  leaves undecided;
- the same split by catalogue type, because the creator of a periodical issue is an
  editor or illustrator and the issue has many authors the record does not list.

The count is a floor. A creator without dates, an author named only in the text and
every contributor to a periodical are outside it. The identifiers of the documents found
are written to the report so a release can act on them.

    python scripts/creator_death_years.py --out metrics/creator_death_years_2026-10-06.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import socket
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from corpus_composition import type_of

REPO = Path(__file__).resolve().parent.parent

LIFE = re.compile(r"(?<!\d)(1[5-9]\d\d)\s*[-–—]\s*(1[6-9]\d\d|20[0-2]\d)(?!\d)")
BIRTH_ONLY = re.compile(r"(?<!\d)(1[5-9]\d\d)\s*[-–—]\s*(?!\d)")
TERM_YEARS = 70


def creators(row: dict) -> list[str]:
    value = row.get("creator")
    if not value:
        return []
    return [value] if isinstance(value, str) else [str(v) for v in value]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metrics/creator_death_years.json")
    ap.add_argument("--ledger", default="ledger/provenance_ledger_2026-08-03_*.csv.gz")
    ap.add_argument("--catalogue", default="metrics/ia_catalogue_metadata_2026-10-06.jsonl.gz")
    ap.add_argument("--as-of", type=int, default=datetime.now(timezone.utc).year)
    args = ap.parse_args()
    first_protected = args.as_of - TERM_YEARS

    size: dict[str, int] = {}
    for path in sorted(REPO.glob(args.ledger)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["source"] == "internet_archive":
                    size[r["document_id"]] = int(r["bytes"])
    total_bytes = sum(size.values())

    counts: Counter = Counter()
    nbytes: Counter = Counter()
    by_type: dict[str, Counter] = {}
    protected: list[dict] = []
    people: Counter = Counter()
    with gzip.open(REPO / args.catalogue, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc = row["document_id"]
            if doc not in size:
                continue
            names = creators(row)
            deaths = [(n, int(m.group(2))) for n in names if (m := LIFE.search(n))]
            late = [(n, y) for n, y in deaths if y >= first_protected]
            open_ended = [n for n in names if not LIFE.search(n) and BIRTH_ONLY.search(n)]
            kind = type_of(row.get("source"))
            tags = ["documents"]
            if names:
                tags.append("with_creator")
            if deaths:
                tags.append("with_death_year")
            if late:
                tags.append("creator_died_in_term")
            elif open_ended:
                tags.append("birth_year_only")
            for tag in tags:
                counts[tag] += 1
                nbytes[tag] += size[doc]
                by_type.setdefault(kind, Counter())[tag] += 1
            if late:
                for n, _ in late:
                    people[n] += 1
                protected.append({"document_id": doc, "type": kind, "bytes": size[doc],
                                  "title": row.get("title"), "year": row.get("year"),
                                  "creators": [f"{n} [{y}]" for n, y in late]})

    pct = lambda a, b: round(100 * a / b, 3) if b else None  # noqa: E731
    out = {
        "meta": {
            "script": "scripts/creator_death_years.py",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "ledger": args.ledger, "catalogue": args.catalogue,
            "as_of": args.as_of, "term_years": TERM_YEARS,
            "first_protected_death_year": first_protected,
        },
        "ledger_documents": len(size),
        "catalogue_documents": counts["documents"],
        "counts": dict(counts),
        "bytes": dict(nbytes),
        "share_documents": {k: pct(v, counts["documents"]) for k, v in counts.items()},
        "share_bytes": {k: pct(v, total_bytes) for k, v in nbytes.items()},
        "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "creators_died_in_term": dict(people.most_common()),
        "documents_creator_died_in_term": sorted(protected, key=lambda d: d["document_id"]),
    }
    path = REPO / args.out
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("ledger_documents", "catalogue_documents", "counts",
                                          "share_documents", "share_bytes", "by_type")},
                     ensure_ascii=False, indent=1))
    print(f"{len(people)} distinct creator strings; wrote {path}")


if __name__ == "__main__":
    main()
