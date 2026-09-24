"""Check Internet Archive collection membership for every non-library document.

The rights review covered the Polish digital-library uploads (explicit per-item
public-domain assertions) and sampled the rest. The 2026-08-02 content audit then found
a 1972 book among the unsampled remainder: an `internetarchivebooks` lending-library
scan whose metadata claims 1900. Those collections are where in-copyright modern
editions live, and no content battery is guaranteed to catch a modern book that happens
to avoid modern vocabulary, so membership itself is the filter: every document whose id
does not carry a Polish-library prefix is checked against the IA metadata API and
classified by collection.

Classes: `ia_modern_scan` (inlibrary / printdisabled / internetarchivebooks; treat as
in-copyright candidates and exclude from redistribution), `community_upload`
(opensource, rights unasserted), `google_books` and `institutional_old` (pre-copyright
library scans), `other`. Items whose own date field parses past 1918 are additionally
listed, whatever their collection.

Results append to a JSONL as they arrive and a rerun skips documents already present,
so an interrupted sweep resumes instead of restarting; the summary JSON is compacted
from the JSONL at the end. A previous sweep's summary seeds the cache (`--seed`), so
auditing a frozen corpus directory (`--txt-dir`) only queries identifiers the earlier
sweep never saw, since the API is asked about each identifier once, ever.

    python scripts/ia_collection_sweep.py                       # ledger of the HF build
    python scripts/ia_collection_sweep.py --txt-dir data/clean \
        --seed metrics/ia_collection_sweep.json \
        --out metrics/ia_collection_sweep_2026-08-03.json
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "metrics/provenance_ledger.csv.gz"
PL_LIBRARY = re.compile(r"^ia_[a-z0-9.-]+\.pl\.")
MODERN_SCAN = {"inlibrary", "printdisabled", "internetarchivebooks"}
OLD_SCAN = {"americana", "toronto", "unclibraries", "europeanlibraries",
            "cdl", "medicineinamericas", "blc", "getty"}


def fetch(identifier: str) -> dict:
    req = Request(f"https://archive.org/metadata/{identifier}",
                  headers={"User-Agent": "wieszcz-xix rights sweep"})
    for attempt in (1, 2, 3):
        try:
            with urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001, transient API noise, retried
            if attempt == 3:
                return {"error": str(e)}
            time.sleep(2 * attempt)
    return {}


def classify(collections: list[str], identifier: str) -> str:
    cset = set(collections)
    if cset & MODERN_SCAN:
        return "ia_modern_scan"
    if "opensource" in cset:
        return "community_upload"
    if "googlebooks" in cset or identifier.endswith("goog"):
        return "google_books"
    if cset & OLD_SCAN:
        return "institutional_old"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metrics/ia_collection_sweep.json")
    ap.add_argument("--txt-dir", default=None,
                    help="take identifiers from a corpus directory of ia_*.txt files "
                         "instead of the provenance ledger")
    ap.add_argument("--seed", default=None,
                    help="summary JSON of a previous sweep; its documents are "
                         "treated as already checked")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    if args.txt_dir:
        targets = [(f.stem, f.stem[3:]) for f in sorted(Path(args.txt_dir).glob("ia_*.txt"))
                   if not PL_LIBRARY.match(f.stem)]
    else:
        rows = list(csv.DictReader(gzip.open(LEDGER, "rt")))
        targets = [(r["document_id"], r["source_identifier"]) for r in rows
                   if r["source"] == "internet_archive"
                   and not PL_LIBRARY.match(r["document_id"])]

    docs: dict[str, dict] = {}
    if args.seed:
        docs.update(json.load(open(REPO / args.seed, encoding="utf-8"))["docs"])
    jsonl = REPO / (args.out + "l")
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            docs[rec.pop("doc_id")] = rec
    targets = [t for t in targets if t[0] not in docs]
    print(f"{len(targets):,} non-library documents to check "
          f"({len(docs):,} already done)", flush=True)

    errors: list[str] = []
    t0 = time.time()

    def work(item: tuple[str, str]) -> dict | None:
        doc_id, ident = item
        meta = fetch(ident).get("metadata")
        if meta is None:
            errors.append(doc_id)
            return None
        coll = meta.get("collection", [])
        coll = [coll] if isinstance(coll, str) else coll
        date = str(meta.get("date") or meta.get("year") or "")
        m = re.search(r"\d{4}", date)
        return {
            "doc_id": doc_id,
            "identifier": ident,
            "collections": coll,
            "date": date,
            "class": classify(coll, ident),
            "date_post1918": bool(m and int(m.group()) > 1918),
        }

    with jsonl.open("a", encoding="utf-8") as sink, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, rec in enumerate(ex.map(work, targets), 1):
            if rec is not None:
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                docs[rec.pop("doc_id")] = rec
            if i % 250 == 0:
                sink.flush()
                print(f"{i:,}/{len(targets):,}  {time.time()-t0:5.0f}s", flush=True)

    by_class: dict[str, int] = {}
    for d in docs.values():
        by_class[d["class"]] = by_class.get(d["class"], 0) + 1

    out = REPO / args.out
    out.write_text(json.dumps({
        "checked": len(docs), "errors": errors, "by_class": by_class,
        "date_post1918": sum(d["date_post1918"] for d in docs.values()),
        "docs": docs,
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nchecked {len(docs):,}, errors {len(errors)}")
    print("by class:", by_class)
    print("metadata date > 1918:", sum(d["date_post1918"] for d in docs.values()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
