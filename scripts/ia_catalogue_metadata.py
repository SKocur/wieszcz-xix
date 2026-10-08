"""Pull the Internet Archive catalogue record of every Internet Archive document.

The provenance ledger keeps identifier, source and size. The catalogue record holds what
a composition table needs: the item's date, its type (`source`, "czasopismo" for a
periodical issue), its place of publication (`coverage`), subjects, publisher, creator,
the uploading library (`contributor`) and the library's rights statement (`rights`).
Every one of these is a claim by the cataloguing library. They describe the corpus as
catalogued, and a hand-checked sample is what validates them.

Two passes. The Scrape API returns the whole crawl query pool in pages of 5000 with the
requested fields, and rows whose identifier is in the ledger are kept. Ledger documents
the pool no longer returns (an item whose language or year field changed after the
crawl, or one taken down) are then asked for one by one through the metadata API.

Rows append to a JSONL as they arrive and the scrape cursor is saved after each page, so
an interrupted pull resumes. At the end the JSONL is compressed and a summary JSON
records the query, the fields and how many documents carry each field.

    python scripts/ia_catalogue_metadata.py --out metrics/ia_catalogue_metadata_2026-10-06
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parent.parent
SCRAPE = "https://archive.org/services/search/v1/scrape"
ITEM = "https://archive.org/metadata/{}/metadata"
# The crawl's query, as in src/prepare_data.py.
QUERY = ("(language:(pol) OR language:(Polish)) "
         "AND year:[1800 TO 1918] AND mediatype:texts")
FIELDS = ["identifier", "date", "year", "title", "creator", "publisher", "subject",
          "coverage", "source", "rights", "contributor", "collection", "language",
          "licenseurl", "possible-copyright-status"]
HEADERS = {"User-Agent": "wieszcz-xix catalogue metadata"}


def get_json(url: str, timeout: int) -> dict:
    for attempt in range(1, 6):
        try:
            with urlopen(Request(url, headers=HEADERS), timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001, transient API noise, retried
            if attempt == 5:
                return {"error": str(e)}
            time.sleep(5 * attempt)
    return {}


def ledger_identifiers(pattern: str) -> dict[str, str]:
    """identifier -> document_id for the Internet Archive rows of the ledger."""
    out: dict[str, str] = {}
    for path in sorted(REPO.glob(pattern)):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["source"] == "internet_archive":
                    out[r["source_identifier"]] = r["document_id"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="metrics/ia_catalogue_metadata",
                    help="output stem; writes <stem>.jsonl.gz and <stem>.json")
    ap.add_argument("--ledger", default="ledger/provenance_ledger_2026-08-03_*.csv.gz")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    wanted = ledger_identifiers(args.ledger)
    stem = REPO / args.out
    jsonl = stem.with_name(stem.name + ".jsonl")
    cursor_file = stem.with_name(stem.name + ".cursor")

    done: set[str] = set()
    if jsonl.exists():
        with jsonl.open(encoding="utf-8") as f:
            done = {json.loads(line)["identifier"] for line in f}
    print(f"{len(wanted):,} ledger documents, {len(done):,} already pulled", flush=True)

    t0 = time.time()
    cursor = cursor_file.read_text().strip() if cursor_file.exists() else None
    scrape_finished = cursor == "END"
    pool_total = None
    with jsonl.open("a", encoding="utf-8") as sink:
        page = 0
        while not scrape_finished:
            # `count` is left out: sent with a cursor it makes the API repeat page one.
            params = {"q": QUERY, "fields": ",".join(FIELDS)}
            if cursor:
                params["cursor"] = cursor
            resp = get_json(f"{SCRAPE}?{urlencode(params)}", timeout=180)
            if "items" not in resp:
                raise SystemExit(f"scrape failed, rerun to resume: {resp}")
            pool_total = resp.get("total", pool_total)
            for item in resp["items"]:
                ident = item["identifier"]
                if ident in wanted and ident not in done:
                    sink.write(json.dumps({"document_id": wanted[ident], **item,
                                           "via": "scrape"}, ensure_ascii=False) + "\n")
                    done.add(ident)
            sink.flush()
            cursor = resp.get("cursor")
            cursor_file.write_text(cursor or "END")
            scrape_finished = cursor is None
            page += 1
            print(f"page {page}: {len(done):,}/{len(wanted):,} matched, "
                  f"pool {pool_total:,}, {time.time()-t0:4.0f}s", flush=True)

        missing = sorted(set(wanted) - done)
        print(f"{len(missing):,} documents outside the pool, asking per item", flush=True)
        errors: list[str] = []

        def work(ident: str) -> dict | None:
            meta = get_json(ITEM.format(ident), timeout=45).get("result")
            if not isinstance(meta, dict):
                errors.append(wanted[ident])
                return None
            row = {k: meta[k] for k in FIELDS if k in meta}
            return {"document_id": wanted[ident], **row, "identifier": ident,
                    "via": "item"}

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, row in enumerate(ex.map(work, missing), 1):
                if row is not None:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    done.add(row["identifier"])
                if i % 500 == 0:
                    sink.flush()
                    print(f"item {i:,}/{len(missing):,} {time.time()-t0:4.0f}s", flush=True)

    coverage = dict.fromkeys(FIELDS, 0)
    via = {"scrape": 0, "item": 0}
    gz = stem.with_name(stem.name + ".jsonl.gz")
    with jsonl.open(encoding="utf-8") as src, gzip.open(gz, "wt", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            via[row["via"]] += 1
            for k in FIELDS:
                if row.get(k) not in (None, "", []):
                    coverage[k] += 1
            dst.write(line)

    summary = {
        "meta": {
            "script": "scripts/ia_catalogue_metadata.py",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "query": QUERY,
            "fields": FIELDS,
            "ledger": args.ledger,
        },
        "ledger_documents": len(wanted),
        "pulled": len(done),
        "via": via,
        "not_retrieved": sorted(wanted[i] for i in set(wanted) - done),
        "field_coverage": coverage,
        "rows": gz.name,
        "rows_sha256": hashlib.sha256(gz.read_bytes()).hexdigest(),
    }
    stem.with_name(stem.name + ".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if not summary["not_retrieved"]:
        jsonl.unlink()
        cursor_file.unlink(missing_ok=True)
    print(f"pulled {len(done):,}/{len(wanted):,} ({via}), "
          f"{len(summary['not_retrieved'])} not retrieved")
    print("field coverage:", coverage)
    print(f"wrote {gz} and {stem.name}.json")


if __name__ == "__main__":
    main()
