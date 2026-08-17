"""Rebuild the provenance ledger for the frozen build, one file per token bin.

The release builder maps the Nth document in a token stream to the Nth row of a ledger,
which is only safe if the two were made for each other. The ledger it used was written for
the previous build: 219,958 rows against a stream of 294,369 documents. Pointed at the
current bins it would have raised IndexError, and left pointed at the previous bin it
would have produced a byte-perfect release of the wrong corpus with its self-check
passing, because that bin and that ledger agree with one another.

So a ledger is written per bin rather than one for the pair. A single file for both would
have carried an implicit rule about which stream its first 291,605 rows belong to, and an
implicit rule is what caused this.

Document order is the ids' own sorted order, which is how the corpus was tokenised: the
terminator counts in both bins match the split's document counts exactly, and the id lists
are already in byte order.

Sizes are what the *tokeniser* saw, not what the file on disk holds, and for Wolne Lektury
those differ: tokenisation drops the digitisation colophon, so seven held-out documents
decode 23 bytes shorter than their cleaned files -- an `ISBN 978-83-288-...` line each. The
colophon pass is imported from the same module tokenize_corpus uses rather than
reimplemented, because a second copy of that rule is exactly how the two would drift apart
again.

    python scripts/build_provenance_ledger.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from clean_ocr import is_anachronism

SOURCES = {"ia": "internet_archive", "wl": "wolne_lektury"}
FIELDS = ("document_id", "source", "source_identifier", "bytes")


def tokenised_text(doc_id: str, clean: Path) -> str:
    """The bytes the tokeniser was given, which for wl_ is not the file on disk."""
    text = (clean / f"{doc_id}.txt").read_text(encoding="utf-8", errors="replace")
    if doc_id.startswith("wl_"):
        text = "\n".join(ln for ln in text.split("\n") if not is_anachronism(ln))
    return text


def row_for(doc_id: str, clean: Path) -> dict:
    prefix, _, identifier = doc_id.partition("_")
    source = SOURCES.get(prefix)
    if source is None:
        raise SystemExit(f"unknown source prefix in {doc_id!r}")
    return {"document_id": doc_id, "source": source, "source_identifier": identifier,
            "bytes": len(tokenised_text(doc_id, clean).encode("utf-8"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="metrics/doc_split_2026-08-03.json")
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--label", default="2026-08-03")
    ap.add_argument("--outdir", default="metrics")
    args = ap.parse_args()

    split = json.loads((REPO / args.split).read_text(encoding="utf-8"))
    clean = REPO / args.clean
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    written = {}
    for part, key in (("train", "train_ids"), ("val", "val_ids")):
        ids = list(split[key])
        if ids != sorted(ids):
            raise SystemExit(f"{key} is not in byte order; the bin's document order "
                             f"cannot be assumed to match it")
        out = outdir / f"provenance_ledger_{args.label}_{part}.csv.gz"
        total = 0
        with gzip.open(out, "wt", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for doc_id in ids:
                row = row_for(doc_id, clean)
                total += row["bytes"]
                w.writerow(row)
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        written[part] = {"file": out.name, "documents": len(ids),
                         "text_bytes": total, "sha256": digest}
        print(f"{part:>5}: {len(ids):>7,} documents, {total/1e9:6.2f} GB of text -> {out.name}")

    report = {
        "script": "scripts/build_provenance_ledger.py",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "split": args.split, "clean": args.clean,
        "documents_total": sum(v["documents"] for v in written.values()),
        "parts": written,
    }
    rp = outdir / f"provenance_ledger_{args.label}.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n{report['documents_total']:,} documents total -> {rp}")


if __name__ == "__main__":
    main()
